# DAC subclass implementing a practical Demonstration Actor-Critic variant
from typing import Sequence, Tuple

class DAC(VanillaSAC):
    """
    Demonstration Actor-Critic (practical variant).
    - Maintains a separate demo replay buffer.
    - During training, samples mixed batches (demo + env).
    - Adds a behaviour-cloning loss on demo samples (policy-dependent shaping).
    - Optionally adds a small demo reward bonus in critic target.
    """

    def __init__(self, config):
        super().__init__(config)
        # demo buffer hyperparams (tunable)
        self.demo_capacity = config.get('demo_capacity', 100_000)
        self.demo_batch_ratio = config.get('demo_batch_ratio', 0.5)  # fraction of batch from demos
        self.demo_bc_coef = config.get('demo_bc_coef', 1.0)  # BC loss weight
        self.demo_reward_scale = config.get('demo_reward_scale', 0.0)  # additive shaping for demo transitions
        # create demo replay (keeps same interface as main replay)
        self.demo_replay = ReplayBuffer(self.demo_capacity, (config.state_dim,), self.replay_action_dim, self.device)

    # ---------------- demo loading utilities ----------------
    def load_demonstrations_from_arrays(self,
                                        obs: np.ndarray,
                                        actions: np.ndarray,
                                        rewards: Optional[np.ndarray] = None,
                                        next_obs: Optional[np.ndarray] = None,
                                        dones: Optional[np.ndarray] = None):
        """
        Load demonstrations into the demo replay.
        Expected shapes:
            obs: (N, obs_dim)
            actions: (N, action_dim)
            rewards: (N,) optional, defaults to zeros
            next_obs: (N, obs_dim) optional (if missing we shift obs by 1)
            dones: (N,) optional
        """
        N = obs.shape[0]
        rewards = np.zeros((N,), dtype=np.float32) if rewards is None else rewards.astype(np.float32)
        if next_obs is None:
            # naive shift: copy obs[1:] as next_obs, last next as same
            next_obs = np.vstack([obs[1:], obs[-1:]]).astype(np.float32)
        if dones is None:
            dones = np.zeros((N,), dtype=np.float32)

        for i in range(N):
            self.demo_replay.add(obs[i].astype(np.float32),
                                 actions[i].astype(np.float32),
                                 float(rewards[i]),
                                 next_obs[i].astype(np.float32),
                                 bool(dones[i]))

    # ---------------- batch sampling (mixed demo + env) ----------------
    def _sample_mixed_batch(self, batch_size: int):
        """
        Returns one combined batch dict with same keys as ReplayBuffer.sample output:
            observation, action, reward, next_observation, done
        Demo fraction determined by self.demo_batch_ratio.
        If demo buffer is empty, returns regular env batch.
        """
        if self.demo_replay.size == 0:
            return self.replay.sample(batch_size)

        demo_count = int(round(batch_size * self.demo_batch_ratio))
        demo_count = min(demo_count, self.demo_replay.size)
        env_count = batch_size - demo_count
        # sample env part
        env_batch = self.replay.sample(env_count) if env_count > 0 else None
        demo_batch = self.demo_replay.sample(demo_count) if demo_count > 0 else None

        # If either part is None, return the other directly
        if env_batch is None:
            return demo_batch
        if demo_batch is None:
            return env_batch

        # concatenate
        concat = {}
        # each is a torch tensor already on device
        concat['observation'] = torch.cat([env_batch['observation'], demo_batch['observation']], dim=0)
        concat['action'] = torch.cat([env_batch['action'], demo_batch['action']], dim=0)
        concat['reward'] = torch.cat([env_batch['reward'], demo_batch['reward']], dim=0)
        concat['next_observation'] = torch.cat([env_batch['next_observation'], demo_batch['next_observation']], dim=0)
        concat['done'] = torch.cat([env_batch['done'], demo_batch['done']], dim=0)

        # We will need to know which indices correspond to demos for BC loss -- return mask
        demo_mask = torch.cat([torch.zeros(env_count, dtype=torch.bool, device=self.device),
                               torch.ones(demo_count, dtype=torch.bool, device=self.device)], dim=0)

        concat['demo_mask'] = demo_mask
        return concat

    # ---------------- override network update to include demo-guidance ----------------
    def _update_networks(self, batch: dict):
        """
        Same core algorithm as VanillaSAC but:
         - Sample mixed_batch using new _sample_mixed_batch interface (handled by train loop).
         - Compute BC loss on demo subset and add to actor loss.
         - Optionally add small reward bonus for demo transitions in the critic target.
        """
        config = self.config
        # Accept either full mixed batch or single-source batch
        # Keep backward compat: VanillaSAC._update_networks expected batch.values() order:
        # context, action, reward, next_context, done = batch.values()
        # But here we accept that train() now provides the mixed batch dict created by _sample_mixed_batch.
        if 'demo_mask' in batch:
            demo_mask = batch.pop('demo_mask')
        else:
            demo_mask = None

        context = batch['observation'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_context = batch['next_observation'].to(self.device)
        done = batch['done'].to(self.device)

        B = context.shape[0]

        # alpha loss (same as Vanilla)
        if self.auto_alpha_learning:
            pi_tmp, log_pi_tmp = self.actor.sample(context)
            alpha = self.log_alpha.exp().detach().item()
            alpha_loss = -(self.log_alpha * (log_pi_tmp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        else:
            alpha = self.init_alpha

        # compute target Q with optional demo reward shaping: for demo indices add demo_reward_scale
        with torch.no_grad():
            a_next, logp_next = self.actor.sample(next_context)
            q1_next, q2_next = self.critic_target(next_context, a_next)
            q_next = torch.min(q1_next, q2_next)

            # if demo_reward_scale > 0, add shaping to reward for demo transitions.
            if (self.demo_reward_scale != 0.0) and (demo_mask is not None):
                # demo_mask shape (B,), reward shape (B,1)
                # create a shaping term added to reward for demo transitions
                shaping = torch.zeros_like(reward)
                shaping[demo_mask] = self.demo_reward_scale
                target_q = (reward + shaping) + (1 - done) * config.gamma * (q_next - alpha * logp_next)
            else:
                target_q = reward + (1 - done) * config.gamma * (q_next - alpha * logp_next)

        # current Q estimates
        a_curr = action.view(B, -1).to(self.device)
        q1_pred, q2_pred = self.critic(context, a_curr)
        critic_loss = 0.5*(F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        # optimize critics
        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip_value)
        self.critic_optim.step()

        # actor loss (base SAC actor loss)
        pi, log_pi = self.actor.sample(context)
        q1_pi, q2_pi = self.critic(context, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_pi - min_q_pi).mean()

        # ------ add BC loss on demo subset (policy-dependent shaping) ------
        if (demo_mask is not None) and (demo_mask.any()):
            # Extract demo subset
            demo_inds = demo_mask.nonzero(as_tuple=False).squeeze(-1)
            demo_obs = context[demo_inds]
            demo_actions = action[demo_inds]

            # Compute actor mean for demo observations (no sampling noise)
            mean_demo, _ = self.actor(demo_obs)  # mean is pre-tanh in your Actor
            # deterministic policy for comparison uses tanh(mean)
            mean_demo_tanh = torch.tanh(mean_demo)

            # Behavior cloning loss (MSE) between policy mean and demonstrated action
            bc_loss = F.mse_loss(mean_demo_tanh, demo_actions)
            actor_loss = actor_loss + self.demo_bc_coef * bc_loss
        else:
            bc_loss = torch.tensor(0.0, device=self.device)

        # optimize actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft updates
        if self.update_steps % self.config.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, config.tau)

        # logging (expand VanillaSAC logging with BC loss and demo counts)
        with torch.no_grad():
            q_stats = {
                'q_mean': min_q_pi.mean().item(),
                'q_max': min_q_pi.max().item(),
                'q_min': min_q_pi.min().item(),
                'logp_mean': log_pi.mean().item(),
                'logp_max': log_pi.max().item(),
                'logp_min': log_pi.min().item(),
                'alpha': alpha,
                'critic_grad_norm': critic_grad_norm.item(),
                'bc_loss': bc_loss.item() if isinstance(bc_loss, torch.Tensor) else float(bc_loss)
            }
            self.logger.log({f"diag/{k}": v for k,v in q_stats.items()}, step=self.act_steps)

        # logging
        self.logger.log({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'bc_loss': bc_loss.item() if isinstance(bc_loss, torch.Tensor) else float(bc_loss),
        }, step=self.act_steps)

        if self.auto_alpha_learning:
            self.logger.log({'alpha_loss': alpha_loss.item()}, step=self.act_steps)

    # ---------------- override train to use mixed sampling ----------------
    def train(self, update_steps, arenas) -> bool:
        # ensure replay buffers exist
        if not self.init_reply:
            self._init_reply_buffer(self.config)

        if arenas is None or len(arenas) == 0:
            raise ValueError("SAC.train requires at least one Arena.")
        arena = arenas[0]
        self.set_train()
        self.last_done = True

        with tqdm(total=update_steps, desc=f"{self.name} (DAC) Training", initial=0) as pbar:
            # first collect initial transitions into env replay (same as VanillaSAC)
            while self.replay.size < self.initial_act_steps:
                self._collect_from_arena(arena)
                pbar.set_postfix(env_step=self.act_steps, updates=self.update_steps)

            for _ in range(update_steps):
                # sample mixed batch here
                batch = self._sample_mixed_batch(self.config.batch_size)
                # optional data augmentation hook
                if hasattr(self, 'data_augmenter') and callable(self.data_augmenter):
                    self.data_augmenter(batch)
                # pass mixed batch to update
                self._update_networks(batch)
                self.update_steps += 1

                if self.update_steps % self.config.gradient_steps == 0:
                    for _ in range(self.config.train_freq):
                        self._collect_from_arena(arena)
                        pbar.set_postfix(
                            phase="training",
                            env_step=self.act_steps,
                            total_updates=self.update_steps,
                        )

                pbar.update(1)
                pbar.set_postfix(env_step=self.act_steps, updates=self.update_steps)

        return True
