import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml
import tools
from controllers.rl.dreamer_v3.dreamer import Dreamer

class DreamerV3Adapter(TrainableAgent):
  
    def __init__(self, config):
        tools.set_seed_everywhere(config.seed)
        if config.deterministic_run:
            tools.enable_deterministic_run()
        
        config.steps //= config.action_repeat
        config.eval_every //= config.action_repeat
        config.log_every //= config.action_repeat
        config.time_limit //= config.action_repeat

        self.logger = WandbLogger() # from agent-arena

        self.replay = # initialise

        self.dreamer  = Dreamer(
            config.observation_space,
            config.action_space,
            config,
            self.logger,
            self.replay,
        ).to(config.device)


    def _collect_from_arena(self, arenas):
        if self.last_done:
            if self.info is not None:
                evaluation = self.info['evaluation']
                success = int(self.info['success'])

                for k, v in evaluation.items():
                    self.logger.log({
                        f"train/eps_lst_step_eval_{k}": v,
                    }, step=self.act_steps) 
                
                self.logger.log({
                    "train/episode_return": self.episode_return,
                    "train/episode_length": self.episode_length,
                    'train/episode_success': success,
                    'train/episode_sim_steps': self.sim_steps - self.last_sim_steps,
                    'train/total_sim_steps': self.sim_steps 
                }, step=self.act_steps)

            self.info = arena.reset()
            self.set_train()
            self.reset([arena.id])
            self.episode_return = 0.0
            self.episode_length = 0
            self.last_done = False
            self.last_sim_steps = self.sim_steps

        # sample stochastic action for exploration
        a, _ = self._select_action(self.info, stochastic=True)
        # clip to action range
        #a = np.clip(a, -self.config.action_range, self.config.action_range)
        #dict_action = {'continuous': a}  # user should adapt to their arena's expected action format
        next_info = arena.step(a)
        if next_info.get("fail_step", False):
            self.last_done = True
            return

        
        
        next_obs_for_process =  self._get_next_obs_for_process(next_info)

        reward = next_info.get('reward', 0.0)[self.reward_key] if isinstance(next_info.get('reward', 0.0), dict) else next_info.get('reward', 0.0)
        self.logger.log(
            {'train/step_reward': reward}, step=self.act_steps
        )
        evaluation = next_info.get('evaluation', {})
        for k, v in evaluation.items():
            self.logger.log({
                f"train/{k}": v,
            }, step=self.act_steps)
        done = next_info.get('done', False)
        self.info = next_info
        #a = next_info['applied_action']
        self.last_done = done

        aid = arena.id
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        obs_for_replay = self._process_context_for_replay(obs_list)
        # print('obs stack', obs_stack)
        # print('next_obs', next_obs)
        # append next
        #print(next_obs_for_process)
        obs_list.append(self._process_obs_for_input(next_obs_for_process))
        next_obs_for_replay = self._process_context_for_replay(obs_list[-self.context_horizon:])
        #next_obs_stack = np.stack(obs_list)[-self.context_horizon:].flatten() #TODO: .reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])

        #print('\napplied action vecotr', a, type(a))
        self._add_transition_replay(obs_for_replay, self._post_process_action_to_replay(a), reward, next_obs_for_replay, done)
        
        
        self.act_steps += 1
        self.sim_steps += next_info['sim_steps']
        self.episode_return += reward
        self.episode_length += 1

    def train(self, update_steps, arenas) -> bool:
        
       
        if not config.offline_traindir:
            prefill = max(0, config.prefill - self.replay.get_action_steps())
            print(f"Prefill dataset ({prefill} steps).")
            if config.action_type ==  "discrete":
                random_actor = tools.OneHotDist(
                    torch.zeros(config.num_actions).repeat(config.envs, 1)
                )
            else:
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.tensor(acts.low).repeat(config.envs, 1),
                        torch.tensor(acts.high).repeat(config.envs, 1),
                    ),
                    1,
                )

            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None

            state = tools.simulate(
                random_agent,
                arenas,
                self.replay,
                # config.traindir,
                self.logger,
                limit=config.dataset_size,
                steps=prefill,
            )
        
        self.dreamer.requires_grad_(requires_grad=False)

            
        state = tools.simulate(
            self.dreamer,
            arenas,
            self.replay,
            #config.traindir,
            self.logger,
            limit=self.config.dataset_size,
            steps=update_steps,
            state=state,
        )


    def load(self):
        if (self.save_dir / "latest.pt").exists():
            checkpoint = torch.load(logdir / "latest.pt")
            self.dreamer.load_state_dict(checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(self.dreamer, checkpoint["optims_state_dict"])
            self.dreamer._should_pretrain._once = False

            