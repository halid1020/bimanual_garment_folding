import numpy as np
from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from actoris_harena.torch_utils import *

# TODO: make it as follow
#from actoris_harena.torch_utils import *

from .rssm import RSSM
from .model import bottle, symlog

def reward_bonus_and_penalty(rewards, observations, actions):
    if isinstance(rewards, torch.Tensor):
        rewards_ = rewards.clone()
    else:
        rewards_ = rewards.copy()

    above_0_9 = observations['normalised_coverage'][:, :-1] > 0.9
    below_0_9 = observations['normalised_coverage'][:, 1:] < 0.9
    first_state_no_term = observations['terminal'][:, :-1] == 0

    rewards_[:, 1:][above_0_9 & below_0_9 & first_state_no_term] = 0

    above_0_9_5 = observations['normalised_coverage'][:] > 0.95
    rewards_[above_0_9_5] = 0.7


    return rewards

class GoalConditionedTransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, 
                 embedding_size, activation_function='relu', min_std_dev=0.1, embedding_layers=1, state_layers=1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.rnn = nn.GRUCell(belief_size, belief_size)
        
        self.fc_embed_state_action = self.make_layers(state_size+action_size, belief_size, hidden_size, embedding_layers)
        self.fc_embed_belief_prior = self.make_layers(belief_size+embedding_size, hidden_size, hidden_size, embedding_layers)
        self.fc_state_prior = self.make_layers(hidden_size, 2 * state_size, hidden_size, state_layers)
        self.fc_embed_belief_posterior = self.make_layers(belief_size + embedding_size*2, hidden_size, hidden_size, embedding_layers)
        self.fc_state_posterior = self.make_layers(hidden_size, 2 * state_size, hidden_size, state_layers)

    def make_layers(self, input_dim, output_dim, hidden_dim, num_layers):

        if num_layers == 1:
            return nn.Linear(input_dim, output_dim)

        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_dim))

        return nn.Sequential(*layers)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, 
                prev_belief:torch.Tensor, goal_observations:torch.Tensor, 
                observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = \
             [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        
        
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
            _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
  
            # Compute belief (deterministic hidden state)
            # print('!state shape', _state.shape)
            # print('!action shape', actions[t].shape)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(torch.cat([beliefs[t + 1], goal_observations[t]], dim=1)))
            #print('hidden shape:', hidden.shape)
            #print('goal_observations shape:', goal_observations[t+1].shape)
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(
                        torch.cat([beliefs[t + 1], goal_observations[t_+1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden
    
class GC_RSSM(RSSM):

    def __init__(self, config):
        super().__init__(config)
        self.reward_processor = reward_bonus_and_penalty

    def init_transition_model(self):
        self.model['transition_model'] = GoalConditionedTransitionModel(
            belief_size=self.config.deterministic_latent_dim,
            state_size=self.config.stochastic_latent_dim,
            action_size = np.prod(np.array(self.config.action_dim)), 
            hidden_size=self.config.hidden_dim,
            embedding_size=self.config.embedding_dim,
            activation_function=self.config.activation,
            min_std_dev=self.config.min_std_dev,
            embedding_layers=self.config.trans_layers,
            state_layers=self.config.get('state_layers', 1)
        ).to(self.config.device)
    
    # def _update_helper(self, info, action, reset_internal=False):
    #     pass

    def _update_helper(self, info, action, reset_internal=False):
        arena_id = info['arena_id']
        obs = info['observation']
        mask = obs['mask']
        self.no_op = self.no_op.flatten()
        obs_ = obs[self.config.input_obs]
        goal_obs_ = obs[f'goal-{self.config.input_obs}']
        goal_mask = obs['goal-mask']


        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        
        if len(goal_mask.shape) == 2:
            goal_mask = np.expand_dims(goal_mask, axis=0)

        
        to_trans_dict = {
            self.config.input_obs: np.expand_dims(obs_, axis=(0)),
            f'goal-{self.config.input_obs}': np.expand_dims(goal_obs_, axis=(0)),
            'mask': np.expand_dims(mask, axis=(0)),
            'goal-mask': np.expand_dims(goal_mask, axis=(0)),
        }
        res = self.data_augmenter(
            to_trans_dict, 
            train=False)
        image = res[self.config.input_obs].to(self.config.device)
        goal_image = res[f'goal-{self.config.input_obs}'].to(self.config.device)
       
        
        image = symlog(image, self.symlog)
        if len(image.shape) == 4:
            image = image.unsqueeze(0)
        goal_image = symlog(goal_image, self.symlog)
        if len(goal_image.shape) == 4:
            goal_image = goal_image.unsqueeze(0)

        
        if reset_internal:
            self.cur_state[arena_id] = {
                'deter': torch.zeros(
                    1, self.config.deterministic_latent_dim, 
                    device=self.config.device),
                'stoch': {
                    'sample': torch.zeros(
                        1, self.config.stochastic_latent_dim, 
                        device=self.config.device)
                },
                'goal_obs': goal_image, # batch*horizon*C*H*W
                'input_obs': image # batch*horizon*C*H*W
            }
        else:
            self.cur_state[arena_id]['input_obs'] = image
            self.cur_state[arena_id]['goal_obs'] = goal_image
        
    
        action = np_to_ts(action.flatten(), self.config.device).unsqueeze(0).unsqueeze(0)

        latent_state,  _ = self.unroll_state_action(self.cur_state[arena_id] , action)
        self.cur_state[arena_id]['deter'] = latent_state['deter'][-1]
        self.cur_state[arena_id]['stoch']['sample'] = latent_state['stoch']['sample'][-1]
        

        ## get the last state for
        if self.config.debug:
            if self.config.input_obs == 'rgb':
                rgb = ts_to_np(image[0, 0, :3, :, :].clip(0, 1)).transpose(1, 2, 0)
                goal = ts_to_np(goal_image[0, 0, :3, :, :].clip(0, 1)).transpose(1, 2, 0)
                plt.imsave(f'tmp/input_obs_rgb.png', rgb) 
                plt.imsave(f'tmp/input_obs_goal_rgb.png', goal)

        self.internal_states[arena_id] = {
            'raw_input_obs': obs_.copy(),
            'input_obs': image.squeeze(0).squeeze(0) \
                .cpu().detach().numpy().transpose(1, 2, 0),
            'deter_state': self.cur_state[arena_id]['deter']\
                .squeeze(1).cpu().detach().numpy(),
            'stoch_state': self.cur_state[arena_id]['stoch']['sample']\
                .squeeze(1).cpu().detach().numpy(),
            'latent_state': self.cur_state[arena_id],
            'input_type': self.config.input_obs,
            'posterior_reward': self.model['reward_model'](self.cur_state[arena_id]['deter'], self.cur_state[arena_id]['stoch']['sample'])
                .squeeze(0).cpu().detach().item(),
            'recon_obs': self.model['observation_model'](self.cur_state[arena_id]['deter'], self.cur_state[arena_id]['stoch']['sample']).cpu().detach()
        }


    def _unroll_action(self, actions, belief_, latent_state_, goal_obs):

        goal_emb = self.model['encoder'](goal_obs)
        img_beliefs_, prior_states_, prior_means_, prior_std_devs_  = \
            self.model['transition_model'](
                latent_state_, 
                actions, 
                belief_,
                goal_emb,
                None,
                None)

        priors_ = {
            'sample': prior_states_,
            'mean': prior_means_,
            'std': prior_std_devs_
        }
        
        return img_beliefs_, priors_
    
    def _unroll_state_action(self, data, horizon=None, batch_size=None):

         # Create initial belief and state for time t = 0
        init_belief = torch.zeros(
            self.config.batch_size if batch_size is None else batch_size,
            self.config.deterministic_latent_dim).to(self.config.device)

        init_state = torch.zeros(
            self.config.batch_size if batch_size is None else batch_size,
            self.config.stochastic_latent_dim).to(self.config.device)

        

        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        if horizon == None:
            horizon = len(data['action'])

        actions = data['action'][:horizon]
        non_terminals = (1 - data['terminal'])[:horizon].unsqueeze(-1)
        #rewards = data['reward']
        input_obs = data['input_obs'][:horizon]
        goal_obs = data['goal_obs'][:horizon]
        #output_obs = data['output_obs']


        obs_emb = {}
        obs_emb['emb'] = self.model['encoder'](input_obs)
        obs_emb['goal-emb'] = self.model['encoder'](goal_obs)


        blfs, prior_states_, prior_means_, prior_std_devs_, posterior_states_, posterior_means_, posterior_std_devs_ = \
            self.model['transition_model'](
                init_state, 
                actions,
                init_belief,
                obs_emb['goal-emb'],
                obs_emb['emb'],
                non_terminals)

        posteriors_ = {
            'sample': posterior_states_,
            'mean': posterior_means_,
            'std': posterior_std_devs_
        }

        priors_ = {
            'sample': prior_states_,
            'mean': prior_means_,
            'std': prior_std_devs_
        }
        
        return blfs, posteriors_, priors_, obs_emb
    
    def unroll_state_action(self, state, action):

        ## print all the shapes
        # print('action shape:', action.shape)
        # print('state stoch sample shape:', state['stoch']['sample'].shape)
        # print('state deter shape:', state['deter'].shape)
        # print('state input obs shape:', state['input_obs'].shape)
        # print('state goal obs shape:', state['goal_obs'].shape)


        blfs, prior_states_, prior_means_, prior_std_devs_, posterior_states_, posterior_means_, posterior_std_devs_ = \
            self.model['transition_model'](
                state['stoch']['sample'],
                action, 
                state['deter'], 
                self.model['encoder'](state['goal_obs']),
                self.model['encoder'](state['input_obs']),
                None)

        posteriors_ = {
            'sample': posterior_states_,
            'mean': posterior_means_,
            'std': posterior_std_devs_
        }

        last_post = {
            'deter': blfs[-1],
            'stoch': {
                'sample': posterior_states_[-1],
                'mean': posterior_means_[-1],
                'std': posterior_std_devs_[-1]
            }
            

        }

        return {
            'deter': blfs,
            'stoch': posteriors_
        }, last_post
    
    def unroll_action_from_cur_state(self, action, state_):

        to_unroll = {}
        candidates, horizons = action.shape[:2]
        action = action.reshape(candidates, horizons, -1)
        state= self.cur_state[state_['arena_id']]
        # #state = self.cur_state
        to_unroll['deter'] = state['deter']\
                .squeeze(dim=1).expand(1, candidates, self.config.deterministic_latent_dim)\
                .reshape(-1, self.config.deterministic_latent_dim)
        
        to_unroll['stoch'] = {
            'sample': state['stoch']['sample']\
                .squeeze(dim=1).expand(1, candidates, self.config.stochastic_latent_dim)\
                .reshape(-1, self.config.stochastic_latent_dim)
        }

        action = np_to_ts(action, self.config.device).permute(1, 0, 2) ## horizon*candidates*actions

        return self.unroll_action(to_unroll, action, state['goal_obs'])

    def unroll_action(self, init_state, actions, goal_obs):
        goal_emb = self.model['encoder'](goal_obs) # in shape 1 * 1 * 10124
        # action in shape 1 * batch * action_dim

        # make the first two dimensions of goal_emb and actions the same
        goal_emb = goal_emb.expand(actions.shape[0], actions.shape[1], -1)
        # print('goal_emb', goal_emb[0, :2, :5])
        
        # print('goal_emb shape:', goal_emb.shape)
        # print('actions shape:', actions.shape)

        img_beliefs_, prior_states_, prior_means_, prior_std_devs_  = \
            self.model['transition_model'](
                init_state['stoch']['sample'], 
                actions, 
                init_state['deter'], 
                goal_emb,
                None,
                None)
        
        return {
            'deter': img_beliefs_,
            'stoch': {
                'sample': prior_states_,
                'mean': prior_means_,
                'std': prior_std_devs_
            }
        }
    


    
    def unscaled_overshooting_losses(self, experience, beliefs, posteriors):
        if self.config.kl_overshooting_scale == 0:
            return torch.tensor(0).to(self.config.device), torch.tensor(0).to(self.config.device)

        actions = experience['action']
        non_terminals = 1 - experience['terminal']
        rewards = experience['reward']
        goal_emb = self.model['encoder'](experience['goal_obs'][1:])


        #overshooting_vars = self._prepare_overshooting_vars(experience)
        
        overshooting_vars = [] 
        for t in range(1, self.config.sequence_size - 1):
            d = min(t + self.config.overshooting_distance, self.config.sequence_size - 1)  # Overshooting distance
            t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
            seq_pad = (0, 0, 0, 0, 0, t - d + self.config.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch

            # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) posterior states, 
            # (5) posterior means, (6) posterior standard deviations and (7) sequence masks
            # (8) goal embeddings
            overshooting_vars.append((
                F.pad(actions[t:d], seq_pad), 
                F.pad(non_terminals[t:d].unsqueeze(2), seq_pad), 
                F.pad(rewards[t:d], seq_pad[2:]), 
                beliefs[t_], 
                posteriors['sample'][t_].detach(), 
                F.pad(posteriors['mean'][t_ + 1:d_ + 1].detach(), seq_pad), 
                F.pad(posteriors['std'][t_ + 1:d_ + 1].detach(), seq_pad, value=1), 
                F.pad(torch.ones(d - t, self.config.batch_size, self.config.stochastic_latent_dim, device=self.config.device), seq_pad),
                F.pad(goal_emb[t_ + 1:d_ + 1].detach(), seq_pad)
            ))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
        

        overshooting_vars = tuple(zip(*overshooting_vars))
        

        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs = self.model['transition_model'](
            torch.cat(overshooting_vars[4], dim=0), 
            torch.cat(overshooting_vars[0], dim=1), 
            torch.cat(overshooting_vars[3], dim=0),
            torch.cat(overshooting_vars[8], dim=1),
            None, 
            torch.cat(overshooting_vars[1], dim=1))

        reward_seq_mask = torch.cat(overshooting_vars[7], dim=1)
        
        

        # Calculate overshooting KL loss with sequence mask

        posteriors = {
            'mean': torch.cat(overshooting_vars[5], dim=1), 
            'std': torch.cat(overshooting_vars[6], dim=1)}
        
        priors = {
            'mean': prior_means, 
            'std': prior_std_devs}

        kl_overshooting_loss  = self.compute_kl_loss(
            posteriors, priors, 
            self.config.kl_overshooting_balance, 
            free=self.config.free_nats)


        if self.config.reward_overshooting_scale != 0:
           
            if self.config.reward_gradient_stop:
                reward_overshooting_loss = F.mse_loss(
                    bottle(self.model['reward_model'],
                    (beliefs.detach(), prior_states.detach())) * reward_seq_mask[:, :, 0], 
                    torch.cat(overshooting_vars[2], dim=1), reduction='none').mean()
            else:
                reward_overshooting_loss = F.mse_loss(
                        bottle(self.model['reward_model'], 
                        (beliefs, prior_states)) * reward_seq_mask[:, :, 0], 
                        torch.cat(overshooting_vars[2], dim=1), 
                        reduction='none').mean()

        else:
            reward_overshooting_loss = torch.tensor(0).to(self.config.device)

        
        
        return kl_overshooting_loss, reward_overshooting_loss
    
    def _preprocess(self, data, train=False, single=False, apply_transform=True):

        if self.config.datasets[0].name in ['default', 'prioritised', 'count-based']:
            if 'observation' in data:
                obs_data = data['observation']
                act_data = data['action']['default']
                if self.apply_reward_processor:
                    #print('apply reward processor')
                    rewards = self.reward_processor(data['observation']['reward'], obs_data, act_data)
                else:
                    rewards = data['observation']['reward']
            else:
                obs_data = data
                act_data = data['action']
                if self.apply_reward_processor:
                    #print('apply reward processor')
                    rewards = self.reward_processor(data['reward'], obs_data, act_data)
                else:
                    rewards = data['reward']
            
           
           
            data = {
                'action': act_data,
            }
            data.update(obs_data)
            
            if single:
                data['reward'] = rewards[1:].squeeze(-1)
                data['terminal'] = data['terminal'][1:].squeeze(-1)
            else:
                data['reward'] = rewards[:, 1:].squeeze(-1)
                data['terminal'] = data['terminal'][:, 1:].squeeze(-1)
        
        # Apply transformations
        if apply_transform:
            if single and not self.apply_transform_in_dataset:
                for k, v in data.items():
                    data[k] = np.expand_dims(v, 0)
            data = self.data_augmenter(data, train=train)
            if self.apply_transform_in_dataset:
                for k, v in data.items():
                    data[k] = v.unsqueeze(0)
            for k, v in data.items():
                data[k] = v.to(self.config.device)

        # Swap axes for all items in data
        for k in data:
            data[k] = torch.swapaxes(data[k], 0, 1)

        # Precompute shapes if needed
        if self.config.input_obs == 'rgbd':
            T, B, C, H, W = data['rgb'].shape
            # Concatenate and apply symlog transformation
            rgbd = torch.cat([data['rgb'], data['depth']], dim=2)
            goal_rgbd = torch.cat([data['goal-rgb'], data['goal-depth']], dim=2)
            data['input_obs'] = symlog(rgbd, self.symlog)
            data['goal_obs'] = symlog(data['goal-rgb'], self.symlog)
       
        else:
            data['input_obs'] = symlog(data[self.config.input_obs], self.symlog)
            data['goal_obs'] = symlog(data[f'goal-{self.config.input_obs}'], self.symlog)

        # Determine output observation based on configuration
        if self.config.output_obs == 'input_obs':
            data['output_obs'] = data['input_obs']
        else:
            data['output_obs'] = symlog(data[self.config.output_obs], self.symlog)

        
        data['reward'] = symlog(data['reward'], self.symlog)

        return data