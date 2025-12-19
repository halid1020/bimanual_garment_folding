
import pathlib

from .tools import *
from controllers.rl.dreamer_v3.dreamer import Dreamer
from agent_arena import TrainableAgent
from gym.spaces import Dict, Box
import gym
import numpy as np
from tqdm import tqdm

class DreamerV3Adapter(TrainableAgent):
  
    def __init__(self, config):
        super().__init__(config)
        set_seed_everywhere(config.seed)
        if config.deterministic_run:
            enable_deterministic_run()
        self.config = config
        #self.config.action_steps //= config.action_repeat
        self.loaded_model = False
        self.loaded_dataset = False
        self.initialised_agent = False
        self.primitive_integration = config.get('primitive_integration', 'none')
        if self.primitive_integration == 'predict_bin_as_output':
            self.primitives = self.config.primitives
            self.K = len(self.primitives)
            self.action_dims = [prim['dim'] if isinstance(prim, dict) else prim.dim for prim in self.primitives]
            self.config.num_actions = max(self.action_dims) + 1
            self.prim_name2id = {item['name']: i for i, item in enumerate(self.primitives)}
            print('[dreamer] num actions', self.config.num_actions)
        
        self.random_prefill = config.get('random_prefill', 2500)
        self.use_bc_policy_to_seed = config.get('use_bc_policy_to_seed', True)
        self.update_step = 0

        

        if self.use_bc_policy_to_seed:
            from omegaconf import OmegaConf
            import os
            import agent_arena.api as ag_ar
            from train.utils import build_data_augmenter   # adjust import path if needed

            self.bc_policy_config_name = config.bc_policy_config_name
            self.bc_data_augmenter_config_name = config.bc_data_augmenter_config_name

            # --- Load policy config (from conf/agent/<name>.yaml) ---
            bc_policy_config_path = os.path.join("conf", "agent", f"{self.bc_policy_config_name}.yaml")
            self.bc_policy_config = OmegaConf.load(bc_policy_config_path)
            self.bc_policy_config.project_name = config.project_name
            self.bc_policy_config.exp_name = self.bc_policy_config_name

            self.bc_prefill = config.get('bc_prefill', 2500)

            # Build the BC policy
            self.bc_policy = ag_ar.build_agent(self.bc_policy_config.name, self.bc_policy_config)

            # --- Load augmenter config (from conf/data_augmenter/<name>.yaml) ---
            bc_augmenter_config_path = os.path.join("conf", "data_augmenter", f"{self.bc_data_augmenter_config_name}.yaml")
            self.bc_augmenter_config = OmegaConf.load(bc_augmenter_config_path)

            # Build the data augmenter
            augmenter = build_data_augmenter(self.bc_augmenter_config)
            self.bc_policy.set_data_augmenter(augmenter)

    def set_data_augmenter(self, data_augmenter):
        self.data_augmenter = data_augmenter
    
    def set_log_dir(self, log_dir):
        logdir = pathlib.Path(log_dir).expanduser()
        super().set_log_dir(log_dir)
        if self.use_bc_policy_to_seed:
            bc_logger_dir = os.path.join(log_dir, 'bc_policy')
            self.bc_policy.set_log_dir(bc_logger_dir)
        
        self.config.traindir = self.config.traindir or logdir / "train_eps"
        #self.config.evaldir = self.config.evaldir or logdir / "eval_eps"
        
        #self.config.eval_every //= self.config.action_repeat
        self.config.log_every //= self.config.action_repeat
        #self.config.time_limit //= self.config.action_repeat

        print("Logdir", logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        self.config.traindir.mkdir(parents=True, exist_ok=True)
        #self.config.evaldir.mkdir(parents=True, exist_ok=True)

        self.action_step = count_steps(self.config.traindir)
        #self.logger.step = self.config.action_repeat * self.action_step
        if self.config.offline_traindir:
            directory = self.config.offline_traindir.format(**vars(self.config))
        else:
            directory = self.config.traindir
        self.logdir = logdir
        self.train_eps = load_episodes(directory, limit=self.config.dataset_size)

        self._load_dataset()
        self._init_agent()

    def _random_prefill_dataset(self, arenas):
        self.train_state = None
        num_env = len(arenas)
        if not self.config.offline_traindir:
            prefill = max(0, self.random_prefill - count_steps(self.config.traindir))
            print(f"[dreamer] Random prefill dataset ({prefill} steps).")
            #acts = arenas[0].action_space # Future object, that is why
            rnd_act_space = gym.spaces.Box(-1, 1, (self.config.num_actions, ), dtype=np.float32)
            if hasattr(rnd_act_space, "discrete"):
                random_actor = OneHotDist(
                    torch.zeros(self.config.num_actions).repeat(num_env, 1)
                )
            else:
                
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.tensor(rnd_act_space.low).repeat(num_env, 1),
                        torch.tensor(rnd_act_space.high).repeat(num_env, 1),
                    ),
                    1,
                )

            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None

            self.train_state = simulate(
                self,
                random_agent,
                arenas,
                self.train_eps,
                self.config.traindir,
                self.logger,
                limit=self.config.dataset_size,
                steps=prefill,
                parallel=self.config.parallel,
                obs_keys = self.config.obs_keys,
                reward_key = self.config.reward_key,
                save_success = self.config.get('save_success', False)
            )
            # self.logger.step += prefill * self.config.action_repeat
            self.action_step = count_steps(self.config.traindir)
            self.dreamer._action_step = self.action_step
            #print(f"Logger: ({self.logger.step} steps).")
    
    def _use_bc(self):
        flg = self.use_bc_policy_to_seed and self.update_step <= self.bc_policy.total_update_steps
        #print('use bc?', flg)
        return flg    

    def reset(self, arena_ids):
        if self._use_bc():
            return self.bc_policy.reset(arena_ids)
        super().reset(arena_ids)
        
    def _bc_prefill_dataset(self, arenas):
        self.train_state = None
        num_env = len(arenas)
        if not self.config.offline_traindir:
            prefill = max(0, self.random_prefill + self.bc_prefill - count_steps(self.config.traindir))
            print(f"[dreamer] Bahaviour cloning policy prefill dataset ({prefill} steps).")
            #acts = arenas[0].action_space # Future object, that is why

            def bc_agent(o, d, s): ## assume single arena
                
                info = {'observation': o, "arena_id": 0}
                info['observation']['rgb'] = info['observation']['rgb'][-1]
                if d[0]: #reset agent
                    self.bc_policy.reset([0])
                    self.bc_policy.init([info])
                else: # diffsuion does not need the last step action
                    self.bc_policy.update([info], [None])
                action = self.bc_policy.single_act(
                    info
                ) # TODO: this may return the multi primitive action in dictionary
                #print('bc action', action)
                if self.config.primitive_integration == 'predict_bin_as_output':
                    action_name = list(action.keys())[0]
                    action_param = action[action_name]
                    prim_id = self.prim_name2id[action_name]
                    prim_act = (1.0*(prim_id+0.5)/self.K *2 - 1)
                    action = np.zeros((1, self.config.num_actions))
                    action[0, 0] = prim_act
                    action[0, 1:action_param.shape[0]+1] = action_param
                    action = torch.tensor(action)

                #print('bc feed action', action)
                return {"action": action}, None

            self.train_state = simulate(
                self,
                bc_agent,
                arenas,
                self.train_eps,
                self.config.traindir,
                self.logger,
                limit=self.config.dataset_size,
                steps=prefill,
                parallel=self.config.parallel,
                obs_keys = self.config.obs_keys,
                reward_key = self.config.reward_key,
                save_success = self.config.get('save_success', False)
            )
            # self.logger.step += prefill * self.config.action_repeat
            self.action_step = count_steps(self.config.traindir)
            self.dreamer._action_step = self.action_step
            # print(f"Logger: ({self.logger.step} steps).")

    def _load_dataset(self):
        if self.loaded_dataset:
            return
        self.train_dataset = make_dataset(self.train_eps, self.config)
        self.vis_dataset = make_dataset_none_cross_trj(self.train_eps, self.config)
        self.loaded_dataset = True

    def _init_agent(self): # We want to call this in the init function instead
        if self.initialised_agent:
            return
        
        h, w = self.config.size  # self.size = [128, 128]
        self.obs_space = Dict({
            "image": Box(
                low=0,
                high=255,
                shape=(h, w, 3),
                dtype=np.uint8
            )
        })

        rnd_act_space = gym.spaces.Box(-1, 1, (self.config.num_actions, ), dtype=np.float32)
        
        self.dreamer = Dreamer(
            self.obs_space, ## TODO: we need to get rid of this.
            rnd_act_space,
            self.config,
            self.logger,
            self.train_dataset,
            self.vis_dataset
        ).to(self.config.device)
        self.dreamer.set_data_augmenter(self.data_augmenter)
        self.dreamer.requires_grad_(requires_grad=False)
        self.load()
            
        self.initialised_agent = True
    
    def init(self, info_list):
        if self._use_bc():
            self.bc_policy.init(info_list)
            return

        for info in info_list:
            self.internal_states[info['arena_id']] = {}
            self.internal_states[info['arena_id']]['done'] = False
            self.internal_states[info['arena_id']]['agent_state'] = None

    def update(self, info_list, actions):
        if self._use_bc():
            self.bc_policy.update(info_list, actions)
            return

        for info in info_list:
            self.internal_states[info['arena_id']]['done'] = info['done']

    def single_act(self, info, update=False):

        if self._use_bc():
            return self.bc_policy.single_act(info, update=update)


        #print('single act!')
        agent_state = self.internal_states[info['arena_id']]['agent_state']
        obs = info['observation']
        done = info['done']

        obs = {k:v for k, v in obs.items() if k in self.config.obs_keys}

        obs = {k: np.stack([o[k] for o in [obs]]) for k in obs if "log_" not in k}
        done = np.stack([done])
        action, agent_state = self.dreamer(obs, done, agent_state, training=self.training)
        
        #print('action', action)
        self.internal_states[info['arena_id']]['agent_state'] = agent_state

        action = action['action'][0].detach().cpu().numpy()

        if self.primitive_integration == 'none':
            return action
        elif self.primitive_integration == 'predict_bin_as_output':
            prim_idx = int(((action[0] + 1)/2)*self.K - 1e-6)
            action = action[1:]
            prim_name = self.primitives[prim_idx]['name'] if isinstance(self.primitives[prim_idx], dict) else self.primitives[prim_idx].name
            out_dict = {prim_name: action}
            return out_dict
        else:
            raise NotImplementedError
        
    def get_phase(self):
        if self._use_bc():
            return self.bc_policy.get_phase()
        return super().get_phase()
    
    def terminate(self):
        if self._use_bc():
            return self.bc_policy.terminate()
        return super().terminate()
    
    def success(self):
        if self._use_bc():
            return self.bc_policy.success()
        return super().success()

    def train(self, update_steps, arenas) -> bool:
        
        ## If demo learning is required first do the demo
        if self._use_bc():
            
            
            bc_update_steps = min(update_steps, self.bc_policy.total_update_steps-self.bc_policy.update_step)
            #print('[dreamer] train bc', bc_update_steps)
            
            self.bc_policy.train(bc_update_steps, arenas)
            self.update_step += bc_update_steps
            
            update_steps -= bc_update_steps
            self.dreamer._logger.update_step = self.update_step
            if update_steps == 0:
                return True
            
            total_bc_update_steps = self.bc_policy.total_update_steps
        else:
            total_bc_update_steps = 0


        ## Seeding, no update steps
        self._random_prefill_dataset(arenas)
        if self.use_bc_policy_to_seed:
            self._bc_prefill_dataset(arenas)

        ## Pretraining
        #print('deamer update count', self.dreamer._update_count)
        if self.update_step < self.config.pretrain + total_bc_update_steps:
            to_update_steps = min(update_steps, self.config.pretrain + total_bc_update_steps - self.update_step)
            for _ in tqdm(range(to_update_steps), desc="[dreamer] Pretraining ..."):
                self.dreamer._train(next(self.train_dataset))
                #print('train!!!')
                #self.dreamer._update_count += 1 ## only belongs to dreamer
                self.dreamer._logger.update_step += 1 ## combination of bc and dreamer
                
                self.dreamer._metrics["action_step"] = self.action_step
                self.dreamer.log()
                self.update_step += 1
            update_steps -= to_update_steps
            if update_steps == 0:
                return True
        
        ## Online Learning
        action_steps = int(self.dreamer.train_every*update_steps/self.config.updates_per_step)
        target_action_steps = min(self.action_step + action_steps, self.config.total_update_steps)
        action_steps_to_do = target_action_steps - self.action_step
        print('[dreamer] Online update steps to do', update_steps)
        print('[dreamer] Online action steps to do', action_steps_to_do)
        self.logger.write()
        self.train_state = simulate(
            self,
            self.dreamer,
            arenas,
            self.train_eps,
            self.config.traindir,
            self.logger,
            limit=self.config.dataset_size,
            steps=action_steps_to_do,
            state=self.train_state,
            parallel=self.config.parallel,
            restart=self.config.restart,
            obs_keys = self.config.obs_keys,
            reward_key = self.config.reward_key
        )
        self.action_step = count_steps(self.config.traindir)
        print(f'[dreamer] action step {self.action_step} end of current epoch training')
        self.update_step += update_steps
    
    def save(self):
        if self._use_bc():
            print(f'[dreamer] Save bc policy only at update step {self.update_step}')
            self.bc_policy.save()
            return
        print(f'[dreamer] Save dreamer policy also at update step {self.update_step}')
        items_to_save = {
            "agent_state_dict": self.dreamer.state_dict(),
            "optims_state_dict": recursively_collect_optim_state_dict(self.dreamer),
            "update_step": self.update_step
        }
        torch.save(items_to_save, self.logdir / "latest.pt")


    def save_best(self):

        if self._use_bc():
            self.bc_policy.save_best()
            return
        
        items_to_save = {
            "agent_state_dict": self.dreamer.state_dict(),
            "optims_state_dict": recursively_collect_optim_state_dict(self.dreamer),
        }
        torch.save(items_to_save, self.logdir / "best.pt")
        

    def load(self):
        if self.loaded_model:
            return self.update_step
        
        self.update_step= 0
        self.dreamer._logger.update_step = self.update_step
        if (self.logdir / "latest.pt").exists():
            checkpoint = torch.load(self.logdir / "latest.pt")
            self.dreamer.load_state_dict(checkpoint["agent_state_dict"])
            self.update_step = checkpoint['update_step']
            print(f'[dreamer] Loaded dreamer on checkpoint {self.update_step}')
            self.logger.step = self.update_step
            self.dreamer._logger.update_step = self.update_step
            recursively_load_optim_state_dict(self.dreamer, checkpoint["optims_state_dict"])
            self.dreamer._should_pretrain._once = False
        
        if self.use_bc_policy_to_seed:
            self.update_step = max(self.bc_policy.load(), self.update_step)

        self.loaded_model = True
        #print('[dreamer] loaded check', self.update_step)
        return self.update_step #count_steps(self.config.traindir)

    def set_eval(self):
        self.training=False
    
    def set_train(self):
        self.training=True
            