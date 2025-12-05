
import pathlib

from .tools import *
from controllers.rl.dreamer_v3.dreamer import Dreamer
from agent_arena import TrainableAgent
import gym

class DreamerV3Adapter(TrainableAgent):
  
    def __init__(self, config):
        super().__init__(config)
        set_seed_everywhere(config.seed)
        if config.deterministic_run:
            enable_deterministic_run()
        self.config = config
        self.config.steps //= config.action_repeat
        self.loaded_model = False
        self.loaded_dataset = False
        self.initialised_agent = False
        self.primitive_integration = config.get('primitive_integration', 'none')
        if self.primitive_integration == 'predict_bin_as_output':
            self.primitives = self.config.primitives
            self.K = len(self.primitives)
            self.action_dims = [prim['dim'] if isinstance(prim, dict) else prim.dim for prim in self.primitives]
            self.config.num_actions = max(self.action_dims) + 1
            print('num actions', self.config.num_actions)
        
    
    def set_log_dir(self, log_dir):
        logdir = pathlib.Path(log_dir).expanduser()
        super().set_log_dir(log_dir)
        
        self.config.traindir = self.config.traindir or logdir / "train_eps"
        #self.config.evaldir = self.config.evaldir or logdir / "eval_eps"
        
        #self.config.eval_every //= self.config.action_repeat
        self.config.log_every //= self.config.action_repeat
        #self.config.time_limit //= self.config.action_repeat

        print("Logdir", logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        self.config.traindir.mkdir(parents=True, exist_ok=True)
        #self.config.evaldir.mkdir(parents=True, exist_ok=True)

        self.step = count_steps(self.config.traindir)
        self.logger.step = self.config.action_repeat * self.step
        # step in logger is environmental step
        # self.logger = WandbLogger(logdir, 
        #     self.config.action_repeat * self.step, 
        #     self.config.project_name,
        #     name=self.config.exp_name,
        #     config=dict(self.config))
        
        # if self.config.offline_evaldir:
        #     directory = self.config.offline_evaldir.format(**vars(self.config))
        # else:
        #     directory = self.config.evaldir
        # self.eval_eps = load_episodes(directory, limit=1)

        if self.config.offline_traindir:
            directory = self.config.offline_traindir.format(**vars(self.config))
        else:
            directory = self.config.traindir
        self.logdir = logdir
        self.train_eps = load_episodes(directory, limit=self.config.dataset_size)

    def _prefill_dataset(self, arenas):
        self.train_state = None
        num_env = len(arenas)
        if not self.config.offline_traindir:
            prefill = max(0, self.config.prefill - count_steps(self.config.traindir))
            print(f"Prefill dataset ({prefill} steps).")
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
            self.logger.step += prefill * self.config.action_repeat
            self.step = count_steps(self.config.traindir)
            print(f"Logger: ({self.logger.step} steps).")

    def _load_dataset(self):
        if self.loaded_dataset:
            return
        self.train_dataset = make_dataset(self.train_eps, self.config)
        self.loaded_dataset = True

    def _init_agent(self, arenas):
        if self.initialised_agent:
            return
        rnd_act_space = gym.spaces.Box(-1, 1, (self.config.num_actions, ), dtype=np.float32)
        self.dreamer = Dreamer(
            arenas[0].observation_space,
            rnd_act_space,
            self.config,
            self.logger,
            self.train_dataset,
            self.data_augmenter
        ).to(self.config.device)
        self.dreamer.requires_grad_(requires_grad=False)
        self.load()
            
        self.initialised_agent = True
    
    def init(self, info_list):
        for info in info_list:
            self.internal_states[info['arena_id']] = {}
            self.internal_states[info['arena_id']]['done'] = False
            self.internal_states[info['arena_id']]['agent_state'] = None

    def update(self, info_list, actions):
        for info in info_list:
            self.internal_states[info['arena_id']]['done'] = info['done']

    def single_act(self, info, update=False):
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

    def train(self, update_steps, arenas) -> bool:
        
        self._prefill_dataset(arenas)
        self._load_dataset()
        self._init_agent(arenas)

        target_steps = min(self.step + update_steps, self.config.total_update_steps)
        steps_to_do = target_steps - self.step
        #print('steps to do', steps_to_do)
        self.logger.write()
        self.train_state = simulate(
            self,
            self.dreamer,
            arenas,
            self.train_eps,
            self.config.traindir,
            self.logger,
            limit=self.config.dataset_size,
            steps=steps_to_do,
            state=self.train_state,
            parallel=self.config.parallel,
            restart=self.config.restart,
            obs_keys = self.config.obs_keys,
            reward_key = self.config.reward_key
        )
        self.step = self.dreamer._step
        #print('end train step', self.step)
    
    def save(self):
        items_to_save = {
            "agent_state_dict": self.dreamer.state_dict(),
            "optims_state_dict": recursively_collect_optim_state_dict(self.dreamer),
        }
        torch.save(items_to_save, self.logdir / "latest.pt")

    def save_best(self):

        items_to_save = {
            "agent_state_dict": self.dreamer.state_dict(),
            "optims_state_dict": recursively_collect_optim_state_dict(self.dreamer),
        }
        torch.save(items_to_save, self.logdir / "best.pt")
        


    def load(self):
        if not self.initialised_agent:
            return count_steps(self.config.traindir)
            
        if self.loaded_model:
            return count_steps(self.config.traindir)
        
        if (self.logdir / "latest.pt").exists():
            checkpoint = torch.load(self.logdir / "latest.pt")
            self.dreamer.load_state_dict(checkpoint["agent_state_dict"])
            recursively_load_optim_state_dict(self.dreamer, checkpoint["optims_state_dict"])
            self.dreamer._should_pretrain._once = False

        self.loaded_model = True

        return count_steps(self.config.traindir)

    def set_eval(self):
        self.training=False
    
    def set_train(self):
        self.training=True
            