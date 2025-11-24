
import pathlib

from .tools import *
from controllers.rl.dreamer_v3.dreamer import Dreamer
from agent_arena import TrainableAgent

class DreamerV3Adapter(TrainableAgent):
  
    def __init__(self, config):
        set_seed_everywhere(config.seed)
        if config.deterministic_run:
            enable_deterministic_run()
        self.config = config
        self.config.steps //= config.action_repeat
        self.loaded_model = False
        self.loaded_dataset = False
        self.initialised_agent = False
        
    
    def set_log_dir(self, log_dir):
        logdir = pathlib.Path(log_dir).expanduser()
        self.config.traindir = self.config.traindir or logdir / "train_eps"
        self.config.evaldir = self.config.evaldir or logdir / "eval_eps"
        
        self.config.eval_every //= self.config.action_repeat
        self.config.log_every //= self.config.action_repeat
        self.config.time_limit //= self.config.action_repeat

        print("Logdir", logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        self.config.traindir.mkdir(parents=True, exist_ok=True)
        self.config.evaldir.mkdir(parents=True, exist_ok=True)

        self.step = count_steps(self.config.traindir)
        # step in logger is environmental step
        self.logger = Logger(logdir, 
            self.config.action_repeat * self.step, 
            self.config.project_name,
            name=self.config.exp_name,
            config=dict(self.config))
        
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
            acts = arenas[0].get_action_space()() # Future object, that is why
            if hasattr(acts, "discrete"):
                random_actor = OneHotDist(
                    torch.zeros(self.config.num_actions).repeat(num_env, 1)
                )
            else:
                
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.tensor(acts.low).repeat(num_env, 1),
                        torch.tensor(acts.high).repeat(num_env, 1),
                    ),
                    1,
                )

            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None

            self.train_state = simulate(
                random_agent,
                arenas,
                self.train_eps,
                self.config.traindir,
                self.logger,
                limit=self.config.dataset_size,
                steps=prefill,
            )
            self.logger.step += prefill * self.config.action_repeat
            print(f"Logger: ({self.logger.step} steps).")

    def _load_dataset(self):
        if self.loaded_dataset:
            return
        self.train_dataset = make_dataset(self.train_eps, self.config)
        self.loaded_dataset = True

    def _init_agent(self, arenas):
        if self.initialised_agent:
            return
        self.agent = Dreamer(
            arenas[0].observation_space,
            arenas[0].get_action_space()(),
            self.config,
            self.logger,
            self.train_dataset,
        ).to(self.config.device)
        self.agent.requires_grad_(requires_grad=False)
        self.load()
            
        self.initialised_agent = True
    
    def init(self, info_list):
        for info in info_list:
            self.internale_state[info['aid']]['done'] = False
            self.internale_state[info['aid']]['agent_state'] = None

    def update(self, info_list, actions):
        for info in info_list:
            self.internale_state[info['aid']]['done'] = info['done']

    def single_act(self, info, update=False):
        
        agent_state = self.internale_state[info['aid']]['agent_state']
        obs = info['observation']['image']
        done = info['aid']['done']

        action, agent_state = self.agent(obs, done, agent_state, training=self.training)
        
        self.internale_state[info['aid']]['agent_state'] = agent_state

        return action['action']

    def train(self, update_steps, arenas) -> bool:
        self._prefill_dataset(arenas)
        self._load_dataset()
        self._init_agent(arenas)

        target_steps = min(self.agent._step + update_steps, self.config.steps)

        while self.agent._step <  target_steps:
            self.logger.write()
            self.train_state = simulate(
                self.agent,
                arenas,
                self.train_eps,
                self.config.traindir,
                self.logger,
                limit=self.config.dataset_size,
                steps=self.config.eval_every,
                state=self.train_state,
            )
    
    def save(self):
        items_to_save = {
            "agent_state_dict": self.agent.state_dict(),
            "optims_state_dict": recursively_collect_optim_state_dict(self.agent),
        }
        torch.save(items_to_save, self.logdir / "latest.pt")

    def save_best(self):

        items_to_save = {
            "agent_state_dict": self.agent.state_dict(),
            "optims_state_dict": recursively_collect_optim_state_dict(self.agent),
        }
        torch.save(items_to_save, self.logdir / "best.pt")
        


    def load(self):
        if self.loaded_model:
            return
        
        if (self.logdir / "latest.pt").exists():
            checkpoint = torch.load(self.logdir / "latest.pt")
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            recursively_load_optim_state_dict(self.agent, checkpoint["optims_state_dict"])
            self.agent._should_pretrain._once = False

        self.loaded_model = True

        return self.step

    def set_eval(self):
        self.training=False
    
    def set_train(self):
        self.training=True
            