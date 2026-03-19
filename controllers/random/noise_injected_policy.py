import os
import numpy as np
from actoris_harena import Agent
from hydra import compose
import actoris_harena as athar
ENV_ASSETS_DIR = os.environ.get("RAVENS_ASSETS_DIR", "")

class NoiseInjectedPolcy(Agent):
    """
    Noisy Injected Policy
    Has a 50% probability of acting perfectly; otherwise injects Gaussian noise.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "noise-injected-policy"
        self.add_noise_prob = config.get('add_noise_prob', 0.50)
        self.noise_scale = config.get('noise_sacle', 0.1)  # Adjust this standard deviation to make the error more or less severe

        # basae policy stablishment.
        base_policy_config = compose(config_name=config.base_policy)
        self.base_policy = athar.build_agent(
            base_policy_config.agent.name,
            base_policy_config.agent,
            project_name=base_policy_config.project_name,
            exp_name=config.base_policy,
            save_dir=os.path.join(base_policy_config.save_root, config.base_policy),
            disable_wandb=True
        )


    def single_act(self, info, update=False):
        # 1. Get the pristine expert action.
        expert_action = self.base_policy.single_act(info, update=update)
        
        # Ensure we don't add noise to a no-op/done action (all zeros)
        if np.all(expert_action == 0.0):
            return expert_action

        # 2. Roll the dice against the 50% success probability
        if np.random.rand() < self.add_noise_prob:
            # Disable noise: return the flawless expert action
            return expert_action
        
        # 3. Generate Gaussian noise for all 5 dimensions
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=expert_action.shape)
        noisy_action = expert_action + noise
        
        # 4. Enforce strict [-1.0, 1.0] boundaries
        clipped_action = np.clip(noisy_action, -1.0, 1.0)
        
        return clipped_action

    def reset(self, arena_ids):
        self.base_policy.reset(arena_ids)

    def get_phase(self):
        return self.base_policy.get_phase()
    
    def terminate(self):
        return self.base_policy.terminate()
        
    
    def init(self, infos):
        # No internal state initialization required for random policy
        self.base_policy.init(infos)

    def update(self, infos, actions):
        self.base_policy.update(infos, actions)

    def get_state(self):
        return self.base_policy.get_state()
    
    def success(self):
        return self.base_policy.success()