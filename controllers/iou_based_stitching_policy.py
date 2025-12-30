from agent_arena import Agent
import numpy as np
import os
from hydra import compose, initialize


class IoUBasedStitchingPolicy(Agent):
    
    def __init__(self, config):
        
        super().__init__(config)
        self.name = 'iou_based_stitching_policy'
        #from omegaconf import OmegaConf
        
        
        # flattening_policy_config_path = os.path.join("conf", "agent", f"{config.flattening_policy}.yaml")
        # folding_policy_config_path = os.path.join("conf", "agent", f"{config.folding_policy}.yaml")

        #with initialize(config_path="conf"):
        flattening_policy_config = compose(
            config_name=config.flattening_policy
        )
        folding_policy_config = compose(
            config_name=config.folding_policy
        )
        
        # flattening_policy_config = OmegaConf.load(flattening_policy_config_path)
        # folding_policy_config = OmegaConf.load(folding_policy_config_path)
        
        # flattening_policy_config.project_name = config.project_name
        # flattening_policy_config.exp_name = config.flattening_policy
        import agent_arena.api as ag_ar
        print(f'[iou_based_stitching_policy] Building flattening agent from {config.flattening_policy}')
        self.flattening_policy = ag_ar.build_agent(
            flattening_policy_config.agent.name, 
            flattening_policy_config.agent,
            project_name=flattening_policy_config.project_name,
            exp_name=config.flattening_policy,
            save_dir= os.path.join(flattening_policy_config.save_root, config.flattening_policy))
        
        print(f'[iou_based_stitching_policy] Building folding agent from {config.folding_policy}')
        self.folding_policy = ag_ar.build_agent(
            folding_policy_config.agent.name, 
            folding_policy_config.agent,
            project_name=folding_policy_config.project_name,
            exp_name=config.folding_policy,
            save_dir= os.path.join(folding_policy_config.save_root, config.folding_policy))
        
        # self.flattening_policy.set_log_dir(
        #     os.path.join(config.flattening_policy_log_dir, config.flattening_policy)
        # )
       
        # TODO: we need to simplify the following structure.
        # folding_policy_config.project_name = config.project_name
        # folding_policy_config.exp_name = config.folding_policy
        # self.folding_policy = ag_ar.build_agent(
        #     folding_policy_config.name,
        #     folding_policy_config)
        # self.folding_policy.set_log_dir(
        #     os.path.join(config.folding_policy_log_dir, config.folding_policy)
        # )
        
        self.flattening_policy.load_best()
        self.folding_policy.load_best()
    
    def set_data_augmenter(self, data_augmenter):
        self.flattening_policy.set_data_augmenter(data_augmenter)
        self.folding_policy.set_data_augmenter(data_augmenter)
        
    
    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}
        self.last_primitive = None
        self.flattening_policy.reset(arena_ids)
        self.folding_policy.reset(arena_ids)
    
    def init(self, infos):
        self.flattening_policy.init(infos)
        self.folding_policy.init(infos)

    def update(self, infos, actions):
        self.flattening_policy.init(infos)
        self.folding_policy.init(infos)

    def act(self, info_list, update=False):
        
        actions = []
        for info in info_list:
            actions.append(self.single_act(info))
        
        return actions
    
    def _should_folding(self, state):
        r = state['reward']['multi_stage_reward']
        print('[oracle-baed-stitching policy] reward', r)
        return r >= 1.0 and (r - int(r)) < 1e-6


    def single_act(self, state, update=False):
        """
        Allow user to choose a primitive, then delegate to the chosen primitive's act method.
        Shows rgb and goal_rgb images while prompting for input.
        """

        if self._should_folding(state):
            print('[oracle-baed-stitching policy] Folding')
            return self.folding_policy.single_act(state)
        else:
            print('[oracle-baed-stitching policy] Flattening')
            return self.flattening_policy.single_act(state)