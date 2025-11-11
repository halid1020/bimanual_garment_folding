import numpy as np
from .robosuite_arena import RoboSuiteArena
from .skill_controller import SkillController
from omegaconf import OmegaConf

class RoboSuiteSkillArena(RoboSuiteArena):
    """
    A skill-level environment wrapper that extends RoboSuiteArena.
    Each .step() executes one full skill using SkillController,
    taking a dict-based skill action: e.g., {"push": np.array([...])}.
    """

    def __init__(self, config):
        super().__init__(config)

        # TODO: check if skill_config is omegaConfig, if so convert it to normal dict
        skill_config = config.skill_config
        if OmegaConf.is_config(skill_config):
            skill_config = OmegaConf.to_container(skill_config, resolve=True)

        self.skill_controller = SkillController(self.env, skill_config)
        self.max_skill_repeats = config.skill_config.get("max_skill_repeats", 1)

        #base_action_dim = self.get_action_space()
        # self.skill_dim = self.skill_controller.get_skill_dim()
        #self.param_dim = self.skill_controller.get_param_dim(base_action_dim)

        #self.action_space = None  # not used directly now
        self.observation_space = self.env.observation_space
        self.reward_scale = config.get('reward_scale', 1.0)

        self.current_obs = None
        self.done = False
        self.skill_step_count = 0

    def get_param_dim(self, skill_name):
        #print('action space', self.action_space)
        return self.skill_controller.get_param_dim(skill_name)

    def reset(self, episode_config=None):
        info = super().reset(episode_config)
        self.done = False
        self.skill_step_count = 0
        self.current_obs = info["observation"]
        return info
        

    def step(self, skill_action: dict):
        """
        Execute one high-level skill action.
        skill_action example:
            {"push": np.array([...])}
        """
        if self.done:
            raise RuntimeError("Environment is done — call reset() first.")

        # Validate input
        if not isinstance(skill_action, dict) or len(skill_action) != 1:
            raise ValueError(
                f"Expected single-skill dict like {{'push': params}}, got {skill_action}"
            )

        skill_name, params = list(skill_action.items())[0]
        #print(f"Executing skill '{skill_name}' with params {np.round(params, 3)}")

        # Reset controller for this specific skill
        self.skill_controller.reset(skill_action)

        cumulative_reward = 0.0
        low_level_steps = 0
        skill_done = False
        info = None

        # Run the skill loop
        while not skill_done:
            low_level_action = self.skill_controller.step()
            #print('low_level_action', low_level_action)
            info = super().step(low_level_action)

            reward = info["reward"]["default"]
            #print('reward scale', self.reward_scale)
            cumulative_reward += reward * self.reward_scale
            low_level_steps += 1

            skill_done = self.skill_controller.done()
            #print(f'done? {skill_done} at sim step{low_level_steps}')
            if info["done"]:
                self.done = True
                break
            
            if info['fail_step']:
                self.done = True
                break

        cumulative_reward_ = self.skill_controller.post_process_reward(cumulative_reward)
        reward_ = self.skill_controller.post_process_reward(reward)
        aff_reward = self.skill_controller.get_aff_reward()

        self.current_obs = info["observation"]
        self.skill_step_count += 1

        return {
            "observation": self.current_obs,
            "reward": {
                "cumulative_reward": cumulative_reward,
                "cumulative_reward_with_affordance_penalty": cumulative_reward_,
                "last_reward": reward,
                "last_reward_with_affordance_penalty": reward_,
                "aff_reward": aff_reward,
            },
            "evaluation": {},
            "success": self.success(),
            "done": self.done,
            "sim_steps": low_level_steps,
            "arena": self,
            "arena_id": self.id,
        }

    def render(self):
        return super().render()

    def close(self):
        super().close()
