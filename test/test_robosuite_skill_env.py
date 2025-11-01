import numpy as np
import time
from env.robosuite_env.robosuite_skill_arena import RoboSuiteSkillArena
from agent_arena.utilities.visual_utils import save_video as sv
from agent_arena.utilities.visual_utils import save_numpy_as_gif as sg


def random_skill_action(skill_env):
    """
    Randomly sample one skill and its parameters.
    Returns a dict like {"push": np.array([...])}
    """
    skill_names = list(skill_env.skill_controller.get_skill_names()) # e.g. ['reach', 'grasp', 'push', 'open', 'close']
    chosen_skill = np.random.choice(skill_names)
    chosen_skill = "reach"

    num_params = skill_env.get_param_dim(chosen_skill)
    #print('num_params', num_params)
    
    
    params = np.random.uniform(-1.0, 1.0, size=num_params)
    params[:3] = [-1, -1, -1]

    return {chosen_skill: params}


def test_skill_env(config, max_episodes=3, max_skill_steps=10, render=False):
    env = RoboSuiteSkillArena(config)
    save_dir = 'tmp/'
    for ep in range(max_episodes):
        print(f"\n===== Episode {ep + 1} =====")
        obs = env.reset({"eid": ep, "save_video": True})
        ep_reward = 0.0

        for step in range(max_skill_steps):
            skill_action = random_skill_action(env)
            print(f"\n[Step {step}] Skill Action: {skill_action}")

            result = env.step(skill_action)
            obs = result["observation"]
            reward = result["reward"]["cumulative_reward"]
            done = result["done"]

            print(f"  -> Reward: {reward:.3f}, Done: {done}")

            ep_reward += reward

            if render:
                env.render()
                time.sleep(0.02)

            if done:
                print("Environment signaled done.")
                break

        print(f"Episode reward: {ep_reward:.3f}")

        sv(env.get_frames(), save_dir, 
            f'robosuite_episode_{ep}')

        sg(
            env.get_frames(), 
            path=save_dir,
            filename=f"robosuite_episode_{ep}"
        )

    #env.close()
    print("\n✅ Test finished.")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg_dict = {
        "name": "robosuite_skill_env",
        "env_name": "Lift",
        "horizon": 200,
        "disp": False,
        "use_camera_obs": False,
        "control_freq": 20,
        "robot_keys": ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'],
        "controller_name": "OSC_POSE",
        "env_kwargs": {
            "robots": ["Panda"],
            "controller_configs": {
                "control_delta": False,
                "use_delta": False
            }
            
        },
        "skill_config": dict(
            skills=['atomic', 'open', 'reach', 'grasp', 'push'],
            aff_penalty_fac=15.0,

            base_config=dict(
                global_xyz_bounds=[
                    [-0.30, -0.30, 0.80],
                    [0.15, 0.30, 0.95]
                ],
                lift_height=0.95,
                binary_gripper=True,

                aff_threshold=0.06,
                aff_type='dense',
                aff_tanh_scaling=10.0,
            ),
            atomic_config=dict(
                use_ori_params=True,
            ),
            reach_config=dict(
                use_gripper_params=False,
                local_xyz_scale=[0.0, 0.0, 0.06],
                use_ori_params=False,
                max_ac_calls=15,
            ),
            grasp_config=dict(
                global_xyz_bounds=[
                    [-0.30, -0.30, 0.80],
                    [0.15, 0.30, 0.85]
                ],
                aff_threshold=0.03,

                local_xyz_scale=[0.0, 0.0, 0.0],
                use_ori_params=True,
                max_ac_calls=20,
                num_reach_steps=2,
                num_grasp_steps=3,
            ),
            push_config=dict(
                global_xyz_bounds=[
                    [-0.30, -0.30, 0.80],
                    [0.15, 0.30, 0.85]
                ],
                delta_xyz_scale=[0.25, 0.25, 0.05],

                max_ac_calls=20,
                use_ori_params=True,

                aff_threshold=[0.12, 0.12, 0.04],
            ),
        ),
    }
    config = OmegaConf.create(cfg_dict)
    test_skill_env(config, max_episodes=2, max_skill_steps=5, render=False)
