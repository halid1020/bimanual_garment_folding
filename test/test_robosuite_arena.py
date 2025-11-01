import os
import cv2
import numpy as np
from dotmap import DotMap
from agent_arena.utilities.visual_utils import save_video as sv
from agent_arena.utilities.visual_utils import save_numpy_as_gif as sg

from env.robosuite_env.robosuite_arena import RoboSuiteArena  # adjust this import path


def save_video(frames, filename, fps=20):
    """Save a list of RGB frames (H,W,3) as an mp4 video."""
    if not frames:
        print("[WARN] No frames to save.")
        return
    h, w, _ = frames[0].shape
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    out = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"[INFO] Saved video to: {filename}")


def test_arena(num_episodes=3, save_dir="tmp/", max_steps=None):
    # Example config
    config = {
        "name": "robosuite_env",
        "env_name": "Lift",
        "horizon": 200,
        "disp": False,              # no onscreen rendering (for headless mode)
        "use_camera_obs": False,
        "control_freq": 20,
        "env_kwargs": {
            "robots": ["Panda"],
            "controller_configs": "OSC_POSE"
        }
    }

    config = DotMap(config)

    arena = RoboSuiteArena(config)
    print(f"[INFO] Created RoboSuiteArena with env: {arena.env_name}")

    episode_rewards = []
    print('observatin space', arena.get_observation_space())
    print('action space', arena.get_action_space())

    for ep in range(num_episodes):
        print(f"\n=== Episode {ep + 1}/{num_episodes} ===")
        info = arena.reset({"eid": ep, "save_video": True})
        done = False
        total_reward = 0.0
        step = 0
        max_steps = max_steps or arena.get_action_horizon()

        

        while not done and step < max_steps:
            action = arena.sample_random_action()
            info = arena.step(action)
            total_reward += info["reward"]
            done = info["done"]
            step += 1

        print(f"Episode {ep + 1} finished after {step} steps. Total reward: {total_reward:.3f}")
        episode_rewards.append(total_reward)

        sv(arena.get_frames(), save_dir, 
            f'robosuite_episode_{ep+1}')

        sg(
            arena.get_frames(), 
            path=save_dir,
            filename=f"robosuite_episode_{ep+1}"
        )


        arena.clear_frames()

    print("\n=== Summary ===")
    print(f"Avg reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Saved videos in '{save_dir}/'.")


if __name__ == "__main__":
    test_arena(num_episodes=3)
