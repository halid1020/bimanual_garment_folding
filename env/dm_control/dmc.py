import gym
import numpy as np
import os
os.environ["MUJOCO_GL"] = "osmesa"
class DeepMindControl:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.reward_range = [-np.inf, np.inf]
        

    

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)

def main():
    # -------------------------------
    # Create the environment
    # -------------------------------
    env_name = "cartpole_balance"  # Example task
    env = DeepMindControl(name=env_name, action_repeat=1, size=(64, 64), seed=42)

    print("Created environment:", env_name)
    print("Action space:", env.action_space)
    print("Observation space keys:", env.observation_space.spaces.keys())

    # -------------------------------
    # Reset environment
    # -------------------------------
    obs = env.reset()
    print("\nInitial observation keys:", obs.keys())
    print("Initial image shape:", obs["image"].shape)
    print('Initial Obs keys', obs.keys() )


    # -------------------------------
    # Step through the environment
    # -------------------------------
    num_steps = 10
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print(f"\nStep {step}")
        print("  Reward:", reward)
        print("  Done:", done)
        print("  Info:", info)
        print("  Obs keys:", obs.keys())
        print("  Image shape:", obs["image"].shape)
        print("  Is Terminal:", obs['is_terminal'])
        print("  Is FirstL", obs["is_first"])

        if done:
            print("Episode terminated early. Resetting...")
            obs = env.reset()

    # -------------------------------
    # Render one frame
    # -------------------------------
    frame = env.render()
    print("\nRendered frame shape:", frame.shape)


if __name__ == "__main__":
    main()