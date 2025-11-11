import torch
import numpy as np
import zarr
from pathlib import Path


class ObsStateReplayBufferZarr:
    def __init__(self, capacity, obs_shape, state_dim, action_dim, device, zarr_path="obs_state_replay.zarr"):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0

        # Setup Zarr store
        self.zarr_path = Path(zarr_path)
        self.zarr_path.parent.mkdir(parents=True, exist_ok=True)
        store = zarr.DirectoryStore(self.zarr_path)
        self.root = zarr.group(store=store, overwrite=True)

        # Create datasets — memory mapped to disk
        self.root.create_dataset(
            "observation", shape=(self.capacity, *obs_shape), dtype="float32", 
            chunks=(min(1024, self.capacity), *obs_shape)
        )
        self.root.create_dataset(
            "state", shape=(self.capacity, state_dim), dtype="float32",
            chunks=(min(1024, self.capacity), state_dim)
        )
        self.root.create_dataset(
            "actions", shape=(self.capacity, action_dim), dtype="float32",
            chunks=(min(1024, self.capacity), action_dim)
        )
        self.root.create_dataset(
            "rewards", shape=(self.capacity,), dtype="float32",
            chunks=(min(1024, self.capacity),)
        )
        self.root.create_dataset(
            "next_observation", shape=(self.capacity, *obs_shape), dtype="float32", 
            chunks=(min(1024, self.capacity), *obs_shape)
        )
        self.root.create_dataset(
            "next_state", shape=(self.capacity, state_dim), dtype="float32",
            chunks=(min(1024, self.capacity), state_dim)
        )
        self.root.create_dataset(
            "dones", shape=(self.capacity,), dtype="float32",
            chunks=(min(1024, self.capacity),)
        )

        # Keep convenient references
        self.observation = self.root["observation"]
        self.state = self.root["state"]
        self.actions = self.root["actions"]
        self.rewards = self.root["rewards"]
        self.next_observation = self.root["next_observation"]
        self.next_state = self.root["next_state"]
        self.dones = self.root["dones"]

    def add(self, observation, state, action, reward, next_observation, next_state, done):
        """Add a transition to the buffer."""
        self.observation[self.ptr] = observation
        self.state[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observation[self.ptr] = next_observation
        self.next_state[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            observation=torch.tensor(self.observation.get_orthogonal_selection((idxs,))).to(self.device),
            state=torch.tensor(self.state.get_orthogonal_selection((idxs,))).to(self.device),
            action=torch.tensor(self.actions.get_orthogonal_selection((idxs,))).to(self.device),
            reward=torch.tensor(self.rewards.get_orthogonal_selection((idxs,))).unsqueeze(-1).to(self.device),
            next_observation=torch.tensor(self.next_observation.get_orthogonal_selection((idxs,))).to(self.device),
            next_state=torch.tensor(self.next_state.get_orthogonal_selection((idxs,))).to(self.device),
            done=torch.tensor(self.dones.get_orthogonal_selection((idxs,))).unsqueeze(-1).to(self.device),
        )
        return batch
