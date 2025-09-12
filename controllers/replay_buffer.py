
class ReplayBuffer:
    def __init__(self, capacity: int, image_shape: Tuple[int, int, int], num_context: int, action_dim: int, device: str):
        self.capacity = int(capacity)
        self.num_context = num_context
        C, H, W = image_shape
        self.device = device

        self.ptr = 0
        self.size = 0

        self.contexts = np.zeros((self.capacity, self.num_context, C, H, W), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_contexts = np.zeros((self.capacity, self.num_context, C, H, W), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def add(self, context: np.ndarray, action: np.ndarray, reward: float, next_context: np.ndarray, done: bool):
        self.contexts[self.ptr] = context
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_contexts[self.ptr] = next_context
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            context=torch.tensor(self.contexts[idxs]).to(self.device),
            action=torch.tensor(self.actions[idxs]).to(self.device),
            reward=torch.tensor(self.rewards[idxs]).unsqueeze(-1).to(self.device),
            next_context=torch.tensor(self.next_contexts[idxs]).to(self.device),
            done=torch.tensor(self.dones[idxs]).unsqueeze(-1).to(self.device),
        )
        return batch