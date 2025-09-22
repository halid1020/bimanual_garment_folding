import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, cnn_channels, out_dim, kernel=3, pool=2):
        super().__init__()
        layers = []
        prev = in_channels
        for c in cnn_channels:
            layers.append(nn.Conv2d(prev, c, kernel_size=kernel, stride=1, padding=kernel // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pool))
            prev = c
        self.cnn = nn.Sequential(*layers)
        self._project = None
        self._initialized = False
        self.out_dim = out_dim

    def _ensure_init(self, x: torch.Tensor):
        if self._initialized:
            return
        with torch.no_grad():
            y = self.cnn(x)
            sz = y.shape[1] * y.shape[2] * y.shape[3]
            self._project = nn.Sequential(nn.Flatten(), nn.Linear(sz, self.out_dim), nn.ReLU()).to(x.device)
            self._initialized = True

    def forward(self, contexts):
        if contexts.dim() == 4:
            contexts = contexts.unsqueeze(0)
        B, N, C, H, W = contexts.shape
        contexts_flat = contexts.reshape(B, N * C, H, W)
        #print(contexts_flat.shape)
        self._ensure_init(contexts_flat)
        features = self.cnn(contexts_flat)
        features = features.view(features.size(0), -1)
        projected = self._project(features)
        projected = projected.view(B, -1)
        return projected #.mean(dim=1)


class MLPActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x)