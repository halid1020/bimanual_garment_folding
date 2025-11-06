import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureCNNEncoder(nn.Module):
    """
    NatureCNN-style encoder used in Stable Baselines3 for image-based SAC.
    Input: (C, H, W), default (3, 84, 84)
    Output: feature vector of size `feature_dim` (default 512)
    """
    def __init__(self, obs_shape=(3, 84, 84), feature_dim=512):
        super().__init__()
        assert len(obs_shape) == 3, "Input must be 3D (C,H,W)"
        self.conv_net = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9x9 -> 7x7
            nn.ReLU()
        )

        # Compute flatten size
        with torch.no_grad():
            n_flatten = self.conv_net(torch.zeros(1, *obs_shape)).view(1, -1).size(1)

        self.fc = nn.Linear(n_flatten, feature_dim)

    def forward(self, obs):
        # Normalize image to [0,1]
        x = obs / 255.0
        x = self.conv_net(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return F.relu(x)

class NatureCNNEncoderRegressor(nn.Module):
    def __init__(self, obs_shape=(3, 84, 84), state_dim=45, feature_dim=512):
        super().__init__()
        assert len(obs_shape) == 3, "Input must be 3D (C,H,W)"
        self.conv_net = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9x9 -> 7x7
            nn.ReLU()
        )

        # Compute flatten size
        with torch.no_grad():
            n_flatten = self.conv_net(torch.zeros(1, *obs_shape)).view(1, -1).size(1)

        self.fc = nn.Linear(n_flatten, feature_dim)

        self.regressor = nn.Linear(feature_dim, state_dim)

    def forward(self, obs):
        # Normalize image to [0,1]
        x = obs / 255.0
        x = self.conv_net(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        e = F.relu(x)
        x = self.regressor(e)
        return e, x


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
        # if contexts.dim() == 4:
        #     contexts = contexts.unsqueeze(0)
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