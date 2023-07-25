import torch.nn as nn
import torch
from diffusion import SinusoidalPosEmb

class MLP(nn.Module):
    def __init__(self, input=49, output=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output),
        )

    def forward(self, x):
        return self.model(x)

class TimeMLP(nn.Module):
    def __init__(self, input=49, output=39):
        super().__init__()

        dim = 32
        self.diffusion_time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        input += dim

        self.model = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output),
        )

    def forward(self, x, t):
        t = self.diffusion_time_mlp(t)
        x = torch.cat([x, t], dim=1)
        return self.model(x)