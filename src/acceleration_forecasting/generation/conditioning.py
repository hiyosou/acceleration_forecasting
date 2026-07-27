from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = int(dim)
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timesteps):
        half = self.dim // 2
        frequency = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
        )
        values = timesteps.float().unsqueeze(1) * frequency.unsqueeze(0)
        embedding = torch.cat([values.sin(), values.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.dim - embedding.shape[1]))
        return self.projection(embedding)


class ConditionEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim)
        )

    def forward(self, current, history, history_mask):
        return self.network(torch.cat([current, history, history_mask], dim=-1))

