from __future__ import annotations

import torch
from torch import nn


class GuideEncoder(nn.Module):
    def __init__(self, guide_dim=64, month_dim=16, rank_dim=8):
        super().__init__()
        self.month_embedding = nn.Embedding(18, month_dim)
        self.rank_embedding = nn.Embedding(3, rank_dim)
        self.continuous = nn.Sequential(nn.Linear(3, 32), nn.SiLU())
        self.network = nn.Sequential(
            nn.Linear(32 + month_dim + rank_dim, guide_dim),
            nn.SiLU(),
            nn.Linear(guide_dim, guide_dim),
        )

    def forward(self, values, deltas, similarities):
        batch, guides, months = values.shape
        similarity = similarities.unsqueeze(-1).expand(-1, -1, months)
        continuous = self.continuous(torch.stack([values, deltas, similarity], dim=-1))
        month_ids = torch.arange(months, device=values.device).view(1, 1, months)
        rank_ids = torch.arange(guides, device=values.device).view(1, guides, 1)
        month = self.month_embedding(month_ids).expand(batch, guides, -1, -1)
        rank = self.rank_embedding(rank_ids).expand(batch, -1, months, -1)
        return self.network(torch.cat([continuous, month, rank], dim=-1))

