from __future__ import annotations

import torch
from torch import nn

from .base import ConditionalDenoiser
from .masked_cross_attention import MaskedCrossAttention


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim=512, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.Dropout(dropout),
        )

    def forward(self, value):
        return value + self.block(value)


class MLPDenoiser(ConditionalDenoiser):
    def __init__(self, future_months=18, token_dim=64, hidden=512, dropout=0.1):
        super().__init__(condition_dim=128, time_dim=128, guide_dim=64)
        self.future_months = future_months
        self.value_projection = nn.Linear(1, token_dim)
        self.month_embedding = nn.Embedding(future_months, token_dim)
        self.global_projection = nn.Linear(256, token_dim)
        self.attention = MaskedCrossAttention(token_dim, guide_dim=64, heads=8, output_dim=token_dim)
        self.input = nn.Linear(future_months * token_dim, hidden)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden, dropout) for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, future_months))

    def forward(self, noisy_future, timesteps, batch, return_attention=False):
        condition, time, guides = self.encode_conditions(batch, timesteps)
        months = torch.arange(self.future_months, device=noisy_future.device)
        query = self.value_projection(noisy_future.unsqueeze(-1))
        query = query + self.month_embedding(months).unsqueeze(0)
        query = query + self.global_projection(torch.cat([condition, time], dim=-1)).unsqueeze(1)
        attended = self.attention(
            query, guides, batch["guide_mask"], batch["retrieval_mask"],
            batch["guide_similarities"], return_weights=return_attention,
        )
        weights = None
        if return_attention:
            attended, weights = attended
        hidden = self.input((query + attended).reshape(noisy_future.shape[0], -1))
        prediction = self.output(self.blocks(hidden))
        return (prediction, weights) if return_attention else prediction

