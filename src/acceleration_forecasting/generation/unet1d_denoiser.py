from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .base import ConditionalDenoiser
from .masked_cross_attention import MaskedCrossAttention


class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels, global_dim=256, dropout=0.1):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.film = nn.Linear(global_dim, channels * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, global_condition):
        scale, shift = self.film(global_condition).chunk(2, dim=-1)
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.norm2(hidden) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return value + hidden


class AttentionStage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = MaskedCrossAttention(channels, 64, heads=8, output_dim=channels)

    def forward(self, value, guides, batch, return_weights=False):
        query = value.transpose(1, 2)
        result = self.attention(
            query, guides, batch["guide_mask"], batch["retrieval_mask"],
            batch["guide_similarities"], return_weights=return_weights,
        )
        weights = None
        if return_weights:
            result, weights = result
        output = value + result.transpose(1, 2)
        return (output, weights) if return_weights else output


class UNet1DDenoiser(ConditionalDenoiser):
    def __init__(self, dropout=0.1, use_cross_attention=True):
        super().__init__(
            condition_dim=128, time_dim=128, guide_dim=64,
            use_guide_encoder=use_cross_attention,
        )
        self.use_cross_attention = bool(use_cross_attention)
        self.input = nn.Conv1d(1, 64, 3, padding=1)
        self.enc0 = nn.ModuleList([ConditionalResidualBlock(64, dropout=dropout) for _ in range(2)])
        self.attn0 = AttentionStage(64) if self.use_cross_attention else None
        self.down1 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.enc1 = nn.ModuleList([ConditionalResidualBlock(128, dropout=dropout) for _ in range(2)])
        self.attn1 = AttentionStage(128) if self.use_cross_attention else None
        self.down2 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        self.mid = nn.ModuleList([ConditionalResidualBlock(256, dropout=dropout) for _ in range(2)])
        self.attn_mid = AttentionStage(256) if self.use_cross_attention else None
        self.up1 = nn.ConvTranspose1d(256, 128, 3, stride=2, padding=1)
        self.merge1 = nn.Conv1d(256, 128, 1)
        self.dec1 = nn.ModuleList([ConditionalResidualBlock(128, dropout=dropout) for _ in range(2)])
        self.up0 = nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.merge0 = nn.Conv1d(128, 64, 1)
        self.dec0 = nn.ModuleList([ConditionalResidualBlock(64, dropout=dropout) for _ in range(2)])
        self.output = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv1d(64, 1, 3, padding=1))

    @staticmethod
    def _blocks(value, blocks, condition):
        for block in blocks:
            value = block(value, condition)
        return value

    def forward(self, noisy_future, timesteps, batch, return_attention=False):
        condition, time, guides = self.encode_conditions(batch, timesteps)
        global_condition = torch.cat([condition, time], dim=-1)
        x0 = self._blocks(self.input(noisy_future.unsqueeze(1)), self.enc0, global_condition)
        if self.use_cross_attention:
            x0 = self.attn0(x0, guides, batch)
        x1 = self._blocks(self.down1(x0), self.enc1, global_condition)
        if self.use_cross_attention:
            x1 = self.attn1(x1, guides, batch)
        x2 = self._blocks(self.down2(x1), self.mid, global_condition)
        if return_attention and self.use_cross_attention:
            x2, weights = self.attn_mid(x2, guides, batch, return_weights=True)
        elif self.use_cross_attention:
            x2 = self.attn_mid(x2, guides, batch)
            weights = None
        else:
            weights = None
        y1 = self.up1(x2)
        y1 = self._blocks(self.merge1(torch.cat([y1, x1], dim=1)), self.dec1, global_condition)
        y0 = self.up0(y1)
        y0 = self._blocks(self.merge0(torch.cat([y0, x0], dim=1)), self.dec0, global_condition)
        prediction = self.output(y0).squeeze(1)
        return (prediction, weights) if return_attention else prediction
