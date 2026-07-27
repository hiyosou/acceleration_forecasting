from __future__ import annotations

import math

import torch
from torch import nn


class MaskedCrossAttention(nn.Module):
    def __init__(self, query_dim, guide_dim=64, heads=8, output_dim=None):
        super().__init__()
        if guide_dim % heads:
            raise ValueError("guide_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = guide_dim // heads
        self.to_query = nn.Linear(query_dim, guide_dim, bias=False)
        self.to_key = nn.Linear(guide_dim, guide_dim, bias=False)
        self.to_value = nn.Linear(guide_dim, guide_dim, bias=False)
        self.to_output = nn.Linear(guide_dim, output_dim or query_dim)
        self.similarity_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, guide_features, guide_mask, retrieval_mask, similarities, return_weights=False):
        batch, query_length, _ = query.shape
        guides, months = guide_features.shape[1:3]
        guide = guide_features.reshape(batch, guides * months, -1)
        token_mask = (guide_mask * retrieval_mask.unsqueeze(-1)).reshape(batch, guides * months) > 0
        similarity_bias = similarities.unsqueeze(-1).expand(-1, -1, months).reshape(batch, 1, 1, -1)
        q = self.to_query(query).reshape(batch, query_length, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_key(guide).reshape(batch, guides * months, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_value(guide).reshape(batch, guides * months, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.similarity_scale * similarity_bias
        scores = scores.masked_fill(~token_mask[:, None, None, :], -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * token_mask[:, None, None, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, query_length, -1)
        output = self.to_output(output)
        if return_weights:
            return output, weights
        return output

