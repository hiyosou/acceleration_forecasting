from __future__ import annotations

from torch import nn

from .conditioning import ConditionEncoder, SinusoidalTimeEmbedding
from .guide_encoder import GuideEncoder


class ConditionalDenoiser(nn.Module):
    def __init__(self, condition_dim=128, time_dim=128, guide_dim=64):
        super().__init__()
        self.condition_encoder = ConditionEncoder(condition_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.guide_encoder = GuideEncoder(guide_dim)

    def encode_conditions(self, batch, timesteps):
        condition = self.condition_encoder(batch["current"], batch["history"], batch["history_mask"])
        time = self.time_embedding(timesteps)
        guides = self.guide_encoder(
            batch["guide_values"], batch["guide_deltas"], batch["guide_similarities"]
        )
        return condition, time, guides

