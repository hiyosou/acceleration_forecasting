from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source[name], alpha=1 - self.decay)
            else:
                value.copy_(source[name])

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

