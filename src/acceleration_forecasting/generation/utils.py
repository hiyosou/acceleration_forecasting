from __future__ import annotations

import torch


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def repeat_conditions(batch, repeats):
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and key != "index":
            output[key] = value.repeat_interleave(repeats, dim=0)
        else:
            output[key] = value
    return output


def choose_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

