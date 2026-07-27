from __future__ import annotations

import sys

from tqdm import tqdm


def progress_bar(iterable=None, *, enabled=True, **kwargs):
    """Create a consistently configured, low-frequency stderr progress bar."""
    defaults = {
        "disable": not enabled,
        "file": sys.stderr,
        "mininterval": 1.0,
        "maxinterval": 10.0,
        "dynamic_ncols": True,
    }
    defaults.update(kwargs)
    return tqdm(iterable, **defaults)


def progress_message(message, *, enabled=True):
    if enabled:
        tqdm.write(str(message), file=sys.stderr)
