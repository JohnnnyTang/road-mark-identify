"""Tiled inference with probability blending (§5.3). Tiles exist only here."""

from __future__ import annotations

import numpy as np


def blend_window(size: int, kind: str = "cosine") -> np.ndarray:
    """2-D weight that falls off towards tile edges; strictly positive."""
    raise NotImplementedError("Roadmap Step 4.4")


def predict_blended(predict_fn, channels: dict[str, np.ndarray], tile: int = 1024,
                    overlap: int = 128, tta: bool = True) -> np.ndarray:
    """Accumulate sum(p*w) and sum(w) over overlapping tiles; return sum(p*w)/sum(w)."""
    raise NotImplementedError("Roadmap Step 4.4")


def seam_test(prob: np.ndarray, truth: np.ndarray, tile: int, band: int = 64) -> dict:
    """Metrics within `band` px of tile edges vs tile centres (acceptance: no meaningful gap)."""
    raise NotImplementedError("Roadmap Step 4.5")
