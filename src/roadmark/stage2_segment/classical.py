"""Classical paint baseline (Roadmap Step 3): the floor every learned model must beat."""

from __future__ import annotations

import numpy as np


def paint_score(int_mean: np.ndarray, asphalt_median: float) -> np.ndarray:
    """Contrast of corrected intensity against asphalt; NaN stays NaN (no data)."""
    raise NotImplementedError("Roadmap Step 3.1")


def hysteresis_paint(score: np.ndarray, t_low: float, t_high: float,
                     coverage: np.ndarray) -> np.ndarray:
    """Seeds above t_high grown through pixels above t_low; no-data never becomes paint."""
    raise NotImplementedError("Roadmap Step 3.2 (skimage.filters.apply_hysteresis_threshold)")
