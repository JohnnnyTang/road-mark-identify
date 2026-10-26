"""Vector annotations rasterize onto the Stage 1 grid in the right place, and the simulated
paint-probability raster follows the Stage 2 output contract."""

import numpy as np
import rasterio

from roadmark.io.labels import IGNORE, paint_labels
from roadmark.io.raster import Grid
from roadmark.synthetic import load_truth, simulate_paint_prob


def _grid(src) -> Grid:
    t = src.transform
    return Grid(t.c, t.f, t.a, src.width, src.height)


def test_truth_labels_align_with_intensity(stage1_run, synth_root):
    _, out = stage1_run
    truth = load_truth(synth_root / "synthetic" / "truth.json")
    with rasterio.open(out / "coverage.tif") as c, rasterio.open(out / "int_mean.tif") as i:
        grid, cov, im = _grid(c), c.read(1), i.read(1)
    lab = paint_labels([g for _, g in truth], grid, cov)
    assert (lab[cov == 0] == IGNORE).all()  # no data is ignored, never background
    obs = lab != IGNORE
    bright = im > 2.5 * np.nanmedian(im)
    paint = lab == 1
    iou = (bright & paint & obs).sum() / ((bright | paint) & obs).sum()
    assert iou > 0.6, iou  # mixed edge pixels keep this below 1 at 5 cm


def test_simulated_prob_contract(stage1_run, synth_root):
    _, out = stage1_run
    truth = load_truth(synth_root / "synthetic" / "truth.json")
    with rasterio.open(out / "coverage.tif") as c:
        grid, cov = _grid(c), c.read(1)
    lab = paint_labels([g for _, g in truth], grid, cov)
    p = simulate_paint_prob(lab == 1, cov, seed=0)
    assert p.dtype == np.float32
    assert np.isnan(p[cov == 0]).all() and np.nanmin(p) >= 0 and np.nanmax(p) <= 1
    assert np.nanmean(p[lab == 1]) > 0.5 > np.nanmean(p[lab == 0])
