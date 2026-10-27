import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from roadmark.io.crosscheck import CrosscheckCfg, run_crosscheck
from roadmark.io.pointcloud import iter_chunks

PS, W = 0.1, 128.0  # pixel size, tile size (m): the synthetic scene fits in one tile


def _write(path, arr, x0, ytop):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0],
                       count=1, dtype="uint8", transform=from_origin(x0, ytop, PS, PS)) as ds:
        ds.write(arr, 1)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_crosscheck_registration_and_encoding(synth_root):
    pts = next(iter_chunks(synth_root / "synthetic" / "pass_0.laz", 10**8))
    g = pts["classification"] == 2
    x, y, z, it = pts["x"][g], pts["y"][g], pts["z"][g], pts["intensity"][g]
    x0, ytop, n = math.floor(x.min()) - 1.0, math.ceil(y.max()) + 1.0, int(W / PS)
    assert x.max() < x0 + W and y.min() > ytop - W
    col = np.floor((x - x0) / PS).astype(int)
    row = np.floor((ytop - y) / PS).astype(int)
    dens = np.zeros((n, n), np.int64)
    np.add.at(dens, (row, col), 1)
    imax = np.zeros((n, n))
    np.maximum.at(imax, (row, col), it / it.max() * 254 + 1)
    hgt = np.zeros((n, n))
    np.maximum.at(hgt, (row, col), (z - z.min()) / 0.01 + 1)  # 1 cm per level
    for ch in ("density", "intensity", "height"):
        (synth_root / "rast" / ch).mkdir(parents=True)
    arrs = {"density": np.minimum(dens, 255), "intensity": imax, "height": np.minimum(hgt, 255)}
    for ch, a in arrs.items():
        _write(synth_root / "rast" / ch / "t_0.tif", a.astype(np.uint8), x0, ytop)
        # Same name, but a tile somewhere else: no points may land on it.
        _write(synth_root / "rast" / ch / "a_b.tif", a.astype(np.uint8), x0 + 10 * W, ytop)
    cfg = CrosscheckCfg(channels={ch: f"rast/{ch}/*.tif" for ch in arrs},
                        points=["synthetic/pass_0.laz"], n_tiles=2, workers=1)
    rep, files = run_crosscheck(cfg, synth_root)
    assert rep["tiles_checked"] == 2 and len(files) == 6
    regs = sorted(rep["registration_per_tile"], key=lambda r: r["points"])
    assert regs[0] == {"points": 0}
    r = regs[1]
    # Same frame: every ground point lands on density > 0 (the file also holds car points).
    assert r["on_occupied"] >= round(g.mean(), 4) - 1e-3
    assert r["intensity_spearman_vs_intensity"] > 0.5
    hz = r["height_vs_z"]
    assert hz["r2"] > 0.9 and abs(hz["slope_m_per_level"] - 0.01) < 0.003
    # 0 means "no data" here: intensity is never 0 where density > 0, and always 0 elsewhere.
    assert rep["where_density_gt_0"]["intensity"]["zero_frac"] == 0.0
    assert rep["where_density_eq_0"]["intensity"]["zero_frac"] == 1.0
    assert rep["origin_offset_px"] == [(0.0, 0.0)]


def test_value_step():
    from roadmark.io.crosscheck import value_step

    assert value_step(np.bincount([0, 0, 5, 10, 15, 30])) == 5
    assert value_step(np.bincount([0, 3, 4])) == 1
    assert value_step(np.zeros(3, int)) == 0
