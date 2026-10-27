import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from roadmark.io.channel_probe import ProbeCfg, run_probe
from roadmark.io.pointcloud import iter_chunks, write_las

PS = 0.03


@pytest.fixture(scope="module")
def ground(synth_root):
    pts = next(iter_chunks(synth_root / "synthetic" / "pass_0.laz", 10**8))
    g = pts["classification"] == 2
    return {k: v[g] for k, v in pts.items()}


def _render(root, sub, g, agg):
    """Known encodings: density = 5 * count, intensity = 4 * max|mean, height = z-range in cm.
    The world-file origin is the pixel corner, so the GeoTIFF origin is off by half a pixel."""
    x, y, z = g["x"], g["y"], g["z"]
    it = np.minimum(g["intensity"] // 256, 60).astype(np.float64)  # 8-bit-like point intensity
    x0, ytop = math.floor(x.min()) - 1.0, math.ceil(y.max()) + 1.0
    nc, nr = int((x.max() + 1 - x0) / PS) + 1, int((ytop - y.min() + 1) / PS) + 1
    col = np.floor((x - x0) / PS).astype(np.int64)
    row = np.floor((ytop - y) / PS).astype(np.int64)
    flat = row * nc + col
    cnt = np.bincount(flat, minlength=nr * nc)
    if agg == "max":
        iv = np.zeros(nr * nc)
        np.maximum.at(iv, flat, it)
    else:
        iv = np.bincount(flat, it, minlength=nr * nc) / np.maximum(cnt, 1)
    zmax = np.full(nr * nc, -np.inf)
    zmin = np.full(nr * nc, np.inf)
    np.maximum.at(zmax, flat, z)
    np.minimum.at(zmin, flat, z)
    zr = np.where(cnt > 0, (zmax - zmin) * 100, 0)
    chans = {"density": np.minimum(cnt * 5, 255), "intensity": np.minimum(iv * 4, 255),
             "height": np.minimum(zr, 255)}
    for ch, a in chans.items():
        (root / sub / ch).mkdir(parents=True)
        with rasterio.open(root / sub / ch / "0_0.tif", "w", driver="GTiff", width=nc, height=nr,
                           count=1, dtype="uint8",
                           transform=from_origin(x0 - PS / 2, ytop + PS / 2, PS, PS)) as ds:
            ds.write(np.round(a).reshape(nr, nc).astype(np.uint8), 1)
    keep = np.arange(len(x)) % 60 == 0  # the "thinned" delivery: a subset of the source points
    write_las(root / sub / "thin.las", {"x": x[keep], "y": y[keep], "z": z[keep],
                                        "intensity": it[keep].astype(np.uint16),
                                        "classification": np.zeros(int(keep.sum()), np.uint8)})


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.parametrize("agg", ["max", "mean"])
def test_probe_recovers_encodings(synth_root, ground, agg):
    sub = f"probe_{agg}"
    _render(synth_root, sub, ground, agg)
    cfg = ProbeCfg(channels={ch: f"{sub}/{ch}/*.tif" for ch in ("density", "intensity", "height")},
                   points=[f"{sub}/thin.las"], n_tiles=1, workers=1)
    rep, _ = run_probe(cfg, synth_root)
    d = rep["density"]
    assert abs(d["k_poisson"]["median"] - 5) < 1.0 and d["share_not_multiple_of_k"] == 0
    assert rep["points_on_pixels"]["share_on_empty_pixel"] == 0  # corner fix applied
    single = rep["raster_int_vs_point_int"]["density=5"]
    assert single["share_at_modal_value"] > 0.99 and single["ratio_raster_over_point"]["p50"] == 4
    by_n = rep["intensity_by_density_value"]
    med1, med3 = by_n["5"]["p50"], by_n["15"]["p50"]
    assert (med3 > med1 * 1.1) if agg == "max" else (abs(med3 - med1) <= 0.1 * med1 + 4)
    assert rep["height_by_density_value"]["5"]["zero_frac"] == 1.0  # z-range of one point
    assert rep["height_by_density_value"]["15"]["zero_frac"] < 0.5
