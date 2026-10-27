import json

import numpy as np

from roadmark.io.inventory import InventoryCfg, hist_quantiles, run_inventory, tile_sets
from roadmark.io.pointcloud import iter_chunks, read_header


def test_inventory_on_synthetic(synth_root):
    cfg = InventoryCfg(groups={"p0": ["synthetic/pass_0.laz"], "both": ["synthetic/pass_*.laz"]},
                       workers=1)
    rep, files = run_inventory(cfg, synth_root)
    g = rep["groups"]["both"]
    hdrs = [read_header(f) for f in sorted((synth_root / "synthetic").glob("pass_*.laz"))]
    assert len(files) == 3 and g["files"] == 2
    assert g["points"] == sum(h.point_count for h in hdrs)
    assert sum(g["classification"].values()) == g["points"] and "2" in g["classification"]
    assert g["passes"]["count"] == 2 and g["passes"]["span_s"]["max"] > 0
    pts = next(iter_chunks(hdrs[0].path, 10**8))
    assert rep["groups"]["p0"]["intensity"]["max"] == int(pts["intensity"].max())
    assert g["density_on_occupied_pts_m2"]["p50"] > 0
    assert g["occupied_area_m2"] <= sum(h.summary()["extent"]["area_m2"] for h in hdrs) + 200
    # Redacted: no bounds or absolute times anywhere in the report.
    text = json.dumps(rep)
    assert "bounds" not in text and "offset" not in text
    assert set(rep["tiles"]) == {"p0|both"}  # tile keys: see test_tile_sets_match_across_suffixes


def test_hist_quantiles():
    h = np.bincount([1, 2, 2, 3, 10])
    q = hist_quantiles(h, (0.0, 0.5, 1.0))
    assert q == {"p0": 1.0, "p50": 2.0, "p100": 10.0}
    assert hist_quantiles(np.zeros(3, int)) == {}


def test_tile_sets_match_across_suffixes():
    t = tile_sets({"g": ["grid_1_2_g.las", "grid_1_3_g.las", "grid_1_3_g.copc.laz"],
                   "ng": ["grid_1_2_ng.las", "grid_5_5_ng.las"]})
    assert t["g|ng"] == {"both": 1, "only_g": 1, "only_ng": 1}
