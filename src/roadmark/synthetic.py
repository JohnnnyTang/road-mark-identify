"""Synthetic MLS scene in local coordinates (0..L m): the only data allowed near the repo.

Contents: carriageway with curbs and raised sidewalks, solid edge lines, dashed lane lines
(2 m / 4 m), a double centre line, a stop line, a zebra crosswalk, a straight arrow, and a
parked car that occludes paint (true no-data, not "no paint"). Two scan passes in opposite
directions with different sensor gain, and intensity that falls off with range/incidence
according to known ``a_true``/``b_true``, so Stage 1 corrections can be checked against truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon, box

from roadmark.io.pointcloud import write_las


@dataclass
class SceneParams:
    length: float = 60.0
    half_road: float = 6.0  # curb at |y| = half_road
    half_total: float = 8.0  # sidewalk outer edge
    curb_height: float = 0.15
    density: float = 3000.0  # ground points per m^2 per pass
    a_true: float = 1.2
    b_true: float = 0.8
    gains: tuple[float, float] = (1.0, 0.7)
    scanner_y: tuple[float, float] = (-3.8, 1.8)
    scanner_z: float = 2.0
    speed: float = 5.0  # m/s
    r_ref: float = 10.0
    scale: float = 6000.0
    noise_sigma: float = 0.08  # multiplicative, lognormal
    seed: int = 0


REFL = {"asphalt": 0.10, "sidewalk": 0.20, "paint": 0.60}
CAR = {"x0": 18.0, "x1": 22.5, "y0": 3.8, "y1": 5.5, "h": 1.5}


def markings(p: SceneParams) -> list[tuple[str, Polygon]]:
    """Ground-truth marking polygons with labels (local coordinates)."""
    L, out = p.length, []
    w = 0.15
    stop_x, cw_x0, cw_x1 = 37.7, 39.0, 42.0
    for s in (-1, 1):  # solid edge lines, continuous
        out.append(("edge_line_solid", box(0, s * 5.6 - w / 2, L, s * 5.6 + w / 2)))
    for s in (-1, 1):  # double centre line (gap 0.15)
        y = s * 0.15
        for x0, x1 in ((0, stop_x - 0.2), (cw_x1 + 0.5, L)):
            out.append(("centre_line_double", box(x0, y - w / 2, x1, y + w / 2)))
    for s in (-1, 1):  # dashed lane lines 2 m / 4 m
        y = s * 2.0
        for x0, x1 in ((0, stop_x - 0.2), (cw_x1 + 0.5, L)):
            x = x0 + 0.5
            while x + 2.0 <= x1:
                out.append(("lane_line_dashed", box(x, y - w / 2, x + 2.0, y + w / 2)))
                x += 6.0
    out.append(("stop_line", box(stop_x, -5.6, stop_x + 0.3, -0.25)))
    y = -5.4
    while y + 0.45 <= 5.5:  # zebra bars: long axis along the road
        out.append(("crosswalk_bar", box(cw_x0, y, cw_x1, y + 0.45)))
        y += 1.05
    ax, ay = 30.0, -3.8  # straight arrow pointing +x in the right-hand lane
    shaft = box(ax, ay - 0.1, ax + 3.0, ay + 0.1)
    head = Polygon([(ax + 3.0, ay - 0.45), (ax + 4.5, ay), (ax + 3.0, ay + 0.45)])
    out.append(("arrow_straight", shapely.union(shaft, head)))
    return out


def _ground_points(p: SceneParams, rng: np.random.Generator, pass_i: int) -> dict:
    area = p.length * 2 * p.half_total
    n = rng.poisson(p.density * area)
    x = rng.uniform(0, p.length, n)
    y = rng.uniform(-p.half_total, p.half_total, n)
    ys, zs = p.scanner_y[pass_i], p.scanner_z

    in_car_x = (x >= CAR["x0"]) & (x <= CAR["x1"])
    occluded = in_car_x & (y >= CAR["y0"]) & (y <= CAR["y1"])
    if ys < CAR["y0"]:  # shadow behind the car seen from this pass
        behind = in_car_x & (y > CAR["y1"])
        h_at_face = zs * (y - CAR["y0"]) / np.maximum(y - ys, 1e-6)
        occluded |= behind & (h_at_face < CAR["h"])
    x, y = x[~occluded], y[~occluded]

    sidewalk = np.abs(y) > p.half_road
    z = np.where(sidewalk, p.curb_height, 0.0) + rng.normal(0, 0.003, len(x))
    refl = np.where(sidewalk, REFL["sidewalk"], REFL["asphalt"])
    refl = refl * rng.normal(1.0, 0.1, len(x))
    paint = np.zeros(len(x), bool)
    for _, poly in markings(p):
        paint |= shapely.contains_xy(poly, x, y)
    refl[paint] = REFL["paint"] * rng.normal(1.0, 0.05, paint.sum())
    return {"x": x, "y": y, "z": z, "refl": refl, "cls": np.full(len(x), 2, np.uint8)}


def _car_points(p: SceneParams, rng: np.random.Generator) -> dict:
    n_top = int(500 * (CAR["x1"] - CAR["x0"]) * (CAR["y1"] - CAR["y0"]))
    n_side = int(500 * (CAR["x1"] - CAR["x0"]) * (CAR["h"] - 0.3))
    x = np.concatenate([rng.uniform(CAR["x0"], CAR["x1"], n_top),
                        rng.uniform(CAR["x0"], CAR["x1"], n_side)])
    y = np.concatenate([rng.uniform(CAR["y0"], CAR["y1"], n_top), np.full(n_side, CAR["y0"])])
    z = np.concatenate([np.full(n_top, CAR["h"]), rng.uniform(0.3, CAR["h"], n_side)])
    return {"x": x, "y": y, "z": z, "refl": np.full(len(x), 0.3),
            "cls": np.full(len(x), 1, np.uint8)}


def load_truth(path: Path) -> list[tuple[str, Polygon]]:
    """(label, polygon) pairs from a truth.json written by ``generate``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(m["label"], shapely.from_wkt(m["wkt"])) for m in data["markings"]]


def simulate_paint_prob(paint: np.ndarray, coverage: np.ndarray, seed: int = 0,
                        blur_px: float = 1.0, noise: float = 0.08, dropout: float = 0.02,
                        false_blobs: int = 20) -> np.ndarray:
    """Stand-in for a Stage 2 model output, so Stages 3-7 can be built before the model exists.

    Blurred truth + noise + random paint dropouts (wear) + small false-positive blobs.
    NaN where coverage == 0, exactly like the real contract (``paint_prob.tif``).
    """
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    p = ndimage.gaussian_filter(paint.astype(np.float32), blur_px)
    drop = ndimage.binary_dilation(rng.random(paint.shape) < dropout / 25, iterations=2)
    p[drop] *= 0.2
    for _ in range(false_blobs):
        r, c = rng.integers(0, paint.shape[0]), rng.integers(0, paint.shape[1])
        p[r:r + 3, c:c + 3] = np.maximum(p[r:r + 3, c:c + 3], rng.uniform(0.4, 0.9))
    p = np.clip(p + rng.normal(0, noise, p.shape), 0, 1).astype(np.float32)
    p[coverage == 0] = np.nan
    return p


def generate(out_dir: Path, p: SceneParams | None = None) -> dict:
    """Write pass_0.laz, pass_1.laz, trajectory.npz and truth.json into ``out_dir``."""
    p = p or SceneParams()
    rng = np.random.default_rng(p.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0s = (0.0, 1000.0)
    traj_parts = []
    counts = {}
    for i in (0, 1):
        g = _ground_points(p, rng, i)
        if i == 0:
            car = _car_points(p, rng)
            g = {k: np.concatenate([g[k], car[k]]) for k in g}
        forward = i == 0  # pass 1 drives in -x
        t = t0s[i] + (g["x"] if forward else p.length - g["x"]) / p.speed
        ys, zs = p.scanner_y[i], p.scanner_z
        rngm = np.sqrt((g["y"] - ys) ** 2 + (g["z"] - zs) ** 2)
        cos_inc = np.abs(zs - g["z"]) / rngm
        inten = (p.scale * g["refl"] * p.gains[i] * (rngm / p.r_ref) ** (-p.a_true)
                 * cos_inc ** p.b_true * rng.lognormal(0, p.noise_sigma, len(rngm)))
        arrays = {"x": g["x"], "y": g["y"], "z": g["z"],
                  "intensity": np.clip(inten, 1, 65535).astype(np.uint16),
                  "classification": g["cls"], "gps_time": t,
                  "point_source_id": np.full(len(t), i + 1, np.uint16)}
        write_las(out_dir / f"pass_{i}.laz", arrays)
        counts[f"pass_{i}"] = len(t)
        tt = np.arange(0, p.length / p.speed + 0.01, 0.01)
        traj_parts.append(np.column_stack([
            t0s[i] + tt, (tt * p.speed) if forward else (p.length - tt * p.speed),
            np.full(len(tt), ys), np.full(len(tt), zs)]))
    traj = np.concatenate(traj_parts)
    np.savez(out_dir / "trajectory.npz", t=traj[:, 0], x=traj[:, 1], y=traj[:, 2], z=traj[:, 3])
    truth = {"params": asdict(p), "car": CAR,
             "markings": [{"label": lab, "wkt": poly.wkt} for lab, poly in markings(p)]}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return counts
