"""Intensity correction (§4.2).

1. Range/incidence model fitted on unpainted asphalt:
       I_corr = I * (R / R_ref)^a / cos(theta)^b
   Taking logs on asphalt (constant true reflectance):
       log I = c_pass - a * log(R / R_ref) + b * log(cos theta)
   so a, b come from least squares. They are estimated, never assumed.
   Asphalt is identified by *residual trimming*: fit on all ground points, keep those within
   k*MAD of the median residual (asphalt is the dominant surface; paint and sidewalks are
   outliers), refit, repeat. Selecting by raw-intensity quantiles instead would select on the
   dependent variable, which is itself range-dependent, and bias a and b badly.
   Identifiability: on a flat road seen by a profile scanner, cos(theta) = h_scanner / R, so
   log R and log cos(theta) are collinear and only a + b is determined. When
   |corr(log R, log cos)| > ``collinear_corr`` the fit fixes b = ``b_prior`` and estimates a
   alone; the report marks ``identifiable = false``. The correction is equivalent on the road
   surface either way; it only matters off-plane (curbs, ramps), so say so in the QA report.
2. Per-pass normalisation: affine map of each pass's corrected asphalt quantiles onto the
   reference pass. Affine (not full quantile mapping) so that paint, which lies above the
   asphalt quantiles, is extrapolated rather than clipped.
3. Validation lives in ``qa.py`` (cross-pass agreement on the same markings).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from roadmark.config import IntensityCfg


@dataclass
class RangeIncidenceModel:
    a: float = 0.0
    b: float = 0.0
    r_ref: float = 10.0
    r2: float = float("nan")  # on the asphalt (kept) samples
    n_samples: int = 0
    identifiable: bool = True  # False: a, b collinear; b fixed to the prior
    corr_logr_logcos: float = float("nan")
    asphalt_fraction: float = float("nan")  # share of ground samples kept as asphalt
    asphalt_median: float = float("nan")  # corrected intensity of asphalt (after pass match)
    # pass_id -> (gain, offset) applied after the range/incidence correction
    pass_affine: dict[int, tuple[float, float]] = field(default_factory=dict)
    reference_pass: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pass_affine"] = {str(k): v for k, v in self.pass_affine.items()}
        return d

    def correct(self, intensity: np.ndarray, rng: np.ndarray, cos_inc: np.ndarray,
                pass_id: np.ndarray | None = None) -> np.ndarray:
        out = intensity.astype(np.float64) * (rng / self.r_ref) ** self.a / cos_inc ** self.b
        if pass_id is not None and self.pass_affine:
            for pid, (g, o) in self.pass_affine.items():
                m = pass_id == pid
                out[m] = g * out[m] + o
        return out.astype(np.float32)


def scanner_geometry(pts: dict[str, np.ndarray], traj: dict[str, np.ndarray] | None,
                     min_cos: float, time_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Range R and cos(incidence) per point, from scanner positions interpolated by GPS time.

    Incidence uses a vertical surface normal (adequate on road surfaces; a DEM normal can be
    substituted later if cross-slope matters).
    """
    if traj is None or "gps_time" not in pts:
        raise ValueError("range/incidence correction needs gps_time and a trajectory "
                         "(stage1.trajectory.path), or disable stage1.intensity")
    t = pts["gps_time"].astype(np.float64)
    tt = traj["t"] + time_offset
    sx, sy, sz = (np.interp(t, tt, traj[k]) for k in ("x", "y", "z"))
    dx, dy, dz = pts["x"] - sx, pts["y"] - sy, pts["z"] - sz
    rng = np.sqrt(dx * dx + dy * dy + dz * dz)
    cos_inc = np.clip(np.abs(dz) / np.maximum(rng, 1e-6), min_cos, 1.0)
    return rng, cos_inc


def fit_model(intensity: np.ndarray, rng: np.ndarray, cos_inc: np.ndarray,
              pass_id: np.ndarray | None, cfg: IntensityCfg, seed: int = 0
              ) -> RangeIncidenceModel:
    """Fit a, b on asphalt (found by residual trimming), then the per-pass normalisation.

    Inputs are *ground* points only (any sample of them).
    """
    idx = np.flatnonzero(intensity > 0)
    if len(idx) > cfg.fit_sample_size:
        idx = np.random.default_rng(seed).choice(idx, cfg.fit_sample_size, replace=False)
    if len(idx) < 100:
        raise ValueError(f"only {len(idx)} ground samples; cannot fit intensity model")
    pid = pass_id[idx] if pass_id is not None else np.zeros(len(idx), np.int64)
    pids = np.unique(pid)

    log_r, log_c = -np.log(rng[idx] / cfg.r_ref), np.log(cos_inc[idx])
    corr_rc = float(np.corrcoef(log_r, log_c)[0, 1])
    identifiable = abs(corr_rc) <= cfg.collinear_corr
    # Per-pass intercepts so that pass gain differences don't bias a and b.
    dummies = [(pid == p).astype(float) for p in pids]
    y = np.log(intensity[idx].astype(np.float64))
    if identifiable:
        A = np.column_stack([log_r, log_c, *dummies])
    else:  # b fixed: move its term to the left-hand side
        A = np.column_stack([log_r, *dummies])
        y = y - cfg.b_prior * log_c
    keep = np.ones(len(idx), dtype=bool)
    for _ in range(cfg.trim_iterations):
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        res = y - A @ coef
        med = np.median(res[keep])
        mad = 1.4826 * np.median(np.abs(res[keep] - med))
        new_keep = np.abs(res - med) <= cfg.trim_mad_k * mad
        if (new_keep == keep).all():
            break
        keep = new_keep
    if keep.sum() < 100:
        raise ValueError("too few asphalt samples after trimming; check ground filtering")
    coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    rk = y[keep] - A[keep] @ coef
    r2 = 1 - rk.var() / y[keep].var() if y[keep].var() > 0 else float("nan")
    model = RangeIncidenceModel(a=float(coef[0]),
                                b=float(coef[1]) if identifiable else cfg.b_prior,
                                r_ref=cfg.r_ref, r2=float(r2), n_samples=int(keep.sum()),
                                identifiable=identifiable, corr_logr_logcos=corr_rc,
                                asphalt_fraction=float(keep.mean()))

    corr = model.correct(intensity[idx][keep], rng[idx][keep], cos_inc[idx][keep])
    p_keep = pid[keep]
    if cfg.per_pass_match and pass_id is not None and len(pids) > 1:
        counts = {int(p): int((p_keep == p).sum()) for p in pids}
        ref = max(counts, key=counts.get)  # most-sampled pass is the reference
        qs = np.linspace(*cfg.match_quantiles, 17)
        q_ref = np.quantile(corr[p_keep == ref], qs)
        for p in pids:
            q_p = np.quantile(corr[p_keep == p], qs)
            g, o = np.polyfit(q_p, q_ref, 1)
            model.pass_affine[int(p)] = (float(g), float(o))
        model.reference_pass = ref
        corr = model.correct(intensity[idx][keep], rng[idx][keep], cos_inc[idx][keep], p_keep)
    model.asphalt_median = float(np.median(corr))
    return model
