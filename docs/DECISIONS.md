# Decision log

One entry per decision that closes a roadmap sub-step: what was chosen, why, and the evidence
(metric values, counts). **No coordinates, place names, extents, screenshots or raw data.**
Site-identifying values belong in `configs/local/` and the data root.

Template:

```
## YYYY-MM-DD  Step x.y  <title>
Decision:
Alternatives considered:
Evidence: (metrics, counts; reference to the run name and config hash in the data root)
Consequences / follow-ups:
```

---

## 2026-10-26  Step 0  Project setup
Decision: Python 3.12 + uv; laspy/rasterio/GDAL stack (PDAL optional via CLI, not a pip dep);
CSF for ground filtering when the delivered classification is not trusted; Apache-2.0 licence
for the code; data never in git (DATA_POLICY.md).
Evidence: Stage 1 runs end-to-end on the synthetic scene. The 2.5 cm grid of a 60 x 16 m
scene builds in ~20 s. Cross-pass consistency is 6.7 % (tol 10 %), and the pass-2 gain is recovered as 1.43
(true 1/0.7).
Consequences:
- Intensity fit must use in-road ground points: with sidewalks included, a + b was biased
  (1.73 vs true 2.0). Stage 1 therefore builds the road mask before fitting intensity.
- On flat roads range and incidence are collinear (corr ≈ 1); only a + b is identifiable.
  Handled explicitly (`identifiable` flag).
- Sampling no-data: 3.75 pts/px with `coverage_min_points: 2` gives ~11 % false no-data.
  Pixel size and threshold must come from measured density (Roadmap 1.4).

## 2026-10-26  Planning  Parallel tracks, data root, GPU machine
Decision:
- Organise the work as six tracks (A data, B specs, C annotation, D downstream, E model,
  F classical) that meet at milestones M1–M5 through frozen interfaces I1–I8.
- Annotations are vectors in the project CRS and get rasterized onto any grid (`io/labels.py`), so
  annotation starts on preview rasters (M1) and survives Stage 1 tuning.
- Stages 3–7 are developed against `simulate_paint_prob` on the synthetic truth before a model exists.
- Data root defaults to `<repo>/data` (fully git-ignored, guarded, AI-denied). `ROADMARK_DATA_ROOT`
  overrides it.
- Model code runs on a separate NVIDIA GPU machine: extra `seg-cuda` = PyTorch 2.14.1 from the
  cu130 index (driver ≥ 580, Turing or newer). The cu128 index stops at torch 2.11. The fallback is
  cu126. `seg-cpu` (same torch version) is for smoke tests.
Alternatives considered: strictly sequential steps (§12), rejected because annotation and model
development would idle until Stage 1 is final. Pixel-mask annotation, rejected because it is tied
to one grid.
Consequences: data moves CPU ↔ GPU by hand inside the data boundary (DATA_POLICY "Compute
machines"). GPU driver version to confirm in Step 1.0.
