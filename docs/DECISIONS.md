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
