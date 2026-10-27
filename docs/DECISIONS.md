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

## 2026-10-26  Step 1.1  Intake of the first delivery (`thin_result`)
Decision:
- The delivery sits unchanged and read-only under `<data_root>/raw/thin_result/` with its three
  folders: `g` (ground), `g_refine` (refined ground, a subset of `g`'s tiles) and `ng`
  (non-ground). Tiles are 100 m. Each folder also holds a tile-index shapefile, which Stage 1
  doesn't use. Inputs are selected by the globs `*_g.las` / `*_ng.las`. This skips one stray
  COPC copy of a near-empty tile in `g`.
- New `roadmark inventory` (`io/inventory.py`): a streaming, redacted per-group summary
  (classes, passes, GPS-time spans, intensity percentiles, points per occupied 1 m cell, tile-set
  overlap). Closes the "extend inspect" TODO of 1.1.
- **The delivery is not usable for Stage 1 at the designed resolution.** It is a thinned
  product. It's kept as a coarse ground/road reference and for picking pilot areas. The
  full-density source point cloud is requested (§4.6).
Evidence (run `thin_result`, `runs/thin_result/inventory/`):
- Ground density on occupied 1 m cells: median 16 pts/m² (p25 14, p75 17; the narrow spread
  points to regular thinning) for both `g` and `g_refine`. Roadmap 1.4 needs ≥ 5 100–10 600
  pts/m² for 2.5–3 cm pixels, so this is short by a factor of ~300–600. That is ~1 point per
  0.06 m², so a 15 cm line gets ~2 points per metre.
- `classification` = 0 for every point. Ground vs non-ground exists only as the folder split.
- One `point_source_id` value, and GPS-time span 0 s: no pass separation, so no trajectory
  join (1.3) and no per-pass intensity normalisation (1.5).
- Intensity is 8-bit (max 255). Ground median 2, p99 19: very coarse quantisation for
  paint/asphalt contrast.
- No CRS in the LAS headers (EPSG unknown). LAS 1.x point format 1, 1 mm scale, no RGB.
- `g_refine` drops 32 of `g`'s 741 tiles, all near-empty. `ng` has 272 tiles without ground.
Alternatives considered: running Stage 1 on this data at ~25 cm pixels as a preview. That
was rejected as an M1 basis: one point per pixel can't separate a line from noise, so labels
drawn on it would not survive.
Consequences / follow-ups:
- Ask the data provider for the un-thinned point cloud with per-pass `point_source_id`,
  real `gps_time`, 16-bit intensity if the scanner records it, the trajectory/SBET, and the
  EPSG. Check whether the delivered ground split (`g` vs `g_refine`) exists for the full data.
- Stage 1 needs a ground method that takes ground from a pre-split file set (classification
  is 0 here). Add it when the full-density delivery arrives in the same layout.

## 2026-10-26  Step 1.8  Delivered rasters (`raster_grid_g`) are the primary input
Decision:
- No original point cloud and no CRS will ever be available. All work happens in the
  delivered local grid reference (`allow_local_crs: true`). Georeferencing comes only from
  the world files.
- The delivered raster tiles sit unchanged and read-only under `<data_root>/raw/raster_grid_g/`
  (`density`, `height`, `intensity`; 193 tiles each). The `.aux.xml` sidecars and an
  `intensity.zip` stay as delivered and are not used.
- The rasters, not the thinned points, are the primary paint evidence. The thinned ground
  points (`thin_result/g_refine`) serve as the 3D reference (ground surface, curbs).
- New redacted tools: `roadmark raster-inventory` and `roadmark crosscheck`
  (`io/raster_inventory.py`, `io/crosscheck.py`).
Evidence (runs `raster_grid_g`; crosscheck on the 12 and then 4 densest tiles):
- 3 cm pixels, tiles 10 800 px = 324 m, uint8, single band, no nodata value, no CRS. All
  three channels share identical grids.
- Tile position = 300 m × name index (r² = 1, residual 0 on all 193 tiles), so neighbouring
  tiles overlap by 24 m.
- World files hold the pixel *corner*, not the centre. Shifting by (−½, −½) px raises the
  share of ground points on occupied pixels from 0.79–0.86 to 0.985–0.991 on every tile.
  Corrected origins lie on the 3 cm grid. The same frame as the point tiles is confirmed.
- No-data: density = 0 ⇒ intensity = 0 and height = 0 in 100 % of pixels. Where density > 0,
  intensity is 0 in only 1.4 %. So coverage = (density > 0), and intensity/height are NaN where
  coverage = 0.
- Intensity: Spearman 0.62–0.75 against the thinned points' intensity on every tile, with a
  different (stretched) scale. Usable as relative intensity.
- Density: on occupied pixels p25/p50 = 5, p75 = 10, p95 = 15, p99 = 30, with 50 distinct
  values. Units unknown. If it is 5 × count, that is ≈ 1 point per 3 cm pixel (~1 000 pts/m²).
- Height: 11 distinct values, 57 % zeros on occupied pixels, r² ≤ 0.24 against point z. Not
  elevation; encoding unknown.
- Thinned points cannot explain the raster coverage (16 pts/m² would fill ~0.1 % of pixels),
  so the rasters were made from much denser data.
Consequences / follow-ups:
- Adapter delivered rasters → I1 channel rasters: apply the half-pixel origin fix, mosaic the
  overlapping tiles, set NaN where density = 0, record the channel semantics in the manifest.
  Check that the 24 m overlaps agree before mosaicking.
- Pixel size: at ≈ 1 point per 3 cm pixel, sampling no-data is ≈ 35–40 %. Evaluate a 6 cm
  aggregation (≈ 4 points per pixel) against Roadmap 1.4.
- Ask whoever produced the rasters what the density and height values mean, and whether
  intensity is a per-pixel max or mean.
- No trajectory, passes or GPS time: intensity is normalised against local asphalt only.
  Steps 1.3 and 1.5 (range/incidence model, per-pass gains) do not apply.

## 2026-10-26  Step 1.8  Channel encodings of the delivered rasters (inferred)
Decision: the encodings were inferred from the rasters and the thinned points
(`roadmark probe-channels`, run `raster_grid_g`, 4 densest tiles). The adapter uses:
- **density** = floor(255 · min(count, 50) / 50), i.e. 5.1 per point, saturating at 50
  points. count = round(value · 50 / 255).
- **intensity** = floor(255 · min(I_max, 45) / 45), where I_max is the **maximum** point
  intensity (8-bit LAS scale) in the pixel. Maps to I1 `int_max`. `int_mean` / `int_p90` are
  not recoverable and are left NaN.
- **height**: not elevation and not height above ground. Not used. `ground_z` and curb
  evidence come from the thinned ground points.
Evidence:
- Density: 99.7 % of covered pixels sit on the 5.1-step ladder (5, 10, 15 … 45, 51, 56, 61:
  floor(n·5.1)). An independent Poisson estimate per 30 cm window gives k = 4.69 (IQR
  4.14–5.15) per point. Count shares on covered pixels: 1 pt 60 %, 2 pts 24 %, 3 pts 9 %, which
  fits Poisson λ ≈ 0.85 points per 3 cm pixel (~950 pts/m²).
- Intensity: on single-point pixels holding a delivered point, raster = floor(5.667 · I) for
  I = 1…11 (5, 11, 17, 22, 28, 34, 39, 45, 51, 56, 62). Median ratio 5.6; 75 % exactly at the
  modal value; Spearman 0.85. On 2–3-point pixels the raster value's p25 equals the
  single-point value of the delivered point even for bright points. A mean would fall below it
  (paint next to asphalt), so the value is the max. The raster median also rises with point
  count (17 at 1 point → 45 at 8), as expected for a max.
- Height: the main values are 0, 85, 127, 170, 255 (255 × {0, ⅓, ½, ⅔, 1}; 63 and 191
  rare). Spearman 0.03 against the points' height above a 1 m local ground. Its zero share
  falls with point count (0.53 at 1 point → 0.27 at 8). It looks like a local relative-height
  ratio, and its meaning can't be recovered.
- 1.3 % of delivered points fall on empty pixels, and 75 % of single-point pairs are exact: the
  thinned points are (almost entirely) a subset of the raster's source points.
Consequences:
- λ ≈ 0.85 per 3 cm pixel means ~43 % sampling no-data on observed road. Aggregating 2×2
  (6 cm, λ ≈ 3.4) gives ~3 %, and 3×3 (9 cm, λ ≈ 7.6) gives < 0.1 %. Roadmap 1.4 targets
  < 2 %, so measure on the real road surface before choosing. Aggregation: count = sum,
  int_max = max (exact for a max channel).
- A per-pixel max rises with point count, so local asphalt normalisation must be conditioned on
  count, or done after aggregation to a near-uniform count.
- Point intensities above 45 saturate at 255 (ground p99.9 = 33, so this is rare and mostly
  retroreflective paint).

## 2026-10-26  Step 1.8  Raster adapter: delivered tiles → I1 channels (M1)
Decision (`roadmark adapt-rasters`, `stage1_raster/raster_adapter.py`):
- Grid: world-file origin + (½, −½) px (`corner_world_file`). Each tile must then sit on one
  integer pixel lattice, otherwise the run stops (no silent rounding). The output grid is the
  bounding box of the selected tiles, snapped to the aggregation factor.
- Decoding with lookup tables from the inferred encodings: count = rint(v / 5.1) capped at 50,
  `int_max` = rint(v · 45 / 255) in 8-bit point-intensity units (exact for integer
  intensities ≤ 45). density = 0 ⇒ `int_max` NaN. Not range/incidence corrected.
- Overlaps: each pixel comes from the covering tile whose centre is nearest (ties: first tile
  by name). On the regular 300 m lattice this is each tile's 300 m core; a tile's margin is
  used only where no neighbour exists. Before mosaicking, each overlapping pair is compared
  (density equal, intensity equal, owner empty while the other is occupied), with totals and
  the worst pair in `qa_adapter.json`.
- `coverage_min_points` = 1 for the delivery (≈ 1 point per 3 cm pixel, and one point is
  evidence). `int_mean`, `int_p90`, `hag_max`, `ground_z` are written all-NaN, so the I1 file
  set is complete and the missing evidence reads as no data, never zero. The manifest's
  `channels` field records each channel's meaning.
- Optional k × k aggregation (count = sum, `int_max` = max) for the 3 vs 6 cm decision.
- A single mosaic per channel, capped by `max_grid_mpx` (default 1500 Mpx): select pilot
  tiles by glob. TODO(scale): per-tile COGs + VRT for the full area.
Alternatives considered: first-valid / fill-from-neighbour overlap rules. Rejected until the
overlap report shows a need, because mixing tiles inside one pixel neighbourhood hides
disagreements.
Evidence: synthetic delivery (`tests/test_raster_adapter.py`): 0.1 m pixels, 12 m tiles every
10 m, one tile missing, every tile's non-owned part corrupted. Density, `int_max` and coverage
match the truth exactly at native resolution and after 2 × 2 aggregation; the overlap report
detects the corruption (intensity-equal < 1, density-equal = 1).
Real pilot (run `pilot_a`, `roadmark adapt-rasters --pick 2`): the 2 × 2 group of neighbouring
tiles with the most occupied pixels. Tile names are in `runs/pilot_a/pilot_tiles.json` only.
The picker considered 102 complete groups. This group has 2.45 × the occupied pixels of the
median group, and its tiles rank 1, 2, 25 and 68 of 193 by occupancy.
- Grid 20 800 × 20 800 px at 3 cm (433 Mpx), 4 tiles.
- Overlaps: 6 pairs, 3 of which contain points (the other 3 overlaps are empty). On all
  2.56 M occupied overlap pixels, every overlap is **identical**: both occupied = 1.0,
  density equal = 1.0, intensity equal = 1.0, owner empty while the other is occupied = 0.0,
  including in the worst pair. The tiles look like cuts from one mosaic, so the ownership rule
  doesn't change any value.
- Coverage: 16.2 % of the grid occupied. Points per occupied pixel: p25 = p50 = 1, p75 = 2,
  p95 = 4. Single-point pixels make up 59.9 % (the 60 % found by the probe). Density saturated
  on 0.01 %, intensity on 0.31 %.
- Fixed in the process: pairs whose overlap has no points were scored 0 in `worst_pair`.
  They are now left out (`pairs_with_points`).
- Visual QA in QGIS (user, 2026-10-26): passed. Tile seams are invisible and markings are
  visible in `int_max`. This closes the 1.8 adapter item, and **M1 (preview rasters) is
  reached** with `pilot_a`.
Follow-ups: Track C can start annotating on `pilot_a` (vectors, I3); Track F can run on real
data. Track F's classical score is written against `int_mean`, which this delivery lacks:
switch it to `int_max`, conditioned on count. Open in 1.8: the 3 vs 6 cm pixel size.
