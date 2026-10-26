# Roadmap

How we get from delivered point clouds to reviewed, typed vector road markings, one step at a
time. It follows the handoff (`docs/handoff.md`, section numbers like §4.2 refer to it).

**Rules of the road**

- A step is finished when its **acceptance check** passes and the result is written down in
  `docs/DECISIONS.md` (parameters + reasoning, never coordinates or data). Only then does the
  next step begin.
- Precision first (§1.2): when in doubt, output `unknown` or flag for review. Never loosen
  acceptance criteria to gain recall.
- Every step is developed and unit-tested on the **synthetic scene** first (`roadmark synth`),
  then run on real data **locally**. Real data, results and screenshots never enter git (see
  `DATA_POLICY.md`).

| Step | Title | Handoff | Status |
|---|---|---|---|
| 0 | Project setup | §11 | done |
| 1 | Data foundation: extraction and preprocessing | §3, §4 | **in progress**: code skeleton runs on synthetic data |
| 2 | Guidelines and gold standard | §1.3, §5.5, §5.6 | not started |
| 3 | Classical reference pipeline | §5.5, §12.3 | not started |
| 4 | Segmentation model | §5 | not started |
| 5 | Paint elements and glyphs | §6 | not started |
| 6 | Object assembly | §7 | not started |
| 7 | Vector output and QA | §8, §9 | not started |
| 8 | Full evaluation and iteration | §10 | not started |

---

## Step 0: Project setup (done)

- [x] Repository layout per §11.3 (`src/roadmark/stage*`, `configs/`, `qa/`, `eval/`, `docs/`)
- [x] Python 3.12 environment with uv (`uv sync --extra ground`), lockfile committed
- [x] Data-leak protection: `.gitignore`, data guard (pre-commit, pre-push, CI), Claude Code
      permission rules, `DATA_POLICY.md`
- [x] Typed configs: pipeline (`configs/pipeline/`) and standard (`configs/standards/`)
- [x] Synthetic MLS scene with known truth (two passes, gain difference, range/incidence
      fall-off, occluding car, curbs, all marking types)
- [x] Stage 1 pipeline runs end-to-end on synthetic data, tests pass

**Each person, once per clone:**

```powershell
uv sync --extra ground
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
copy .env.example .env    # then set ROADMARK_DATA_ROOT, ideally outside the repo
uv run pytest
```

---

## Step 1: Data foundation: extraction and preprocessing (§3, §4)

**Goal:** from the delivered point clouds, produce seamless, gap-aware channel rasters
(`int_max`, `int_mean`, `int_p90`, `density`, `coverage`, `hag_max`, `ground_z`) plus a
road-surface prior, with intensity that is consistent across ranges and scan passes.

**Acceptance check (§12.1):**
1. Cross-pass intensity consistency on the same markings within the agreed tolerance
   (`qa_stage1.json → cross_pass_consistency.passed`).
2. The density/coverage channel separates occlusions from observed ground: on the pilot
   areas, sampling no-data on observed road < 2 % of cells, and every occlusion (vehicle,
   pole shadow) shows as contiguous no-data.
3. Visual check in QGIS on 3 pilot areas: no paint lost by ground filtering, curbs found.

### 1.0 Confirm inputs and open decisions (§3, §14)

Decisions to record in `docs/DECISIONS.md`. Values that reveal the project stay in
`configs/local/`.

- [ ] Source point cloud available (strongly preferred over pre-flattened rasters, §4.6)
- [ ] Point attributes present: intensity (required); GPS time, pass/source ID, scanner
      position or range, RGB (`roadmark inspect` reports them)
- [ ] Trajectory / SBET file available and its time base (GPS week seconds vs. adjusted standard time)
- [ ] CRS: projected metric EPSG for all outputs
- [ ] Jurisdiction and standard edition. `configs/standards/gb5768_3_2009.yaml` holds **draft,
      unverified** values. Check each one against the text, add missing dash patterns, record
      clause numbers, then set `verified: true`.
- [ ] Paint colour required? If yes, RGB (colourized cloud or co-registered orthophoto) is
      mandatory (§3)
- [ ] Number of scan passes per area
- [ ] Compute for training (Step 4): this machine has no NVIDIA GPU. Training must run on
      hardware **inside the data boundary**: no cloud GPUs unless the data owner approves.
- [ ] Pilot areas: 3 small areas (~200 m each): a straight road with dashed lines, a
      signalised intersection with crosswalks and arrows, and a street with parked cars and curbs.

### 1.1 Data intake and inventory

- [ ] Copy deliveries to `<data_root>/raw/` and make them read-only. Never modify raw files.
- [ ] `uv run roadmark inspect <data_root>/raw/*.laz` → inventory: points, density, extent
      size, EPSG, attributes. Output is redacted by default and safe to share inside the team.
- [ ] Extend `inspect` with: pass-ID histogram, GPS-time range per pass, classification
      histogram, intensity percentiles (TODO in `io/pointcloud.py`).
- [ ] Check CRS consistency across files; reproject upstream if needed (PDAL or the
      supplier), never inside the pipeline.
- **Deliverable:** inventory table in the data root; summary (counts only) in DECISIONS.md.

### 1.2 Ground filtering (§4.1)

- [ ] On the pilot areas, compare `existing` classification vs `csf` (and `grid_min` only as a
      sanity floor).
- [ ] Checks: (a) **paint must be ground**: bright points (top intensity percentile) classified
      non-ground should be ≈ 0; (b) vehicles, pedestrians, bins must be non-ground;
      (c) curb tops and islands should be ground.
- [ ] Tune CSF (`cloth_resolution`, `class_threshold`, `rigidness`) if used. Keep low
      non-ground (HAG ≤ 0.30 m) for curb detection.
- **Deliverable:** chosen method and parameters in `configs/pipeline/default.yaml`; rationale in
  DECISIONS.md.

### 1.3 Scanner geometry (needed for intensity correction)

- [ ] Load the trajectory (`stage1.trajectory.path`, columns t, x, y, z) and find the time
      offset to LAS `gps_time`.
- [ ] Sanity check: the minimum range per pass ≈ scanner height above road (≈ 2 m on a car).
      Ranges must vary smoothly along the track.
- [ ] If there is no trajectory, there are two options: (a) obtain it from the supplier (preferred), or
      (b) fall back to per-pass normalisation only (`intensity.enabled: false` plus a
      per-pass affine). Option (b) leaves range effects in and must be documented.

### 1.4 Pixel size and coverage threshold

The synthetic run showed that with 3.75 points/pixel and `coverage_min_points: 2`, **11 % of
road cells are no-data from Poisson sampling alone**. That would turn sampling noise into
"occlusion" and corrupt the three-state evidence. Choose pixel size and threshold from the
measured density:

| coverage_min_points | λ = expected points/pixel for < 1 % sampling no-data | ground density needed at 2.5 cm | at 3 cm |
|---|---|---|---|
| 1 | ≥ 4.6 | ≥ 7 400 pts/m² | ≥ 5 100 pts/m² |
| 2 | ≥ 6.6 | ≥ 10 600 pts/m² | ≥ 7 400 pts/m² |

- [ ] Per pilot area: histogram of ground density on the road surface (the near-range lanes
      are dense; far lanes and edges are sparse).
- [ ] Pick pixel size (2–3 cm, §3) and `coverage_min_points` so that sampling no-data on
      observed road is < 2 %. If density is too low for 2.5 cm, use 3 cm rather than
      accept false no-data.
- [ ] Add an *occlusion* layer: no-data blobs larger than a minimum area (e.g. 0.05 m²) after
      a small closing, distinct from isolated sampling holes (TODO in `stage1_raster/qa.py`).

### 1.5 Intensity correction (§4.2)

- [ ] Run Stage 1 on the pilot areas. Inspect `qa_stage1.json → intensity_model`.
- [ ] **Identifiability:** on flat roads seen by a profile scanner, `cos θ ≈ h / R`, so range
      and incidence are collinear and only `a + b` is determined. The fitter detects this
      (`identifiable: false`, `b` fixed to `b_prior`). The correction is still correct on the
      road plane. Record which case applies.
- [ ] Check residuals vs range and vs incidence: no remaining trend on asphalt after
      correction.
- [ ] Per-pass affine gains: plausible (close to 1, stable along the area).
- [ ] Agree the consistency tolerance with the project owner (default 10 % median relative
      difference on paint interiors) and record it.
- [ ] If consistency fails, look for these causes: a wrong trajectory time offset, multiple sensors per
      pass (normalise per sensor), or saturation/auto-gain in the raw intensity.

### 1.6 Road-surface prior (§4.4)

- [ ] Tune `curb_step_min/max`, `step_window_m` on the pilot areas.
- [ ] Check medians, traffic islands and driveways (lowered curbs). Remember that the mask is a
      **prior**: later stages flag paint outside it and never delete it.
- [ ] TODO(scale): compute the mask on a coarser grid for large areas (noted in
      `stage1_raster/pipeline.py`).

### 1.7 Production run and visual QA

- [ ] Run on all pilot areas with the chosen config. All outputs (COGs, `qa_stage1.json`,
      `manifest.json`) go to `<data_root>/runs/<name>/stage1/`.
- [ ] Create a QGIS project in the data root with intensity, density, coverage, road mask and
      curbs styled.
- [ ] Walk the acceptance checks above. Record the results in DECISIONS.md, then close Step 1.

### 1.8 Only pre-flattened rasters available? (§4.6)

- [ ] No reliable density means occlusion gaps can't be told from paint gaps. Treat all dash
      gaps as lower-confidence evidence and expect a higher review rate.
- [ ] Measure brightness consistency of known markings across the area and document it.
- [ ] Keep pushing for the point cloud before production runs.

---

## Step 2: Guidelines and gold standard (§1.3, §5.5, §5.6)

**Goal:** a frozen class list, annotation guidelines, geographic data splits and a vector gold
standard on held-out areas. **Acceptance:** inter-annotator IoU and boundary agreement measured
and disagreements resolved into the guidelines.

- [ ] Freeze the class list (§1.3) and the symbol types in scope (`symbols.classes`).
- [ ] Guidelines in `docs/annotation_guidelines.md`: what counts as paint (worn, faint), boundary
      placement (outer edge), occluded → ignore label, every class with edge cases.
      Illustrations in the public repo are **synthetic only**. A version with real example crops
      lives in the data root.
- [ ] Split by geography (whole intersections or road sections, never tiles) into train / val /
      test. The test set covers several campaigns, worn paint, heavy occlusion, curves, complex
      intersections and every class. Split definitions (polygons) live in the data root.
- [ ] Annotation tooling: QGIS project template, layer schema = §8.2 (`stage5_vector/schema.py`).
- [ ] Double-annotate ~10 % of the area, compute IoU and boundary agreement (`eval/`), and fold
      the resolutions back into the guidelines.
- [ ] Vector gold standard on the held-out areas: tracks with s-intervals, crosswalks, stop
      lines, symbols with type and heading.

## Step 3: Classical reference (§12.3)

**Goal:** a performance floor and pre-labels. **Acceptance:** metrics on val reported. This is the
baseline every later model must beat at equal or better precision.

- [ ] `stage2_segment/classical.py`: paint score = corrected intensity contrast vs asphalt,
      hysteresis threshold, morphology, size filter. No-data never becomes paint.
- [ ] Minimal Stages 3–5 (polygonize, descriptors, simple typing, GeoPackage writer) so the
      floor is measured end-to-end.
- [ ] Export pre-labels for Step 2/4 annotation, to be corrected in QGIS.

## Step 4: Segmentation model (§5)

**Goal:** calibrated paint probability rasters. **Acceptance:** pixel metrics beat the classical
floor at the precision target, and the seam test passes.

- [ ] Training tiles cut from stitched rasters of training areas. Normalise with **global**
      statistics, never per tile. No-data → ignore label.
- [ ] Model: high-resolution output (U-Net/SegFormer with a high-res decoder, or HRNet) via
      `uv sync --extra seg`. Inputs: `int_max`, `int_mean`, log `density`, optional `hag_max`/RGB.
- [ ] Loss: BCE + Dice + clDice or boundary loss. Auxiliary coarse-class head.
- [ ] Track experiments **locally** only (TensorBoard/MLflow file store in the data root).
- [ ] Calibrate with temperature scaling on val. Pick `paint_t_high/low` at the precision target
      and write them to the standards config `thresholds`.
- [ ] Tiled inference: 1024 px tiles, cosine blend, TTA (flips, 90° rotations). Set the overlap
      empirically with the half-tile shift test, then run the seam test (`stage2_segment/tiling.py`).

## Step 5: Paint elements and glyphs (§6)

**Acceptance:** element-level detection P/R at IoU ≥ 0.5; glyph confusion matrix with
open-set rejection.

- [ ] Block polygonization with halo, `unary_union` at block edges, centroid ownership.
- [ ] Descriptors (`PaintElement` in `stage3_elements/elements.py`).
- [ ] Road-direction field from the structure tensor, with a coherence map (intersection cores).
- [ ] Fragment repair: collinear, similar width, gap ≤ wear tolerance **or** covered by no-data
      (→ `evidence = inferred`). Never merge across observed no-paint gaps longer than the wear
      tolerance.
- [ ] Rule typing, then a glyph CNN on road-aligned metric crops with open-set rejection.
      Discards are logged, never deleted silently.

## Step 6: Object assembly (§7)

**Acceptance:** object and attribute metrics (§10.2) on val.

- [ ] Candidate graph, then mutual-best linking, then curvature-limited centreline splines.
- [ ] Lateral profiles → single/double sub-lines with independent s-extents.
- [ ] Cross-block linking, **then** longitudinal profiles → solid/dashed/unknown intervals,
      transitions, inferred bridging over short no-data gaps with consistent phase.
- [ ] Crosswalks (bars parallel to traffic, regular spacing, `partial` flag), stop lines
      (perpendicular, wide, terminating tracks), symbols (lane assignment, heading check), area
      markings, conflict resolution, residual layer.

## Step 7: Vector output and QA (§8, §9)

**Acceptance:** precision of the auto-accepted set ≥ the per-class target (§10.3) on held-out data.

- [ ] GeoPackage writer (`stage5_vector/schema.py` is the schema source of truth), EPSG
      recorded, 1 mm precision, provenance on every object.
- [ ] Rule checks, confidence score (weights fitted on val, validated with a reliability
      diagram), and decision bands `c_accept` / `c_reject`.
- [ ] Review queue with reason codes, QGIS review project with forms. Reviewed corrections
      are exported as new training and gold data.

## Step 8: Full evaluation and iteration (§10)

- [ ] Full metric suite on the test areas. Report `unknown` as coverage, not as error.
- [ ] Error analysis by cause (`qa.ERROR_CAUSES`), and improve the responsible stage.
- [ ] Optional: learned grouping (GNN) on the same candidate graph. Adopt it only if it beats
      the rules at equal or better precision (§7.7).
- [ ] OpenDRIVE / Lanelet2 exporters from `track` + `track_interval` (§8.3).
