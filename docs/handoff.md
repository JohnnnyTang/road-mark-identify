# Road Marking Extraction from Point-Cloud Rasters: Technical Handoff

## 1. Purpose and scope

### 1.1 Goal

Extract road surface markings from georeferenced rasters derived from mobile/airborne laser scanning point clouds, and deliver them as complete, typed vector objects. Each object runs from its true start to its true end, even where the paint itself is discontinuous (dashed lines, crosswalks, hatched areas).

### 1.2 Governing priority

Precision is the top priority throughout. Concretely:

- When evidence is insufficient, the pipeline outputs `unknown` or flags the object for review. It never fills in with a plausible guess.
- Missing data (occlusion, scan gaps) is never treated as absence of paint.
- Every output object carries a confidence score, its evidence status (observed vs. inferred), and provenance.
- Thresholds are chosen on validation data at a high-precision operating point. Recall is improved afterwards through review and retraining, never by loosening acceptance criteria.

### 1.3 Target classes

| Group | Class | Output geometry |
|---|---|---|
| Longitudinal | Lane / edge / centre lines: solid, dashed, double (any side combination), mixed along length | LineString track + linear-referenced attribute intervals |
| Longitudinal | Guide lines through the intersection core (usually curved, dashed) | LineString track |
| Transverse | Stop lines | LineString (centreline) + width |
| Transverse | Yield / give-way lines (if present in the jurisdiction) | LineString |
| Area | Crosswalks (zebra and other bar patterns) | Polygon + member bars |
| Area | Channelization / hatched areas, chevrons | Polygon + outline tracks |
| Area | Waiting zones (e.g. left-turn waiting areas) | Polygon + boundary tracks |
| Symbol | Turn arrows (all types in the applicable standard) | Point + heading + type, plus outline Polygon |
| Symbol | Text and other symbols (diamonds, bus lane text, etc.) | Point + type, plus outline Polygon |

Freeze the class list, with definitions and example crops, before annotation begins (§5.5).

---

## 2. Core design principles

1. **Bottom-up assembly.** Paint pixels → paint elements (one physical painted piece each) → semantic objects. The semantic type is decided at the object level, where the geometric context exists. It is never decided by partitioning the image in advance.
2. **Tracks with linear referencing.** Every longitudinal marking is represented as a track: a centreline plus attributes indexed by distance *s* along it. Line-type combinations are not enumerated as classes. They are described by two independent attributes:
   - the **lateral structure**: how many sub-lines are side by side (single / double), and their offsets;
   - the **longitudinal pattern** of each sub-line: intervals of solid / dashed / unknown along *s*, with measured dash and gap lengths.

   Any combination (solid–dashed doubles, solid turning into dashed, one side of a double ending early) is then a value of these attributes, not a new class. This matches the OpenDRIVE `roadMark` model, so the result can later be exported to HD-map formats.
3. **Three-state evidence.** Every location is *paint*, *no paint*, or *no data*. No-data comes from point density / coverage channels and is carried through every stage.
4. **Standards as priors.** Markings are engineered objects built to a national standard (e.g. GB 5768.3 in China, MUTCD in the US). Dimensional parameters from the applicable standard drive the assembly rules and the plausibility checks. They live in one versioned configuration file (§11.2).
5. **Metric space.** All processing after rasterization is done in a projected metric CRS, so dimensional rules are expressed in metres.
6. **Stitch before analysis.** Tiling exists only inside network inference. All object-level steps run on stitched, seamless rasters.
7. **Determinism and reproducibility.** Fixed model versions, config hashes and seeds. Every output records the versions that produced it.

---

## 3. Inputs and items to confirm

| Item | Status / required decision |
|---|---|
| Source point cloud available? | **Confirm.** Strongly preferred over pre-flattened rasters (§4). |
| Point attributes | Intensity required; scan-pass ID / GPS time, scanner position or range, and RGB are valuable if available. |
| Existing raster pixel size | **Confirm.** Target 2–3 cm per pixel (a ~15 cm line ≈ 5–7 px). |
| CRS | **Confirm** a projected metric CRS. Record EPSG for all outputs. |
| Applicable marking standard | **Confirm** jurisdiction and standard edition. Fill §11.2. |
| Paint colour requirement | Intensity cannot distinguish yellow from white. If colour is required, RGB (colourized point cloud or co-registered orthophoto) is mandatory. |
| Number of scan passes per area | Needed for intensity normalization (§4.2). |

---

## 4. Stage 1: Raster generation from the point cloud

If only the pre-flattened rasters exist, use them, but note the limitations in §4.6.

### 4.1 Ground filtering

- Classify ground points (e.g. PDAL `filters.smrf`, `filters.csf`, or the existing classification if trustworthy).
- Keep only ground points for marking rasters. This removes most building, vegetation and vehicle edges, which are the dominant false-positive source in the current images.
- Separately keep low non-ground points within ~0.3 m of ground for curb detection (§4.4).

### 4.2 Intensity correction

Raw intensity varies with range, incidence angle, scan pass and sensor. Uncorrected, the same paint appears bright on one side of a road and dim on the other, which corrupts any fixed threshold.

1. Fit a range/incidence model on unpainted asphalt samples, e.g. `I_corr = I · (R / R_ref)^a / cos(θ)^b`, with *a* and *b* estimated empirically rather than assumed.
2. Normalize per scan pass: match the asphalt intensity distribution of each pass to a reference (quantile matching on asphalt pixels).
3. Validate: corrected intensity of the same marking observed from different passes / ranges must agree within a set tolerance. Record that tolerance in the QA report.

### 4.3 Raster channels

All channels share one grid, one CRS, one pixel size.

| Channel | Definition | Purpose |
|---|---|---|
| `int_max` | Max corrected intensity per cell | Primary paint signal (thin lines) |
| `int_mean` | Mean corrected intensity per cell | Robust paint signal, noise suppression |
| `int_p90` | 90th percentile intensity | Robust alternative to max |
| `density` | Ground point count per cell | No-data detection (critical) |
| `coverage` | Binary: cell has ≥ N ground points | Explicit valid-data mask |
| `hag_max` | Max height above local ground (from the low non-ground set) | Curb and road-edge detection |
| `rgb` (optional) | Mean colour per cell | Paint colour |

Use `nodata` consistently. A cell without points is no-data, not zero intensity.

### 4.4 Road-surface mask

- Detect curbs as height steps of ~0.08–0.25 m (tune per area) in `hag_max` / the ground surface.
- Build the road-surface polygon from the curb lines plus the continuous ground surface.
- The mask is a prior, not a hard filter: paint outside it is flagged, not deleted, because islands and medians can carry markings.

### 4.5 Output format

- Cloud-Optimized GeoTIFF per channel (or one multi-band COG), with internal tiling and overviews.
- Processing blocks follow §6.1.

### 4.6 If only pre-flattened rasters are available

- No reliable density channel. Occlusion gaps cannot be separated from paint gaps. All dash gaps must then be treated as lower-confidence evidence, and the review rate will be higher.
- No control over intensity normalization. Measure the brightness consistency of known markings across the area and document it.
- Recommendation: obtain the point cloud before production runs.

---

## 5. Stage 2: Paint segmentation

### 5.1 Task definition

- **Primary output:** per-pixel paint probability (binary: paint vs. not paint).
- **Auxiliary output:** coarse per-pixel class: `longitudinal_line`, `transverse_bar`, `symbol`, `hatch`, `other_paint`. This is a hint for the later stages only; final types are decided at the object level.
- **Ignore label:** no-data pixels are excluded from the loss and the metrics.

### 5.2 Model

- High-resolution architecture with full-resolution or ½-resolution output, e.g. HRNet, or a U-Net / SegFormer-style encoder with a high-resolution decoder. Thin 5–7 px structures must not be lost to output stride.
- Input: normalized `int_max`, `int_mean`, `density` (log-scaled), optional `hag_max` and RGB.
- Loss: BCE + Dice for paint, plus a thin-structure term (clDice) or a boundary loss. Class-balanced cross-entropy for the auxiliary head.
- Normalize input using global statistics from the whole dataset (or per scan pass), never per tile.

### 5.3 Inference tiling and stitching

1. Tile size 1024 px. Set the overlap empirically (step 4); start at ≥128 px.
2. Blend each tile's probabilities (not binary masks) with a weight window that falls off toward tile edges (cosine or Gaussian). Accumulate `Σ(p·w)` and `Σw` in two float rasters (windowed writes or memmap); final probability = `Σ(p·w) / Σw`.
3. Test-time augmentation: average over flips and 90° rotations. This lowers variance and improves precision on thin lines.
4. Determine the overlap: predict the same area with the tile grid shifted by half a tile. Measure the distance from the tile edge within which the two predictions diverge. Set the overlap to at least twice that distance.
5. Seam acceptance test: segmentation metrics on pixels within 64 px of a tile edge must not differ meaningfully from tile-centre pixels.

A paint element split across tiles is rejoined automatically by the stitched raster. No later stage sees tile boundaries.

### 5.4 Probability calibration and thresholding

- Calibrate probabilities on a validation set (temperature scaling) so they are meaningful across areas.
- Use hysteresis thresholding: pixels above `t_high` seed components; connected pixels above `t_low` extend them. This keeps thin lines continuous without admitting isolated noise.
- Choose `t_high` / `t_low` on validation data at the precision target (§10.3), and record them in the config.

### 5.5 Annotation protocol

- **Bootstrap:** generate initial masks with classical processing (normalized intensity, adaptive threshold, morphology, size filter), then correct them manually in QGIS. Interactive segmentation tools (SAM-style) may speed this up.
- **Guidelines document,** with example crops, defining:
  - what counts as paint (worn paint included or not, and how faint);
  - paint boundary placement (outer edge of the paint);
  - occluded / no-data regions → ignore label, never "background";
  - each class in §1.3, with edge cases.
- **Agreement:** double-annotate ~10% of tiles, compute inter-annotator IoU and boundary agreement, and resolve disagreements into the guidelines.
- **Vector gold standard:** on the held-out evaluation areas, also annotate complete objects (tracks with *s*-intervals, crosswalk polygons, stop lines, symbols with type and heading). This is the reference for §10.
- **Pretraining (optional):** public datasets with lane-marking classes (e.g. SkyScapes for aerial imagery; Toronto-3D includes a road-marking class for point clouds). Expect a domain gap; fine-tuning on own data is mandatory.

### 5.6 Data splits

- Split by geographic area (whole intersections / road sections), never by tile. Neighbouring tiles leak.
- The evaluation set must cover: different scan campaigns, worn paint, heavy occlusion, curved geometry, complex intersections, and every target class.

---

## 6. Stage 3: Paint element extraction

### 6.1 Analysis blocks

If the stitched raster is too large to process in one piece:

- Process in blocks (e.g. 200–500 m) with a halo of at least the largest bounded object (≈50 m covers crosswalks and stop lines).
- Polygonize the thresholded mask per block. Merge polygons touching block boundaries with `unary_union`. Because the mask is seamless, the shared edges match exactly and no tolerance is needed.
- Objects are owned by the block containing their centroid (prevents duplicates). Tracks are handled separately (§7.1.5).

### 6.2 Element descriptors

For each paint element (connected component / polygon):

| Descriptor | Notes |
|---|---|
| Oriented min-area rectangle: length, width, orientation | Primary shape cues |
| Area, solidity, rectangularity | Separates bars/dashes from glyphs |
| Skeleton length and branch count | Separates arrows / text from strokes |
| Orientation relative to local road direction (§6.3) | Longitudinal vs. transverse |
| Mean / max paint probability | Confidence |
| No-data fraction in a buffer around the element | Occlusion flag |
| Inside / outside road-surface mask | Plausibility |

### 6.3 Local road-direction field

- Compute a dense orientation field from the smoothed structure tensor of the paint and curb evidence (large-scale smoothing, tens of metres). Optionally constrain it with available road centrelines.
- The field supplies "along-road" and "across-road" directions for every subsequent rule.
- At intersection cores the field is ambiguous. Mark these areas (low coherence of the tensor) so that rules there use the element's own geometry instead.

### 6.4 Fragment repair

Worn or partially occluded lines break into several elements. Merge two elements only if all of these hold:

- collinear (angle difference and lateral offset within tight limits);
- similar width (within the standard's tolerance);
- the gap is either below a small wear tolerance, or covered by no-data. In the second case, mark the merged element `evidence = inferred` over the gap.

Never merge across a gap that is observed no-paint and longer than the wear tolerance. That gap may be a genuine dash gap.

### 6.5 Element typing

- Rule-based coarse typing from descriptors: `stroke` (long, thin), `bar` (rectangular, standard bar dimensions), `glyph` (complex shape), `hatch_stripe`, `noise`, `unknown`.
- Glyph classifier for arrows / text / symbols:
  - input: crop rotated so the local road direction points up, scaled to fixed metric size;
  - small CNN trained on own annotated glyphs, with augmentation (intensity variation, wear simulation, partial occlusion);
  - open-set rejection: if the top-class probability is below threshold, or the shape doesn't match any class template within tolerance, output `unknown` and flag for review;
  - output: class and heading (the arrow tip direction is resolved explicitly, not inferred from the road direction alone).
- Noise rejection: discard elements below a minimum area or with implausible dimensions. Log the discards for audit; never delete silently.

---

## 7. Stage 4: Object assembly

Each paint element is assigned to at most one object. Elements left unassigned are kept and reported (§9).

### 7.1 Longitudinal tracks

#### 7.1.1 Candidate graph

- Nodes: `stroke` elements and dash-like `bar` elements whose long axis is roughly parallel to the local road direction.
- Edges between node pairs where:
  - the endpoint gap along the direction is ≤ max dash gap from the standard (+ tolerance), or the gap is covered by no-data;
  - the lateral offset between the extended axes is ≤ a small tolerance (≈ half a line width);
  - the angle difference is ≤ a threshold consistent with the maximum design curvature at that gap length;
  - widths are similar.
- Edge score combines these terms, normalized by their tolerances.

#### 7.1.2 Chain linking

- Link elements into chains with mutual-best matching: an edge is accepted only if each endpoint is the other's best candidate in that direction. This is conservative and avoids zig-zag linking between adjacent parallel lines.
- Resolve remaining conflicts with a global min-cost path cover if needed.
- Reject chains that violate curvature continuity.

#### 7.1.3 Centreline fitting

- Fit a smoothing spline through element centrelines (solid parts) and dash centroids / axes (dashed parts). Weight each point by its paint probability.
- Constrain curvature by the design limits. Report residuals; large residuals lower confidence.
- In intersection cores (guide lines), allow higher curvature, but require more supporting elements.

#### 7.1.4 Lateral profile: single vs. double lines

- Every ~0.5 m along the track, sample a perpendicular profile of paint probability (±0.5 m).
- Detect peaks. Two peaks with spacing within the standard's double-line range → two sub-lines, each with its own lateral offset.
- Sub-lines carry independent *s*-extents: one side of a double line may start or end on its own.
- Parallel lines farther apart than the double-line range are separate tracks, not a double line.

#### 7.1.5 Cross-block linking

- Build chains within each block, then link chain endpoints that meet at block boundaries using the same edge rules (§7.1.1).
- Run longitudinal profiling (§7.1.6) only after cross-block linking, so dash periods are not cut at block edges.

#### 7.1.6 Longitudinal profile: solid / dashed / unknown intervals

For each sub-line:

1. Sample along *s* at pixel spacing, in a band of one line width centred on the sub-line. Per sample, record paint state (max probability in the band ≥ threshold), no-data state (coverage mask), and probability.
2. Run-length encode into alternating paint / no-paint runs, with no-data runs kept separate.
3. Classify intervals:
   - **solid:** continuous paint, interrupted only by gaps ≤ the wear tolerance;
   - **dashed:** alternating paint/no-paint runs whose lengths match a standard dash/gap pattern within tolerance, with regular period (coefficient of variation below threshold) over at least N periods;
   - **unknown:** anything else, including runs dominated by no-data.
4. Locate transitions (solid ↔ dashed) at the boundary between the last run of one pattern and the first run of the next. Report the position to pixel precision. Change-point detection (e.g. PELT on the local paint fraction) is a cross-check, not the primary locator.
5. Across occlusions: a dashed or solid interval may continue across a no-data gap only if the pattern (and, for dashes, the phase) is consistent on both sides and the gap is shorter than a set limit. The bridged sub-interval is labelled `evidence = inferred`.
6. Record measured statistics per interval: mean dash length, gap length, line width, and their deviation from the standard.

### 7.2 Crosswalks

Zebra bars run parallel to the traffic direction, arranged in a row across the road. Grouping criteria:

- bars with similar orientation (within a few degrees) and similar length (within ~10%);
- bar centroids collinear along an axis perpendicular to the bars;
- spacing regular (low coefficient of variation) and matching the standard's bar/gap dimensions;
- a minimum bar count (config).

Output: polygon (minimum rotated rectangle of the bar union), member bar polygons, bar count, measured width/spacing. If bars are missing under no-data, flag `partial = true`; do not invent bars.

### 7.3 Stop lines

A `bar` element qualifies when:

- it is perpendicular to the road direction (within a tolerance). This is the key difference from crosswalk bars, which are parallel to it;
- its width is within the standard's stop-line range, which is wider than lane lines;
- its length covers at least one lane width;
- the approach-side ends of longitudinal tracks terminate at or near it; it is typically upstream of a crosswalk where one exists.

Output: centreline LineString, width, list of the lane tracks it terminates.

### 7.4 Symbols

- Output: point (glyph centroid), heading, type, outline polygon, classifier confidence.
- Assign each symbol to a lane: the pair of tracks enclosing it at that *s*.
- Plausibility: arrows must point roughly along the lane direction.

### 7.5 Area markings

- **Hatched / channelization areas:** a group of parallel diagonal stripes bounded by outline tracks. Output: polygon + outline tracks + stripe count and orientation.
- **Waiting zones:** closed or near-closed boundaries made of tracks (often dashed) inside the intersection. Output: polygon + boundary tracks.
- **Guide lines:** handled as tracks (§7.1) with relaxed curvature, labelled `role = guide`.

### 7.6 Conflict resolution and residuals

- If an element qualifies for several objects (e.g. a bar that could be a crosswalk bar or a stop line), keep the assignment with the higher normalized fit score. Flag the case if the scores are close.
- Unassigned elements are written to a `residual_paint` layer with descriptors, for review and for improving the rules or the model.

### 7.7 Learned grouping (optional extension)

Once enough vector gold standard exists, a graph neural network can classify edges ("same object") and nodes (element role) on the same candidate graph. Adopt it only if it beats the rule-based assembly on the held-out evaluation at equal or better precision. Keep the rules as the plausibility check on its outputs.

---

## 8. Stage 5: Vector output

### 8.1 Format

A GeoPackage, one per processing run, with the CRS recorded (EPSG). Coordinates are stored at ≥ 1 mm precision. Geometry simplification tolerance is ≤ 1 cm, so simplification never degrades accuracy.

### 8.2 Layers and fields

**`track`** (LineString, centreline of the whole track)

| Field | Type | Description |
|---|---|---|
| `track_id` | text | Stable ID |
| `role` | text | `lane_line`, `edge_line`, `centre_line`, `guide`, `area_boundary`, `unknown` |
| `n_sublines_max` | int | 1 or 2 |
| `length_m` | real | |
| `confidence` | real | §9.2 |
| `review_status` | text | `auto_accepted`, `needs_review`, `reviewed_ok`, `reviewed_fixed` |
| `provenance` | text | Model version, config hash, source rasters |

**`track_interval`** (no geometry; linear-referenced table)

| Field | Type | Description |
|---|---|---|
| `track_id` | text | FK to `track` |
| `subline` | int | 0 = single; 1 = left, 2 = right (relative to digitizing direction) |
| `s_start`, `s_end` | real | Metres along the track centreline |
| `lateral_offset_m` | real | Sub-line offset from the centreline |
| `pattern` | text | `solid`, `dashed`, `unknown` |
| `dash_len_m`, `gap_len_m` | real | Measured means (dashed only) |
| `width_m` | real | Measured paint width |
| `evidence` | text | `observed`, `inferred` |
| `colour` | text | `white`, `yellow`, `unknown` (only if RGB is available) |
| `confidence` | real | |

**`crosswalk`** (Polygon): `cw_id`, `bar_count`, `bar_width_m`, `bar_spacing_m`, `partial`, `confidence`, `review_status`, `provenance`.

**`crosswalk_bar`** (Polygon): `cw_id`, `bar_index`, measured dimensions.

**`stop_line`** (LineString): `sl_id`, `width_m`, `terminated_track_ids`, `confidence`, `review_status`, `provenance`.

**`symbol`** (Point): `sym_id`, `type`, `heading_deg`, `lane_track_left`, `lane_track_right`, `confidence`, `review_status`, `provenance`. Plus **`symbol_outline`** (Polygon).

**`area_marking`** (Polygon): `area_id`, `type` (`hatch`, `chevron`, `waiting_zone`, …), `boundary_track_ids`, `confidence`, `review_status`, `provenance`.

**`paint_element`** (Polygon): every extracted element with descriptors and the object it belongs to (or none). Kept for audit and retraining.

**`residual_paint`** (Polygon): unassigned elements.

**`review_queue`** (Point/Polygon): items flagged for review, with reason codes (§9.3).

### 8.3 Optional exports

Converters to OpenDRIVE `roadMark` records or to Lanelet2 line-string types can be written later from `track` + `track_interval` without loss.

---

## 9. Quality control

### 9.1 Automated rule checks

| Check | Rule |
|---|---|
| Dimensional | Line width, dash/gap length, bar dimensions within standard ± tolerance |
| Periodicity | Dash period regular within a dashed interval |
| Geometry | Curvature within design limits; no self-intersections |
| Topology | Tracks don't cross each other except at defined locations (guide lines in intersection cores); lane tracks end at stop lines or continue |
| Consistency | Stop line perpendicular to its terminated tracks; crosswalk bars parallel to the road direction |
| Symbols | Inside a lane; heading along the lane |
| Coverage | Fraction of each object's extent with no-data |
| Duplicates | No two objects claim the same paint element |

### 9.2 Confidence score

Per object, combine (weights fitted on validation data):

- mean calibrated paint probability over its elements;
- normalized deviation of its measured dimensions from the standard;
- fit residual (centreline fit, periodicity, crosswalk regularity);
- observed fraction (1 − no-data fraction over its extent);
- for symbols, classifier probability.

Validate the score: on the evaluation set, binned confidence must correlate with actual correctness (reliability diagram).

### 9.3 Decision bands and human review

- **Auto-accept:** confidence ≥ `c_accept` and all rule checks pass.
- **Needs review:** below `c_accept`, or any rule check fails, or an assignment conflict was close (§7.6), or an inferred interval exceeds a set length.
- **Reject:** below `c_reject`; written to `residual_paint`.

Set `c_accept` so that the auto-accepted set meets the precision target (§10.3) on held-out data.

Review in QGIS, with a project file containing:

- the intensity and density rasters, the paint probability raster, and all vector layers with styles;
- the `review_queue` layer with reason codes;
- forms to record decisions (`reviewed_ok` / `reviewed_fixed` + edit).

Reviewed corrections are exported as new training and gold data (feedback loop).

---

## 10. Evaluation protocol

### 10.1 Reference data

Vector gold standard on held-out areas (§5.5–5.6), annotated to the same guidelines and schema as the output.

### 10.2 Metrics

| Level | Metric |
|---|---|
| Pixel | Precision, recall, IoU for paint; boundary F-score at 1 and 2 px tolerance; seam test (§5.3) |
| Element | Detection precision/recall at IoU ≥ 0.5; glyph classification accuracy and confusion matrix |
| Track geometry | Buffer-based correctness, completeness and quality at 5 cm and 10 cm tolerance; RMS and 95th-percentile lateral offset to the reference |
| Track attributes | Fraction of reference length with the correct pattern; transition position error (m); dash/gap length error; single/double accuracy |
| Crosswalk / stop line / area | Object precision/recall at IoU ≥ 0.5; bar count accuracy; stop line endpoint error |
| Symbol | Detection precision/recall; type accuracy; heading error |
| Confidence | Reliability diagram; precision of the auto-accepted set |
| Throughput | Share of objects auto-accepted vs. sent to review |

Report `unknown` outputs separately as a coverage figure, not as errors. Wrong labels are errors; `unknown` is abstention.

### 10.3 Acceptance targets

Define per class, before development, with the project owner:

- minimum precision for auto-accepted objects (primary target);
- maximum geometric error (RMS and 95th percentile);
- minimum coverage (share not `unknown` or rejected), as a secondary target.

### 10.4 Error analysis

Tag each error with a cause (`segmentation_miss`, `segmentation_false_positive`, `fragment_merge`, `linking`, `profile`, `classification`, `occlusion`, `reference_error`) to direct improvement work to the responsible stage.

---

## 11. Implementation notes

### 11.1 Suggested tooling

- Point cloud: PDAL (ground filtering, rasterization via `writers.gdal`), laspy
- Rasters: GDAL, rasterio (windowed I/O, COG)
- Segmentation: PyTorch; a segmentation model library or custom HRNet / U-Net
- Geometry: scikit-image (labelling, skeletons), shapely, GeoPandas, pyogrio
- Graph linking: networkx or a custom min-cost matcher
- Profiles: NumPy/SciPy; `ruptures` for change-point cross-checks
- Review: QGIS project with styles and forms

### 11.2 Standards configuration (fill from the applicable standard)

```yaml
standard: "<name and edition>"
units: metres
line:
  width:            {nominal: <>, tol: <>}
  wear_gap_max:     <>
  double_spacing:   {min: <>, max: <>}
dashed_patterns:          # one entry per pattern in the standard
  - {name: "<>", dash: <>, gap: <>, tol: <>}
crosswalk:
  bar_width:        {nominal: <>, tol: <>}
  bar_gap:          {nominal: <>, tol: <>}
  bar_length_min:   <>
  min_bars:         <>
stop_line:
  width:            {min: <>, max: <>}
symbols:
  classes: ["<arrow types and other symbols in the standard>"]
geometry:
  max_curvature_road:  <>
  max_curvature_guide: <>
thresholds:              # set on validation data, not by hand
  paint_t_high: <>
  paint_t_low: <>
  c_accept: <>
  c_reject: <>
```

### 11.3 Repository layout

```
configs/            standards and pipeline configs (versioned)
stage1_raster/      ground filtering, intensity correction, channel rasters
stage2_segment/     training, tiled inference, stitching, calibration
stage3_elements/    polygonize, descriptors, direction field, glyph classifier
stage4_assembly/    tracks, profiles, crosswalks, stop lines, symbols, areas
stage5_vector/      GeoPackage writer, exports
qa/                 rule checks, confidence, review queue, QGIS project
eval/               metrics, gold standard tools, reports
docs/               annotation guidelines, this handoff
```

---

## 12. Development sequence

Each step finishes with its acceptance check before the next begins.

1. **Data foundation.** Confirm the §3 items. Implement ground filtering, intensity correction and the channel rasters. Check: cross-pass intensity consistency within tolerance; density channel separates occlusions.
2. **Guidelines and gold standard.** Write the annotation guidelines; annotate the vector gold standard on held-out areas; measure inter-annotator agreement.
3. **Classical reference.** Normalized intensity + hysteresis threshold + morphology, run through stages 3–5. This serves as a performance floor and produces pre-labels.
4. **Segmentation model.** Annotate training areas, train, calibrate; implement tiled inference with blending and TTA. Check: pixel metrics and seam test.
5. **Elements and glyphs.** Polygonization, block merging, descriptors, direction field, fragment repair, glyph classifier with rejection. Check: element-level metrics.
6. **Assembly.** Track linking, lateral and longitudinal profiles, crosswalks, stop lines, symbols, area markings. Check: object and attribute metrics.
7. **Output and QA.** GeoPackage schema, rule checks, confidence calibration, review queue, QGIS review project. Check: auto-accepted precision meets the §10.3 target.
8. **Full evaluation and iteration.** Run the full evaluation; error analysis by cause; improve the responsible stage; optionally evaluate learned grouping (§7.7).

---

## 13. Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| Occlusion gaps mistaken for dash gaps | Wrong solid/dashed attribution | Density / coverage channel; three-state evidence; `inferred` labelling; review of long inferred spans |
| Intensity inconsistency between passes/ranges | Missed faint paint, false positives | Range/incidence correction; per-pass normalization; global (not per-tile) input normalization |
| Worn paint | Fragmented lines, missed dashes | Hysteresis thresholds; strict fragment repair; wear examples in training data |
| Tile seams | Broken thin lines | Probability blending, TTA, empirically sized overlap, seam test |
| Adjacent parallel lines mis-linked | Zig-zag tracks | Mutual-best linking; tight lateral tolerance; lateral profile check |
| Intersection-core ambiguity | Wrong linking of guide lines | Low-coherence zone handling; stricter support requirements; review flag |
| Rare or non-standard markings | Misclassification | Open-set rejection → `unknown`; residual layer; guidelines updated |
| Paint colour unknown | Incomplete attributes | RGB channel if required; otherwise `colour = unknown` |
| Annotation inconsistency | Unreliable training and evaluation | Guidelines with examples; double annotation; agreement tracking |
| Standard deviations in the field (non-compliant paint) | Rule rejections of real markings | Tolerances fitted on observed data; rejected items go to review, not deletion |

---

## 14. Open decisions

1. Point cloud availability and attributes (§3).
2. Pixel size and CRS for production rasters.
3. Applicable marking standard and edition; fill §11.2.
4. Whether paint colour is required (determines whether RGB is needed).
5. Per-class acceptance targets (§10.3).
6. Gold-standard and training areas: selection and size.
7. Final class list, including which symbol and text types are in scope.
