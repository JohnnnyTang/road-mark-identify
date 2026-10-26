"""Typed configuration: the marking standard (§11.2) and the pipeline parameters.

Both are YAML files under ``configs/``. Site-specific overrides (real paths, EPSG of the
project area) live in ``configs/local/`` which is git-ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Standards configuration (handoff §11.2)
# ---------------------------------------------------------------------------


class Tol(_Strict):
    nominal: float
    tol: float


class Span(_Strict):
    min: float
    max: float


class DashPattern(_Strict):
    name: str
    dash: float
    gap: float
    tol: float


class LineStd(_Strict):
    width: Tol
    widths_allowed: list[float] = Field(default_factory=list)
    wear_gap_max: float
    double_spacing: Span


class CrosswalkStd(_Strict):
    bar_width: Tol
    bar_gap: Tol
    bar_length_min: float
    min_bars: int


class StopLineStd(_Strict):
    width: Span


class SymbolStd(_Strict):
    classes: list[str]


class GeometryStd(_Strict):
    max_curvature_road: float  # 1/m
    max_curvature_guide: float  # 1/m


class Thresholds(_Strict):
    """Set on validation data at the precision target, never by hand (§5.4, §9.3)."""

    paint_t_high: float | None = None
    paint_t_low: float | None = None
    c_accept: float | None = None
    c_reject: float | None = None


class StandardConfig(_Strict):
    standard: str
    edition: str
    verified: bool = False  # flip to true only after checking every value against the text
    units: Literal["metres"] = "metres"
    line: LineStd
    dashed_patterns: list[DashPattern]
    crosswalk: CrosswalkStd
    stop_line: StopLineStd
    symbols: SymbolStd
    geometry: GeometryStd
    thresholds: Thresholds = Field(default_factory=Thresholds)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


class CsfParams(_Strict):
    cloth_resolution: float = 0.5
    rigidness: int = 3
    class_threshold: float = 0.15
    time_step: float = 0.65
    iterations: int = 500
    slope_smooth: bool = False


class GridMinParams(_Strict):
    cell: float = 1.0
    window_cells: int = 5
    max_height: float = 0.15
    slope_tol: float = 0.05  # m per m, added over half the window


class GroundCfg(_Strict):
    method: Literal["existing", "csf", "grid_min"] = "existing"
    ground_classes: list[int] = Field(default_factory=lambda: [2])
    csf: CsfParams = Field(default_factory=CsfParams)
    grid_min: GridMinParams = Field(default_factory=GridMinParams)
    dem_cell: float = 0.25  # ground surface resolution used for height-above-ground
    low_nonground_max_hag: float = 0.30  # §4.1: keep low non-ground for curbs


class IntensityCfg(_Strict):
    enabled: bool = True
    r_ref: float = 10.0
    # Asphalt = ground points whose model residual is within trim_mad_k * MAD of the median
    # (iterated). Paint, sidewalks and dropouts fall outside.
    trim_mad_k: float = 2.5
    trim_iterations: int = 8
    # a and b are only separable if range and incidence vary independently (§4.2 note).
    collinear_corr: float = 0.95
    b_prior: float = 1.0  # used when collinear (Lambertian)
    fit_sample_size: int = 2_000_000
    min_cos_incidence: float = 0.05
    per_pass_match: bool = True
    match_quantiles: tuple[float, float] = (0.10, 0.90)
    consistency_tol: float = 0.10  # relative; §4.2 step 3, recorded in the QA report


class TrajectoryCfg(_Strict):
    """Scanner positions over time; needed for range/incidence when the LAS has no range."""

    path: str | None = None  # npz/csv with columns t, x, y, z (relative to data_root)
    time_offset: float = 0.0  # added to trajectory time to match LAS gps_time


class RasterCfg(_Strict):
    pixel_size: float = 0.025
    coverage_min_points: int = 2
    block_size_px: int = 4000  # 100 m at 2.5 cm
    halo_m: float = 10.0  # context for ground filtering / DEM at block edges
    p_quantile: float = 0.90
    compress: str = "DEFLATE"


class RoadMaskCfg(_Strict):
    curb_step_min: float = 0.08
    curb_step_max: float = 0.25
    step_window_m: float = 0.30
    min_road_area_m2: float = 50.0


class Stage1Cfg(_Strict):
    inputs: list[str] = Field(default_factory=list)  # LAS/LAZ paths or globs under data_root
    pass_id_field: str = "point_source_id"
    chunk_points: int = 5_000_000
    ground: GroundCfg = Field(default_factory=GroundCfg)
    trajectory: TrajectoryCfg = Field(default_factory=TrajectoryCfg)
    intensity: IntensityCfg = Field(default_factory=IntensityCfg)
    raster: RasterCfg = Field(default_factory=RasterCfg)
    road_mask: RoadMaskCfg = Field(default_factory=RoadMaskCfg)


class PipelineConfig(_Strict):
    name: str = "default"
    data_root: str = "${ROADMARK_DATA_ROOT}"
    crs_epsg: int | None = None  # None only allowed with allow_local_crs (synthetic data)
    allow_local_crs: bool = False
    seed: int = 0
    standard: str = "configs/standards/gb5768_3_2009.yaml"
    stage1: Stage1Cfg = Field(default_factory=Stage1Cfg)

    @model_validator(mode="after")
    def _crs_required(self) -> PipelineConfig:
        if self.crs_epsg is None and not self.allow_local_crs:
            raise ValueError(
                "crs_epsg is required (projected metric CRS, §2.5). Put it in a "
                "configs/local/*.yaml override, or set allow_local_crs for synthetic data."
            )
        return self

    def resolved_data_root(self) -> Path:
        root = os.path.expandvars(self.data_root)
        if "$" in root:
            raise ValueError(f"data_root '{self.data_root}' not resolved; set it in .env")
        return Path(root)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        both_dicts = isinstance(v, dict) and isinstance(out.get(k), dict)
        out[k] = _deep_merge(out[k], v) if both_dicts else v
    return out


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader (no dependency); existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_yaml_layers(paths: list[Path]) -> dict:
    data: dict = {}
    for p in paths:
        data = _deep_merge(data, yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
    return data


def load_pipeline(*paths: str | Path) -> PipelineConfig:
    """Load a pipeline config; later files override earlier ones (base, then local)."""
    load_dotenv()
    return PipelineConfig.model_validate(load_yaml_layers([Path(p) for p in paths]))


def load_standard(path: str | Path) -> StandardConfig:
    return StandardConfig.model_validate(load_yaml_layers([Path(path)]))


def config_hash(cfg: BaseModel) -> str:
    """Stable short hash of a config, recorded in every output's provenance (§2.7)."""
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
