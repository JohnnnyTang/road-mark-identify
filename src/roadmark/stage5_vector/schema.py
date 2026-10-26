"""GeoPackage layer schemas (§8.2). Single source of truth for writer, QA and eval."""

from __future__ import annotations

LAYERS: dict[str, dict] = {
    "track": {"geometry": "LineString", "fields": {
        "track_id": "str", "role": "str", "n_sublines_max": "int", "length_m": "float",
        "confidence": "float", "review_status": "str", "provenance": "str"}},
    "track_interval": {"geometry": None, "fields": {
        "track_id": "str", "subline": "int", "s_start": "float", "s_end": "float",
        "lateral_offset_m": "float", "pattern": "str", "dash_len_m": "float",
        "gap_len_m": "float", "width_m": "float", "evidence": "str", "colour": "str",
        "confidence": "float"}},
    "crosswalk": {"geometry": "Polygon", "fields": {
        "cw_id": "str", "bar_count": "int", "bar_width_m": "float", "bar_spacing_m": "float",
        "partial": "bool", "confidence": "float", "review_status": "str", "provenance": "str"}},
    "crosswalk_bar": {"geometry": "Polygon", "fields": {
        "cw_id": "str", "bar_index": "int", "length_m": "float", "width_m": "float"}},
    "stop_line": {"geometry": "LineString", "fields": {
        "sl_id": "str", "width_m": "float", "terminated_track_ids": "str",
        "confidence": "float", "review_status": "str", "provenance": "str"}},
    "symbol": {"geometry": "Point", "fields": {
        "sym_id": "str", "type": "str", "heading_deg": "float", "lane_track_left": "str",
        "lane_track_right": "str", "confidence": "float", "review_status": "str",
        "provenance": "str"}},
    "symbol_outline": {"geometry": "Polygon", "fields": {"sym_id": "str"}},
    "area_marking": {"geometry": "Polygon", "fields": {
        "area_id": "str", "type": "str", "boundary_track_ids": "str", "confidence": "float",
        "review_status": "str", "provenance": "str"}},
    "paint_element": {"geometry": "Polygon", "fields": {
        "element_id": "str", "etype": "str", "object_id": "str", "evidence": "str",
        "prob_mean": "float", "nodata_buffer_fraction": "float"}},
    "residual_paint": {"geometry": "Polygon", "fields": {
        "element_id": "str", "etype": "str", "reason": "str"}},
    "review_queue": {"geometry": "Geometry", "fields": {
        "item_id": "str", "layer": "str", "object_id": "str", "reason_codes": "str"}},
}

REVIEW_STATUS = ("auto_accepted", "needs_review", "reviewed_ok", "reviewed_fixed")
COORD_PRECISION_M = 0.001  # §8.1
MAX_SIMPLIFY_TOL_M = 0.01
