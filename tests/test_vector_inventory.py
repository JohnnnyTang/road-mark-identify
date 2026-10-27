import json

import geopandas as gpd
import pytest
import shapely

from roadmark.io.vector_inventory import VectorInventoryCfg, run_vector_inventory
from roadmark.synthetic import load_truth


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
@pytest.mark.filterwarnings("ignore:.crs. was not provided:UserWarning")
def test_vector_inventory_lines_and_boxes(stage1_run, synth_root):
    _, s1 = stage1_run
    truth = load_truth(synth_root / "synthetic" / "truth.json")
    ann = synth_root / "annotations"
    ann.mkdir(exist_ok=True)
    # Road lines as centrelines of the solid edge lines; markings as axis-aligned boxes.
    lines = gpd.GeoDataFrame(
        {"type": ["edge_solid", "edge_solid"], "road_name": ["Secret Rd", "Secret Rd"]},
        geometry=[shapely.LineString([(1, s * 5.6), (49, s * 5.6)]) for s in (-1, 1)])
    boxes = [shapely.envelope(g) for lab, g in truth if lab.startswith(("symbol", "crosswalk"))]
    assert boxes
    gpd.GeoDataFrame({"cls": ["mark"] * len(boxes)}, geometry=boxes).to_file(
        ann / "secret_place.gpkg", layer="boxes")
    lines.to_file(ann / "secret_place.gpkg", layer="lines")
    cfg = VectorInventoryCfg(paths=["annotations/**/*"], grid=str(s1.relative_to(synth_root)),
                             group_by=["type"])
    rep, names, files = run_vector_inventory(cfg, synth_root)
    blob = json.dumps(rep)
    assert "secret" not in blob.lower()  # file, layer and name-like values stay private
    assert names["#1"]["file"].endswith("secret_place.gpkg") and len(files) == 1
    by = {sum(1 for f in lyr["fields"] if f["field"] == "cls"): lyr for lyr in rep["layers"]}
    box_l, line_l = by[1], by[0]
    assert line_l["geometry_types"] == {"LineString": 2} and line_l["crs"] is None
    assert {"field": "type", "dtype": "str", "null": 0, "distinct": 1,
            "values": {"edge_solid": 2}} in line_l["fields"]
    assert line_l["lines"]["length_m"]["p50"] == 48.0
    prof = line_l["line_profile"]
    assert abs(prof["peak_offset_m"]) <= 0.03 and prof["peak_over_edge"] > 2
    assert 0.06 <= prof["width_at_half_peak_m"] <= 0.27  # 15 cm paint at 5 cm pixels
    pol = box_l["polygons"]
    assert pol["axis_aligned_box"] == pol["parts"] == len(boxes)
    assert box_l["polygon_contrast"]["share_inside_above_ring_p90"] > 0.2
    solid = line_l["line_profile_by"]["type"]["edge_solid"]
    assert solid["bright_share_on_line"] > 0.8 > 0.2 > solid["bright_share_45cm_beside"]
    assert len(line_l["geometry_sha"]) == 12
