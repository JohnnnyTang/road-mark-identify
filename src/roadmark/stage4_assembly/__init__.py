"""Stage 4: object assembly (§7). Roadmap Step 6.

  graph       candidate graph over stroke/dash elements, edge scores (§7.1.1)
  linking     mutual-best chain linking, min-cost cover, cross-block linking (§7.1.2, 7.1.5)
  centreline  probability-weighted smoothing spline with curvature limits (§7.1.3)
  lateral     perpendicular profiles -> single/double sub-lines with own s-extents (§7.1.4)
  longitudinal  run-length profile -> solid/dashed/unknown intervals, transitions,
                inferred bridging over no-data (§7.1.6)
  crosswalk, stop_line, symbols, areas, conflicts (§7.2-7.6)
"""
