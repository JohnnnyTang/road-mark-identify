"""Stage 2: paint segmentation (§5). Roadmap Steps 3 (classical) and 4 (learned).

Modules:
  classical  normalised intensity + hysteresis threshold + morphology (performance floor,
             pre-labels for annotation)
  tiling     1024 px tiles, weighted probability blending, TTA, seam test (§5.3)
  calibrate  temperature scaling, hysteresis thresholds at the precision target (§5.4)
  model      network definition / training (needs ``uv sync --extra seg``)
"""
