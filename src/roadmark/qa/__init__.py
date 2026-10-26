"""Quality control (§9). Roadmap Step 7.

  rules       dimensional, periodicity, geometry, topology, consistency, coverage, duplicates
  confidence  per-object score from probability, dimension deviation, fit residual,
              observed fraction, classifier probability; weights fitted on validation data
  review      decision bands (auto-accept / needs review / reject), review queue, QGIS project
"""

ERROR_CAUSES = ("segmentation_miss", "segmentation_false_positive", "fragment_merge", "linking",
                "profile", "classification", "occlusion", "reference_error")  # §10.4
