"""Evaluation against the vector gold standard (§10). Roadmap Steps 2 and 8.

  pixel      precision/recall/IoU, boundary F at 1-2 px, seam test
  element    detection P/R at IoU >= 0.5, glyph confusion matrix
  track      buffer correctness/completeness/quality at 5/10 cm, lateral RMS/p95,
             pattern-correct length fraction, transition position error
  objects    crosswalk/stop line/area P/R, bar count, symbol type + heading error
  confidence reliability diagram, precision of the auto-accepted set
`unknown` is reported as abstention (coverage), never as an error.
"""
