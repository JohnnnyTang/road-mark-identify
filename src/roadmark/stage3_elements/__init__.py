"""Stage 3: paint elements (§6). Roadmap Step 5.

  polygonize   per-block polygonization, unary_union across block edges, centroid ownership
  descriptors  oriented rectangle, area/solidity, skeleton, no-data buffer fraction (§6.2)
  direction    structure-tensor road-direction field + coherence (intersection cores) (§6.3)
  repair       fragment merging: collinear + similar width + (wear gap | no-data gap) (§6.4)
  typing       stroke / bar / glyph / hatch_stripe / noise / unknown; glyph CNN with open-set
               rejection (§6.5)
"""
