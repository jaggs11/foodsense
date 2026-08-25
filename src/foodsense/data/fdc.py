"""Food database access: id lookup, fuzzy name matching, and nutrient recomputation.

Wraps ``data/processed/food_db.sqlite`` behind a small API used by Stage 3
(grounding generated names in real USDA ids) and Stage 4 (recomputing nutrients
from ground truth).

TODO(Phase 1): implement.
"""
