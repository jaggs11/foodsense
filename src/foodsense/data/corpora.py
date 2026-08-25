"""Corpus loaders: Food.com (primary) and Nutrition5k dish metadata (secondary).

Maps recipes/dishes onto meal nutrient vectors via the curated food database,
discarding ingredient matches below 60% confidence. Falls back to the committed
samples in ``data/samples/`` when the full downloads are unavailable.

TODO(Phase 1): implement.
"""
