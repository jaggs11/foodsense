"""Build the curated USDA food database from FoodData Central CSVs.

Downloads (or reads from ``data/raw/``) the Foundation Foods and SR Legacy releases,
filters to ~2-3k everyday foods, keeps the ~30 nutrients in
:data:`foodsense.schemas.NUTRIENTS` per 100 g, attaches ``category``,
``default_form``, ``allowed_forms`` and ``tags``, and writes
``data/processed/food_db.sqlite`` plus a parquet mirror.

TODO(Phase 1): implement.
"""
