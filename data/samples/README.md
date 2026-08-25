# Committed data samples

Small, redistributable extracts that let the pipeline run without any download.
Each is documented with its source, extraction rule and seed, so it can be regenerated
exactly.

> **Phase 0 skeleton** -- the samples themselves are generated in Phase 1 and this table
> is filled in with real row counts at that point. No file is listed here before it exists.

| File | Source dataset | Rows | Extraction | Seed |
|------|----------------|------|------------|------|
| `foodcom_sample.csv` | Food.com Recipes and Interactions (Kaggle) | _(Phase 1)_ | _(Phase 1)_ | 42 |
| `nutrition5k_sample.csv` | Nutrition5k dish metadata (Google Research) | _(Phase 1)_ | _(Phase 1)_ | 42 |

## Licensing

- **Food.com** data is redistributed here only as a small excerpt for reproducibility of
  the demo; the full dataset remains under its Kaggle terms.
- **Nutrition5k** is released by Google Research under CC BY 4.0.
- **USDA FoodData Central** is a U.S. Government work and is in the public domain.
