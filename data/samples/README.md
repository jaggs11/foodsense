# Committed data samples

Small extracts of the two meal corpora, committed so that a fresh clone can run the whole
pipeline and the full test suite **with no network access and no credentials**. Both are
real rows from the real datasets — nothing here is synthetic or hand-written.

Regenerate with:

```bash
python -m foodsense.data.corpora --prepare
```

| File | Source dataset | Rows | Size | Extraction | Seed |
|------|----------------|------|------|------------|------|
| `foodcom_sample.csv` | Food.com recipes, via the Hugging Face mirror `Karo8870/food.com-parsed-dataset` | 500 | 365 KB | Uniform random sample (`random.Random(42).sample`) of the first 40,000 recipes, restricted to rows with a name, 2–12 ingredients, and stated calories in 80–1600 | 42 |
| `nutrition5k_sample.csv` | Nutrition5k dish metadata (Google Research), `dish_metadata_cafe1.csv` + `cafe2.csv` | 300 | 83 KB | Uniform random sample of all 5,006 parsed dishes with 2–12 ingredients | 42 |

## Columns

**`foodcom_sample.csv`** — `recipe_id`, `name`, `ingredients` (JSON list of
`{quantity, unit, name, description}`), `servings`, `serving_size` (g), and the recipe's
own per-serving nutrition: `calories`, `total_fat`, `saturated_fat`, `cholesterol`,
`sodium`, `carbohydrates`, `fiber`, `sugar`, `protein`.

**`nutrition5k_sample.csv`** — `dish_id`, `total_calories`, `total_mass_g`,
`total_fat_g`, `total_carb_g`, `total_protein_g`, `ingredients` (JSON list of
`{name, grams}` with real measured gram weights).

The corpora's own nutrition columns are kept **only as a check** on our USDA-derived
reconstruction — the pipeline never trains on them. See
[`../README.md`](../README.md) for the measured reconstruction error.

## Licensing

- **Food.com** — recipe data scraped from Food.com. Redistributed here only as a 500-row
  excerpt for reproducibility of the demo; the full dataset remains under its original
  terms.
- **Nutrition5k** — Google Research, released under CC BY 4.0.
- **USDA FoodData Central** (`../processed/`) — a U.S. Government work, public domain.
