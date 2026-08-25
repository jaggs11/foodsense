# Data

> **Phase 0 skeleton.** The builders and loaders referenced here land in Phase 1; the
> download instructions are exact and can be followed today.

## What is committed, and what is not

| Path | Committed? | Contents |
|------|-----------|----------|
| `data/raw/` | **No** (gitignored) | Full dataset downloads |
| `data/processed/` | Yes | `food_db.sqlite` + `food_db.parquet` -- the curated USDA subset (< 25 MB) |
| `data/samples/` | Yes | Small provenance-documented samples of Food.com and Nutrition5k |

The repository is designed so that **cloning it is enough** to run the full pipeline
offline. The downloads below are only needed to reproduce the curation step or to train
on the full corpora.

## 1. USDA FoodData Central (primary nutrient ground truth)

Source: <https://fdc.nal.usda.gov/download-datasets.html>
Releases used: **Foundation Foods** and **SR Legacy**, CSV format.
Licence: public domain (U.S. Government work).

```bash
# Download both CSV bundles into data/raw/fdc/ and unzip them, then:
python -m foodsense.data.build_food_db
```

This curates ~2-3k everyday foods across fruit, vegetable, grain, dairy, meat, legume and
common prepared-food categories, keeps ~30 nutrients per 100 g, and attaches the
`category`, `default_form`, `allowed_forms` and `tags` columns that the constraint layer
depends on.

## 2. Food.com Recipes and Interactions (primary meal corpus)

Source: Kaggle, `shuyangli94/food-com-recipes-and-user-interactions`
<https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions>

```bash
pip install -r requirements-optional.txt      # brings in kagglehub
# Requires Kaggle credentials: KAGGLE_USERNAME / KAGGLE_KEY in .env, or ~/.kaggle/kaggle.json
python -m foodsense.data.corpora --download foodcom
```

Without credentials the loader falls back to `data/samples/foodcom_sample.csv` and says so
explicitly rather than failing.

## 3. Nutrition5k (secondary corpus, dish metadata only)

Source: Google Research
<https://github.com/google-research-datasets/Nutrition5k>

Only the small per-dish macronutrient CSVs are used -- **not** the video/depth data:

```
https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv
https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv
```

```bash
python -m foodsense.data.corpora --download nutrition5k
```

A 300-dish sample is committed at `data/samples/nutrition5k_sample.csv`.

## Provenance

Each file in `data/samples/` is documented in `data/samples/README.md`: where it came from,
which rows were taken, and with what seed.
