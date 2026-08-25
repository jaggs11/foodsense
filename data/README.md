# Data

Everything the pipeline needs to run is **already in this repository**. The downloads
below are only required to reproduce the curation step or to train on the full corpora.

| Path | Committed? | Contents |
|------|-----------|----------|
| `data/raw/` | **No** (gitignored) | Full dataset downloads (~200 MB) |
| `data/processed/` | Yes (1.3 MB) | `food_db.sqlite` + `food_db.parquet` — the curated USDA subset |
| `data/samples/` | Yes (0.2 MB) | 500 Food.com recipes + 300 Nutrition5k dishes, with provenance |

```bash
make data        # ./make.ps1 data on Windows — downloads, curates, writes samples
```

---

## 1. USDA FoodData Central — the nutrient ground truth

Source: <https://fdc.nal.usda.gov/download-datasets.html> · Licence: public domain (U.S. Government work)

Two releases are consumed, pinned by filename so a rebuild is reproducible:

| Release | File | Rows in |
|---------|------|---------|
| SR Legacy | `FoodData_Central_sr_legacy_food_csv_2018-04.zip` | 7,793 |
| Foundation Foods | `FoodData_Central_foundation_food_csv_2025-04-24.zip` | 411 |

```bash
python -m foodsense.data.build_food_db --stats
```

The script downloads both bundles into `data/raw/fdc/` (cached after the first run),
then curates them down to everyday foods.

### What curation actually does

| Step | Foods remaining |
|------|-----------------|
| SR Legacy + Foundation, mapped to a FoodSense category | 8,039 |
| after exclusion patterns (raw muscle food, dry staples, `unprepared`, infant formula, …) | 6,377 |
| with an energy value | 6,353 |
| after de-duplicating identical descriptions | 6,236 |
| **after per-category caps** | **2,590** |

SR Legacy carries 954 beef entries that differ only in trim and grade. Rows are ranked
by an "everyday-ness" score (comma count, parentheses, shouty brand tokens, length) and
capped per category. 376 foods are **force-included** regardless of the cap, because the
demo scenarios and safety rules depend on them existing — curation must never silently
delete the thing under test.

### Columns beyond the nutrients

Each row carries 33 nutrients per 100 g plus four columns the constraint layer needs:

- **`category`** — broad taxonomy (20 categories: vegetable 290, meat 230, fruit 210,
  prepared 180, dairy 170, baked 160, fish 140, soup_sauce 135, grain 120, legume 120,
  beverage 110, sweets 110, poultry 110, nut_seed 95, snack 95, baby_food 75,
  processed_meat 70, cereal 70, fat_oil 55, herb_spice 45).
- **`hazard_class`** — the specific choking-hazard class the AAP/CDC rules key on, kept
  separate from `category` because "fruit" is not what makes a grape unsafe:
  meat_chunk 421, nut 35, seed 24, hard_raw_vegetable 23, nut_butter 22, popcorn 15,
  hot_dog 14, grape 7, hard_candy 4, marshmallow 3, gum 2.
- **`allowed_forms` / `default_form`** — which preparation forms the food can physically
  take. Popcorn, marshmallow, hard candy and gum allow **only** `whole`: no preparation
  makes them safe for a small child, so the optimiser is given no escape hatch.
- **`tags`** — medication and diet interaction markers: high_sodium 640,
  high_potassium 505, high_tyramine 99, low_sodium_variant 86, leafy_green_vitk 83,
  added_sugar_source 63, cured_meat 57, alcohol 36, aged_cheese 35, whole_nut 35,
  raw_hard_veg 23, nut_butter 22, caffeine 20, grapefruit 18, honey 9.

### Known data limitations

- **`added_sugars_g` is 0.0 for every food.** Neither SR Legacy (2018) nor Foundation
  (2025-04) reports FDC nutrient 1235. The column is stored as `0.0` and **not**
  estimated. The toddler added-sugar rule therefore cannot use it directly; Phase 2
  documents the proxy it uses instead.
- Nutrient coverage varies by nutrient: energy 99.6%, protein 95.1%, sodium 94.4%,
  potassium 96.8%, calcium 96.4%, carbohydrate 83.1%, sugars 56.6%, fibre 59.0%,
  vitamin D 16.6%. A 0% share for cholesterol in plant foods is a *true zero*, not a gap
  — the coverage figure counts non-zero values, not presence.
- `low_sodium_variant` is a claim made by the food's **name**; `high_sodium` is its
  **measured** value. A food can carry both — USDA lists "Crackers, saltines, fat-free,
  low-sodium" at 849 mg/100 g. Exactly the sort of gap Stage 4 exists to catch.

---

## 2. Food.com — primary meal corpus

Recipes scraped from Food.com, with structured ingredient lists (quantity, unit, name),
per-serving nutrition in absolute units, and a serving size in grams.

**Obtained from:** the public Hugging Face mirror
[`Karo8870/food.com-parsed-dataset`](https://huggingface.co/datasets/Karo8870/food.com-parsed-dataset)
— 380,000 recipes in the first parquet shard, no credentials required.

> **Deviation from the project brief, stated plainly.** The brief names the Kaggle
> bundle `shuyangli94/food-com-recipes-and-user-interactions`. That is the same
> website's data, but Kaggle requires credentials, which breaks "clone and run", and its
> nutrition columns are percentages of daily value rather than absolute amounts. Two
> other Hugging Face mirrors of the Food.com scrape (`AkashPS11/recipes_data_food.com`
> and `Vatazh0k/recipes_data_food.com`) were tried first and **are truncated** — 1,228
> real rows out of 1,048,543, the rest null — so they were rejected. The loader still
> prefers the Kaggle bundle automatically whenever `~/.kaggle/kaggle.json` exists.

```bash
pip install -r requirements-optional.txt   # only if you want the Kaggle path
python -m foodsense.data.corpora --download foodcom
```

---

## 3. Nutrition5k — secondary meal corpus

Google Research. Only the small per-dish metadata CSVs are used — **never** the video or
depth data. Public GCS bucket, no credentials, CC BY 4.0.

```
https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv
https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv
```

```bash
python -m foodsense.data.corpora --download nutrition5k
```

5,006 dishes, each with real per-ingredient **gram weights** — so nothing about
Nutrition5k's masses is estimated.

---

## 4. How a recipe becomes a meal

Both corpora are resolved into the same representation: a list of
`(food_id, quantity_g, form)` items against the curated USDA database, with the
33-nutrient vector recomputed from it.

**Nutrients are never taken from the corpus.** Both corpora ship their own macro columns,
and both are kept — but only as a *check*. `CorpusMeal.reconstruction_error()` reports how
far our USDA-derived vector lands from what the corpus claimed, and nothing is fitted to
those numbers, so every figure below is an independent measurement.

1. **Match** each ingredient name to a USDA food (fuzzy, ≥ 60 confidence; below that the
   ingredient is dropped).
2. **Convert to grams.**
   - *Nutrition5k* ships real gram weights — used directly.
   - *Food.com* gives `{quantity, unit, name}`. A unit table covering ~98% of mentions
     converts mass units exactly, volume units through a per-category density
     (a cup of oil ≠ a cup of flour), and bare counts through a per-piece mass.
3. **Scale to one serving** using the stated grams-per-serving against the reconstructed
   total mass (Food.com only; Nutrition5k dishes are already single servings).
4. **Filter**: 2–12 items, 80–1600 kcal, and reconstructed energy within 60% of the
   corpus's stated energy. That last one is a junk filter — when our reconstruction and
   the corpus disagree that badly, the ingredient match or the unit conversion failed,
   and the meal is not worth training on.

### Measured reconstruction quality

Run `python -m foodsense.data.corpora --stats` to regenerate. Median absolute relative
error against each corpus's own stated figures:

| | Food.com | Nutrition5k |
|---|---|---|
| usable meals | 10,575 from 20,000 rows (52.9%) | 1,602 from 5,006 rows (32.0%) |
| mean ingredient match score | 90.0 | 93.0 |
| ingredients matched | 6.23 of 9.20 | 5.44 of 6.02 |
| **energy** | 21.5% | **15.1%** |
| carbohydrate | 36.0% | 22.1% |
| fat | 34.4% | 23.3% |
| protein | 35.5% | 20.3% |
| sodium | 55.8% | — |
| fibre | 54.4% | — |
| sugars | 48.6% | — |

Nutrition5k reconstructs better because it supplies real gram weights; Food.com's error
includes unit-conversion uncertainty on top of ingredient matching. Sodium is the worst
column in both, which is expected — it is dominated by added salt and by
brand-to-brand variation that an ingredient name cannot capture.

---

## Provenance of the committed samples

See [`data/samples/README.md`](samples/README.md).
