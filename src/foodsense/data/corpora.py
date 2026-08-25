"""Meal corpora: Food.com recipes (primary) and Nutrition5k dishes (secondary).

Both corpora are turned into the same thing -- a :class:`CorpusMeal`, which is a
list of ``(food_id, quantity_g, form)`` items resolved against the curated USDA
database, plus the full 33-nutrient vector recomputed from it. That is what makes
the two-corpus comparison meaningful: identical representation, identical nutrient
definitions, only the source of the meals differs.

Nutrients are **not** taken from the corpus. Both corpora ship their own macro
columns, and we keep them -- but only as a *check*: ``CorpusMeal.stated`` records
what the corpus claimed and ``reconstruction_error`` reports how far our
USDA-derived vector lands from it. A corpus we cannot reconstruct is a corpus we
should not train on, and that number says so out loud.

    python -m foodsense.data.corpora --prepare      # download, build samples, report
    python -m foodsense.data.corpora --stats        # report on what is already local

Sources
-------
Food.com
    Recipes scraped from Food.com, with structured ingredient lists (quantity,
    unit, name), per-serving nutrition in absolute units, and a serving size in
    grams. Obtained from the public Hugging Face mirror
    ``Karo8870/food.com-parsed-dataset`` -- no credentials, so a clean clone can
    reproduce the whole pipeline.

    The project brief names the Kaggle bundle
    ``shuyangli94/food-com-recipes-and-user-interactions``. That is the same
    website's data, but Kaggle requires credentials, which breaks "clone and
    run"; and its nutrition columns are percentages of daily value rather than
    absolute amounts. The loader still prefers the Kaggle bundle when
    credentials happen to be configured. See data/README.md.
Nutrition5k
    Google Research; only the small per-dish metadata CSVs are used, never the
    video or depth data. Public GCS bucket, no credentials. CC BY 4.0. Ships real
    per-ingredient gram weights, so nothing about its masses is estimated.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from foodsense import RAW_DIR, SAMPLES_DIR, SEED
from foodsense.data.fdc import FoodDB, FoodRecord, get_food_db
from foodsense.schemas import Meal, MealItem, NutrientVector

__all__ = [
    "CorpusMeal",
    "load_foodcom",
    "load_meals",
    "load_nutrition5k",
    "recipes_to_meals",
]

FOODCOM_DIR = RAW_DIR / "foodcom"
NUTRITION5K_DIR = RAW_DIR / "nutrition5k"

FOODCOM_PARQUET_INDEX = (
    "https://huggingface.co/api/datasets/Karo8870/food.com-parsed-dataset/parquet/default/train"
)
KAGGLE_DATASET = "shuyangli94/food-com-recipes-and-user-interactions"
NUTRITION5K_URLS = {
    "dish_metadata_cafe1.csv": (
        "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"
        "metadata/dish_metadata_cafe1.csv"
    ),
    "dish_metadata_cafe2.csv": (
        "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"
        "metadata/dish_metadata_cafe2.csv"
    ),
}

FOODCOM_SAMPLE = SAMPLES_DIR / "foodcom_sample.csv"
NUTRITION5K_SAMPLE = SAMPLES_DIR / "nutrition5k_sample.csv"
FOODCOM_SAMPLE_SIZE = 500
NUTRITION5K_SAMPLE_SIZE = 300

#: Ingredient names below this fuzzy-match score are dropped, per the design brief.
MIN_INGREDIENT_MATCH = 60.0

#: A "meal" with one ingredient or thirty is not what the optimiser is for.
MIN_ITEMS, MAX_ITEMS = 2, 12

#: Per-meal energy sanity band, kcal. Outside this it is not a single meal.
KCAL_BAND = (80.0, 1600.0)

#: A single ingredient contributing more than this is a unit-parsing failure.
MAX_INGREDIENT_G = 1500.0

#: Data-quality gate. Our USDA reconstruction and the corpus's own stated energy
#: should broadly agree; when they do not, either the ingredient match or the unit
#: conversion went wrong, and the meal is not worth training on. Deliberately
#: loose -- it is a junk filter, not an accuracy claim.
MAX_ENERGY_DISAGREEMENT = 0.60


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
#
# Food.com ingredients arrive as {quantity, unit, name}. Roughly 98% of unit
# mentions fall into the table below; the long tail ("envelope", "sprig") and the
# 17% with no unit at all are handled by per-piece masses.

#: Units that convert straight to grams.
_MASS_UNITS_G: dict[str, float] = {
    "g": 1.0, "gram": 1.0, "grams": 1.0, "gm": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "mg": 0.001,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.6, "lbs": 453.6, "pound": 453.6, "pounds": 453.6,
}

#: Units that convert to millilitres; grams then depend on the food's density.
_VOLUME_UNITS_ML: dict[str, float] = {
    "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0, "millilitre": 1.0,
    "l": 1000.0, "liter": 1000.0, "liters": 1000.0, "litre": 1000.0,
    "cup": 236.6, "cups": 236.6, "c": 236.6,
    "tablespoon": 14.79, "tablespoons": 14.79, "tbsp": 14.79, "tbs": 14.79, "tb": 14.79,
    "teaspoon": 4.93, "teaspoons": 4.93, "tsp": 4.93, "ts": 4.93,
    "pint": 473.2, "pints": 473.2,
    "quart": 946.4, "quarts": 946.4,
    "gallon": 3785.0, "gallons": 3785.0,
    "fl oz": 29.57, "fluid ounce": 29.57, "fluid ounces": 29.57,
    "pinch": 0.36, "pinches": 0.36, "dash": 0.6, "dashes": 0.6,
    "drop": 0.05, "drops": 0.05,
}

#: Container and piece units with a typical mass, independent of the food.
_PIECE_UNITS_G: dict[str, float] = {
    "can": 400.0, "cans": 400.0, "jar": 400.0, "jars": 400.0,
    "box": 350.0, "boxes": 350.0, "bag": 350.0, "bags": 350.0,
    "package": 250.0, "packages": 250.0, "pkg": 250.0, "container": 250.0,
    "envelope": 7.0, "envelopes": 7.0, "packet": 7.0, "packets": 7.0,
    "clove": 3.0, "cloves": 3.0,
    "slice": 25.0, "slices": 25.0,
    "stalk": 40.0, "stalks": 40.0,
    "bunch": 100.0, "bunches": 100.0,
    "head": 500.0, "heads": 500.0,
    "sprig": 1.0, "sprigs": 1.0,
    "leaf": 0.5, "leaves": 0.5,
    "ear": 90.0, "ears": 90.0,
    "strip": 15.0, "strips": 15.0,
    "sheet": 10.0, "sheets": 10.0,
    "stick": 113.0, "sticks": 113.0,
    "fillet": 150.0, "fillets": 150.0,
    "breast": 170.0, "breasts": 170.0,
    "link": 60.0, "links": 60.0,
}

#: Size adjectives Food.com uses in the unit slot; multipliers on a piece.
_SIZE_MULTIPLIERS: dict[str, float] = {
    "large": 1.25, "medium": 1.0, "small": 0.7, "whole": 1.0, "piece": 1.0,
    "pieces": 1.0, "extra large": 1.5, "jumbo": 1.5,
}

#: Density in g/ml by FoodSense category, for the volume units.
_DENSITY_G_PER_ML: dict[str, float] = {
    "fat_oil": 0.92, "beverage": 1.0, "dairy": 1.03, "soup_sauce": 1.03,
    "sweets": 0.85, "grain": 0.78, "legume": 0.80, "nut_seed": 0.55,
    "herb_spice": 0.30, "baked": 0.45, "fruit": 0.65, "vegetable": 0.62,
    "meat": 0.90, "poultry": 0.90, "fish": 0.90, "processed_meat": 0.90,
    "cereal": 0.40, "snack": 0.35, "prepared": 0.90, "baby_food": 1.0,
}
_DEFAULT_DENSITY = 0.80

#: Mass of one piece by category, for ingredients given as a bare count.
_PIECE_G_BY_CATEGORY: dict[str, float] = {
    "fruit": 120.0, "vegetable": 85.0, "dairy": 40.0, "meat": 110.0,
    "poultry": 120.0, "fish": 120.0, "processed_meat": 45.0, "baked": 40.0,
    "grain": 50.0, "legume": 50.0, "nut_seed": 4.0, "fat_oil": 14.0,
    "sweets": 15.0, "snack": 20.0, "beverage": 240.0, "soup_sauce": 120.0,
    "cereal": 30.0, "herb_spice": 2.0, "prepared": 150.0, "baby_food": 100.0,
}
_DEFAULT_PIECE_G = 60.0

#: Foods counted by the piece often enough to be worth naming exactly.
_PIECE_G_BY_NAME: tuple[tuple[str, float], ...] = (
    (r"\begg", 50.0),
    (r"\bgarlic\b", 3.0),
    (r"\bonion", 110.0),
    (r"\bcarrot", 61.0),
    (r"\bpotato", 170.0),
    (r"\bbanana", 118.0),
    (r"\bapple", 182.0),
    (r"\blemon\b", 58.0),
    (r"\blime\b", 67.0),
    (r"\btomato", 123.0),
    (r"\bcelery\b", 40.0),
    (r"\bbay lea", 0.2),
)

_FRACTION_RE = re.compile(r"^\s*(\d+)?\s*(?:(\d+)\s*/\s*(\d+))?\s*$")


def _parse_quantity(text: object) -> float:
    """Turn "2", "1/4", "2 1/2" into a float. Unparseable amounts become 1.0."""
    if isinstance(text, int | float):
        return float(text) if text and text > 0 else 1.0
    if not isinstance(text, str):
        return 1.0
    match = _FRACTION_RE.match(text.replace("-", " ").strip())
    if not match:
        return 1.0
    whole, numerator, denominator = match.groups()
    total = float(whole) if whole else 0.0
    if numerator and denominator and float(denominator):
        total += float(numerator) / float(denominator)
    return total if total > 0 else 1.0


def _piece_mass(record: FoodRecord) -> float:
    name = record.name.lower()
    for pattern, grams in _PIECE_G_BY_NAME:
        if re.search(pattern, name):
            return grams
    return _PIECE_G_BY_CATEGORY.get(record.category, _DEFAULT_PIECE_G)


def _to_grams(quantity: float, unit: object, record: FoodRecord) -> float:
    """Convert one ``(quantity, unit)`` pair to grams for a specific food.

    Volume units need the food's density and bare counts need its typical piece
    mass, which is why this takes the matched :class:`FoodRecord` rather than
    working on the text alone.
    """
    key = (unit or "").strip().lower() if isinstance(unit, str) else ""

    if key in _MASS_UNITS_G:
        return quantity * _MASS_UNITS_G[key]
    if key in _VOLUME_UNITS_ML:
        density = _DENSITY_G_PER_ML.get(record.category, _DEFAULT_DENSITY)
        return quantity * _VOLUME_UNITS_ML[key] * density
    if key in _PIECE_UNITS_G:
        return quantity * _PIECE_UNITS_G[key]
    multiplier = _SIZE_MULTIPLIERS.get(key, 1.0)
    return quantity * multiplier * _piece_mass(record)


# ---------------------------------------------------------------------------
# Corpus meal
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CorpusMeal:
    """A corpus recipe/dish resolved onto the curated USDA food database."""

    source: str
    source_id: str
    name: str
    items: list[MealItem]
    nutrients: NutrientVector
    stated: dict[str, float] = field(default_factory=dict)
    match_confidence: float = 0.0
    n_ingredients_input: int = 0
    n_ingredients_matched: int = 0

    @property
    def meal(self) -> Meal:
        return Meal(items=self.items)

    @property
    def match_rate(self) -> float:
        if not self.n_ingredients_input:
            return 0.0
        return self.n_ingredients_matched / self.n_ingredients_input

    def reconstruction_error(self) -> dict[str, float]:
        """Signed relative error of our USDA vector against the corpus's own claims.

        ``(ours - theirs) / theirs`` per macro. Nothing here is fitted to the
        corpus's numbers, so every entry is an independent check.
        """
        ours = self.nutrients.as_dict()
        return {
            key: (ours[key] - value) / value
            for key, value in self.stated.items()
            if key in ours and value
        }


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def _fetch(url: str, dest: Path, label: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  cached   {label} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"  download {label} <- {url[:96]}")
    req = urllib.request.Request(url, headers={"User-Agent": "foodsense/0.1"})
    with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"           {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def download_foodcom() -> Path | None:
    """Fetch the Food.com recipe corpus. Returns ``None`` if the network is unavailable."""
    kaggle_path = _try_kaggle()
    if kaggle_path is not None:
        return kaggle_path
    dest = FOODCOM_DIR / "recipes.parquet"
    if dest.exists():
        print(f"  cached   food.com recipes.parquet ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    try:
        req = urllib.request.Request(
            FOODCOM_PARQUET_INDEX, headers={"User-Agent": "foodsense/0.1"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            shards = json.loads(r.read())
        # One shard is 380k recipes -- far more than the pipeline needs.
        return _fetch(shards[0], dest, "food.com recipes.parquet")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, IndexError) as exc:
        print(f"  WARNING: Food.com download failed ({exc}); falling back to the committed sample.")
        return None


def _try_kaggle() -> Path | None:
    """Use the Kaggle bundle from the brief only when credentials are already configured."""
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        return None
    try:
        import kagglehub
    except ImportError:
        print("  Kaggle credentials found but kagglehub is not installed")
        print("  (pip install -r requirements-optional.txt). Using the public mirror instead.")
        return None
    try:
        root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:  # noqa: BLE001 - kagglehub raises a wide variety
        print(f"  Kaggle download failed ({exc}); using the public mirror instead.")
        return None
    for candidate in ("RAW_recipes.csv", "recipes.csv"):
        path = root / candidate
        if path.exists():
            print(f"  kaggle   {path}")
            return path
    return None


def download_nutrition5k() -> list[Path]:
    """Fetch the two Nutrition5k dish-metadata CSVs (public, no credentials)."""
    paths: list[Path] = []
    for name, url in NUTRITION5K_URLS.items():
        try:
            paths.append(_fetch(url, NUTRITION5K_DIR / name, f"nutrition5k {name}"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  WARNING: {name} download failed ({exc}).")
    return paths


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

FOODCOM_MACROS = {
    "calories": "energy_kcal",
    "total_fat": "fat_g",
    "saturated_fat": "saturated_fat_g",
    "cholesterol": "cholesterol_mg",
    "sodium": "sodium_mg",
    "carbohydrates": "carbohydrate_g",
    "fiber": "fiber_g",
    "sugar": "sugars_g",
    "protein": "protein_g",
}


def _parse_ingredients(value: object) -> list[dict]:
    """Parse the Food.com ingredient JSON: ``[{quantity, unit, name, description}]``."""
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [v for v in parsed if isinstance(v, dict)] if isinstance(parsed, list) else []


def _parse_nutrition5k_row(fields: list[str]) -> dict | None:
    """One Nutrition5k metadata row -> dish totals plus (ingredient, grams) pairs.

    Rows are ragged: six dish-level fields followed by seven fields per ingredient.
    """
    if len(fields) < 6:
        return None
    try:
        dish = {
            "dish_id": fields[0],
            "total_calories": float(fields[1]),
            "total_mass_g": float(fields[2]),
            "total_fat_g": float(fields[3]),
            "total_carb_g": float(fields[4]),
            "total_protein_g": float(fields[5]),
        }
    except ValueError:
        return None

    ingredients: list[dict] = []
    rest = fields[6:]
    for i in range(0, len(rest) - 6, 7):
        chunk = rest[i : i + 7]
        if not chunk[0].startswith("ingr_"):
            continue
        try:
            ingredients.append({"name": chunk[1], "grams": float(chunk[2])})
        except ValueError:
            continue
    dish["ingredients"] = ingredients
    return dish


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_foodcom(path: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load Food.com recipes, from the full download if present else the sample."""
    full = path or (FOODCOM_DIR / "recipes.parquet")
    columns = ["name", "ingredients", "servings", "serving_size", *FOODCOM_MACROS]
    if full.exists():
        frame = pd.read_parquet(full, columns=columns)
        if limit:
            frame = frame.head(limit)
        frame = frame.reset_index(names="recipe_id")
    elif FOODCOM_SAMPLE.exists():
        frame = pd.read_csv(FOODCOM_SAMPLE, nrows=limit)
    else:
        raise FileNotFoundError(
            "No Food.com data available. Run `python -m foodsense.data.corpora --prepare` "
            "(needs a network connection once). See data/README.md."
        )
    frame["ingredients"] = frame["ingredients"].apply(_parse_ingredients)
    return frame


def load_nutrition5k(limit: int | None = None) -> pd.DataFrame:
    """Load Nutrition5k dish metadata, from the full download if present else the sample."""
    files = sorted(NUTRITION5K_DIR.glob("dish_metadata_cafe*.csv"))
    if files:
        rows: list[dict] = []
        for path in files:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parsed = _parse_nutrition5k_row(line.rstrip("\n").split(","))
                    if parsed is not None:
                        rows.append(parsed)
        frame = pd.DataFrame(rows)
    elif NUTRITION5K_SAMPLE.exists():
        frame = pd.read_csv(NUTRITION5K_SAMPLE)
        frame["ingredients"] = frame["ingredients"].apply(
            lambda v: json.loads(v) if isinstance(v, str) else v
        )
    else:
        raise FileNotFoundError(
            "No Nutrition5k data available. Run `python -m foodsense.data.corpora --prepare`. "
            "See data/README.md."
        )
    return frame.head(limit) if limit else frame


# ---------------------------------------------------------------------------
# Recipe -> meal
# ---------------------------------------------------------------------------


def _resolve(
    db: FoodDB, ingredients: list[dict], grams_key: str | None
) -> tuple[list[MealItem], list[float]]:
    """Fuzzy-match ingredient names onto USDA foods and assign gram amounts.

    ``grams_key`` names a field holding real gram weights (Nutrition5k); when it
    is ``None`` the amount is derived from ``quantity`` + ``unit`` (Food.com).
    """
    items: list[MealItem] = []
    scores: list[float] = []
    for ingredient in ingredients:
        raw_name = str(ingredient.get("name") or "").strip()
        if not raw_name:
            continue
        record, score = db.match(raw_name, threshold=MIN_INGREDIENT_MATCH)
        if record is None:
            continue

        if grams_key is not None:
            grams = float(ingredient.get(grams_key) or 0.0)
        else:
            grams = _to_grams(
                _parse_quantity(ingredient.get("quantity")), ingredient.get("unit"), record
            )

        grams = round(min(grams, MAX_INGREDIENT_G), 1)
        if grams <= 0:
            continue
        items.append(record.as_item(grams))
        scores.append(score)
    return items, scores


def _serving_scale(items: list[MealItem], serving_size: float, servings: float) -> float:
    """Factor that turns a whole-recipe ingredient list into one serving.

    Prefers the stated grams-per-serving against our reconstructed total mass;
    falls back to 1/servings; otherwise leaves the recipe alone.
    """
    total_mass = sum(i.quantity_g for i in items)
    if total_mass <= 0:
        return 1.0
    if 0 < serving_size < total_mass:
        return serving_size / total_mass
    if servings and servings >= 1:
        return 1.0 / servings
    return 1.0


def recipes_to_meals(
    frame: pd.DataFrame,
    source: str,
    db: FoodDB | None = None,
    limit: int | None = None,
) -> list[CorpusMeal]:
    """Convert a loaded corpus frame into :class:`CorpusMeal` objects."""
    db = db or get_food_db()
    meals: list[CorpusMeal] = []
    is_n5k = source == "nutrition5k"

    for row in frame.to_dict(orient="records"):
        ingredients = list(row.get("ingredients") or [])
        if not ingredients:
            continue

        if is_n5k:
            items, scores = _resolve(db, ingredients, grams_key="grams")
            stated = {
                "energy_kcal": float(row.get("total_calories") or 0.0),
                "fat_g": float(row.get("total_fat_g") or 0.0),
                "carbohydrate_g": float(row.get("total_carb_g") or 0.0),
                "protein_g": float(row.get("total_protein_g") or 0.0),
            }
            source_id = str(row.get("dish_id", ""))
            name = source_id
        else:
            items, scores = _resolve(db, ingredients, grams_key=None)
            scale = _serving_scale(
                items,
                float(row.get("serving_size") or 0.0),
                float(row.get("servings") or 0.0),
            )
            items = [
                i.model_copy(update={"quantity_g": round(i.quantity_g * scale, 1)}) for i in items
            ]
            stated = {
                nutrient: float(row[column])
                for column, nutrient in FOODCOM_MACROS.items()
                if column in row and pd.notna(row[column])
            }
            source_id = str(row.get("recipe_id", ""))
            name = str(row.get("name", source_id))

        items = [i for i in items if i.quantity_g > 0]
        if not MIN_ITEMS <= len(items) <= MAX_ITEMS:
            continue

        nutrients = db.nutrients_for(items)
        if not KCAL_BAND[0] <= nutrients.energy_kcal <= KCAL_BAND[1]:
            continue

        # Junk filter: if our reconstruction and the corpus disagree wildly about
        # energy, either the ingredient match or the unit conversion failed.
        stated_kcal = stated.get("energy_kcal", 0.0)
        if stated_kcal > 0:
            disagreement = abs(nutrients.energy_kcal - stated_kcal) / stated_kcal
            if disagreement > MAX_ENERGY_DISAGREEMENT:
                continue

        meals.append(
            CorpusMeal(
                source=source,
                source_id=source_id,
                name=name,
                items=items,
                nutrients=nutrients,
                stated=stated,
                match_confidence=sum(scores) / len(scores) if scores else 0.0,
                n_ingredients_input=len(ingredients),
                n_ingredients_matched=len(items),
            )
        )
        if limit and len(meals) >= limit:
            break
    return meals


def load_meals(source: str, limit: int | None = None, rows: int | None = None) -> list[CorpusMeal]:
    """Convenience entry point used by Stage 1: ``load_meals("foodcom", limit=8000)``."""
    if source == "foodcom":
        frame = load_foodcom(limit=rows)
    elif source == "nutrition5k":
        frame = load_nutrition5k(limit=rows)
    else:
        raise ValueError(f"Unknown corpus {source!r}; expected 'foodcom' or 'nutrition5k'")
    return recipes_to_meals(frame, source=source, limit=limit)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def write_samples() -> dict[str, int]:
    """Write the committed samples that make the repository runnable without downloads."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    rng = random.Random(SEED)

    if (FOODCOM_DIR / "recipes.parquet").exists():
        frame = load_foodcom(limit=40_000)
        usable = frame[
            frame["ingredients"].apply(lambda x: MIN_ITEMS <= len(x) <= MAX_ITEMS)
            & frame["calories"].between(*KCAL_BAND)
            & frame["name"].notna()
        ]
        picks = sorted(rng.sample(range(len(usable)), min(FOODCOM_SAMPLE_SIZE, len(usable))))
        sample = usable.iloc[picks].copy()
        sample["ingredients"] = sample["ingredients"].apply(json.dumps)
        sample.to_csv(FOODCOM_SAMPLE, index=False)
        written["foodcom_sample.csv"] = len(sample)

    if list(NUTRITION5K_DIR.glob("dish_metadata_cafe*.csv")):
        frame = load_nutrition5k()
        usable = frame[frame["ingredients"].apply(lambda x: MIN_ITEMS <= len(x) <= MAX_ITEMS)]
        picks = sorted(rng.sample(range(len(usable)), min(NUTRITION5K_SAMPLE_SIZE, len(usable))))
        sample = usable.iloc[picks].copy()
        sample["ingredients"] = sample["ingredients"].apply(json.dumps)
        sample.to_csv(NUTRITION5K_SAMPLE, index=False)
        written["nutrition5k_sample.csv"] = len(sample)

    return written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def report(source: str, meals: list[CorpusMeal], n_rows: int) -> None:
    print(f"\n--- {source}: {len(meals)} usable meals from {n_rows} rows "
          f"({len(meals) / n_rows * 100:.1f}%) ---")
    if not meals:
        print("  (none)")
        return
    print(f"  mean ingredient match score : "
          f"{sum(m.match_confidence for m in meals) / len(meals):.1f}")
    print(f"  mean ingredients matched    : "
          f"{sum(m.n_ingredients_matched for m in meals) / len(meals):.2f} of "
          f"{sum(m.n_ingredients_input for m in meals) / len(meals):.2f}")
    print(f"  median energy (kcal)        : {_median([m.nutrients.energy_kcal for m in meals]):.0f}")
    print(f"  median protein (g)          : {_median([m.nutrients.protein_g for m in meals]):.1f}")
    print(f"  median sodium (mg)          : {_median([m.nutrients.sodium_mg for m in meals]):.0f}")

    print("  reconstruction vs the corpus's own figures (median |relative error|):")
    errors_by_key: dict[str, list[float]] = {}
    for meal in meals:
        for key, error in meal.reconstruction_error().items():
            errors_by_key.setdefault(key, []).append(abs(error))
    for key in sorted(errors_by_key):
        values = errors_by_key[key]
        print(f"    {key:<20} {_median(values) * 100:6.1f}%   (n={len(values)})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="download corpora and write samples")
    parser.add_argument("--download", choices=["foodcom", "nutrition5k", "all"], default=None)
    parser.add_argument("--stats", action="store_true", help="report on locally available corpora")
    parser.add_argument("--rows", type=int, default=20_000, help="corpus rows to read for the report")
    args = parser.parse_args(argv)

    if args.prepare or args.download in {"foodcom", "all"}:
        print("Food.com:")
        download_foodcom()
    if args.prepare or args.download in {"nutrition5k", "all"}:
        print("Nutrition5k:")
        download_nutrition5k()

    if args.prepare:
        print("\nWriting committed samples...")
        for name, count in write_samples().items():
            print(f"  {name}: {count} rows")

    if args.prepare or args.stats:
        db = get_food_db()
        print(f"\nFood database: {len(db)} foods")
        for source in ("foodcom", "nutrition5k"):
            try:
                frame = (
                    load_foodcom(limit=args.rows)
                    if source == "foodcom"
                    else load_nutrition5k(limit=args.rows)
                )
            except FileNotFoundError as exc:
                print(f"\n--- {source}: unavailable ---\n  {exc}")
                continue
            report(source, recipes_to_meals(frame, source=source, db=db), len(frame))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
