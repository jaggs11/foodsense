"""Build the curated USDA food database that the whole pipeline is grounded in.

Reads the USDA FoodData Central **SR Legacy** and **Foundation Foods** CSV releases,
curates them down to a few thousand everyday foods, and writes
``data/processed/food_db.sqlite`` plus a parquet mirror.

Beyond the ~30 nutrients per 100 g, every row carries four columns the constraint
layer depends on:

``category``
    Broad FoodSense taxonomy (fruit, vegetable, meat, dairy, ...).
``hazard_class``
    The specific choking-hazard class the AAP/CDC rules are written against
    (``grape``, ``nut``, ``popcorn``, ``hot_dog``, ...), or empty. This is the
    "food_category" half of the ``(food_category, form)`` ban pairs -- kept
    separate from ``category`` because "fruit" is not what makes a grape unsafe.
``allowed_forms`` / ``default_form``
    Which preparation forms this food can physically take. The optimiser searches
    over these, which is what lets it repair a hazard by re-forming a food
    instead of removing it.
``tags``
    Medication- and diet-interaction markers (``grapefruit``, ``high_tyramine``,
    ``leafy_green_vitk``, ``alcohol``, ...) used by the age/medication rules.

Run it with::

    python -m foodsense.data.build_food_db            # download if needed, then build
    python -m foodsense.data.build_food_db --stats    # rebuild and print the full report

Sources (both public domain, U.S. Government works):
    https://fdc.nal.usda.gov/download-datasets.html
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sqlite3
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from foodsense import FOOD_DB_PARQUET, FOOD_DB_SQLITE, PROCESSED_DIR, RAW_DIR, SEED
from foodsense.schemas import NUTRIENTS, Form

FDC_DIR = RAW_DIR / "fdc"

#: Release bundles we consume. Pinned by filename so a rebuild is reproducible;
#: update deliberately, not implicitly.
SOURCES: dict[str, str] = {
    "sr_legacy.zip": (
        "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"
    ),
    "foundation.zip": (
        "https://fdc.nal.usda.gov/fdc-datasets/"
        "FoodData_Central_foundation_food_csv_2025-04-24.zip"
    ),
}

# ---------------------------------------------------------------------------
# Nutrient mapping
# ---------------------------------------------------------------------------

#: FoodSense nutrient name -> the FDC ``nutrient.id`` values that can supply it,
#: in preference order. Several nutrients have more than one FDC representation
#: (Energy has three; sugars two), and which one is populated differs between
#: SR Legacy and Foundation, so we take the first that is present per food.
FDC_NUTRIENT_IDS: dict[str, tuple[int, ...]] = {
    "energy_kcal": (1008, 2047, 2048),
    "protein_g": (1003,),
    "carbohydrate_g": (1005, 1050),
    "sugars_g": (2000, 1063, 1235),
    "added_sugars_g": (1235,),
    "fiber_g": (1079, 2033),
    "fat_g": (1004,),
    "saturated_fat_g": (1258,),
    "trans_fat_g": (1257,),
    "monounsaturated_fat_g": (1292,),
    "polyunsaturated_fat_g": (1293,),
    "cholesterol_mg": (1253,),
    "sodium_mg": (1093,),
    "potassium_mg": (1092,),
    "calcium_mg": (1087,),
    "iron_mg": (1089,),
    "magnesium_mg": (1090,),
    "zinc_mg": (1095,),
    "phosphorus_mg": (1091,),
    "copper_mg": (1098,),
    "selenium_ug": (1103,),
    "vitamin_a_rae_ug": (1106,),
    "vitamin_c_mg": (1162,),
    "vitamin_d_ug": (1114,),
    "vitamin_e_mg": (1109,),
    "vitamin_k_ug": (1185,),
    "thiamin_mg": (1165,),
    "riboflavin_mg": (1166,),
    "niacin_mg": (1167,),
    "vitamin_b6_mg": (1175,),
    "folate_dfe_ug": (1190,),
    "vitamin_b12_ug": (1178,),
    "water_g": (1051,),
}

# ``added_sugars_g`` shares FDC id 1235 with the sugars fallback chain. Resolving
# sugars first and added sugars second would double-count, so added sugars are
# taken only from 1235 and sugars prefer 2000/1063 -- see ``_pivot_nutrients``.

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

#: FDC food-category description -> FoodSense category. Anything not listed is
#: dropped (currently only "American Indian/Alaska Native Foods", which is
#: regionally specific and outside the everyday-meal scope of the demo).
CATEGORY_MAP: dict[str, str] = {
    "Dairy and Egg Products": "dairy",
    "Vegetables and Vegetable Products": "vegetable",
    "Fruits and Fruit Juices": "fruit",
    "Beef Products": "meat",
    "Pork Products": "meat",
    "Lamb, Veal, and Game Products": "meat",
    "Poultry Products": "poultry",
    "Sausages and Luncheon Meats": "processed_meat",
    "Finfish and Shellfish Products": "fish",
    "Legumes and Legume Products": "legume",
    "Nut and Seed Products": "nut_seed",
    "Cereal Grains and Pasta": "grain",
    "Baked Products": "baked",
    "Breakfast Cereals": "cereal",
    "Fats and Oils": "fat_oil",
    "Soups, Sauces, and Gravies": "soup_sauce",
    "Sweets": "sweets",
    "Snacks": "snack",
    "Beverages": "beverage",
    "Fast Foods": "prepared",
    "Restaurant Foods": "prepared",
    "Meals, Entrees, and Side Dishes": "prepared",
    "Spices and Herbs": "herb_spice",
    "Baby Foods": "baby_food",
}

#: Per-category ceiling on SR Legacy rows, applied after ranking by the
#: "everyday-ness" score below. SR Legacy carries 954 beef entries that differ
#: only in trim and grade; a recommender needs a handful, not all of them.
CATEGORY_CAPS: dict[str, int] = {
    "vegetable": 290,
    "fruit": 210,
    "meat": 230,
    "poultry": 110,
    "processed_meat": 70,
    "fish": 140,
    "dairy": 170,
    "legume": 120,
    "grain": 120,
    "baked": 160,
    "cereal": 70,
    "nut_seed": 95,
    "fat_oil": 55,
    "soup_sauce": 135,
    "sweets": 110,
    "snack": 95,
    "beverage": 110,
    "prepared": 180,
    "herb_spice": 45,
    "baby_food": 75,
}

#: Descriptions matching these are kept regardless of the per-category cap.
#: They are the foods the three demo scenarios and the safety rules need to
#: exist, so curation can never silently delete the thing under test.
MUST_INCLUDE: tuple[str, ...] = (
    r"^Grapes?, ",
    r"^Peanuts, ",
    r"^Peanut butter, ",
    r"^Rice, white, .*(?<!un)cooked",
    r"^Rice, brown, .*(?<!un)cooked",
    r"^Yogurt, plain",
    r"^Carrots, (raw|cooked|baby)",
    r"^Chicken, broilers or fryers, breast, meat only",
    r"^Chicken, ground.*(?<!un)cooked",
    r"^Soup, .*canned",
    r"^Crackers, (saltines|whole.wheat|standard)",
    r"^Soup, .*(broth|bouillon).*(canned|ready.to.serve|prepared with water)",
    r"^Soup, stock, \w+, home-prepared",
    r"^Fish broth",
    r"^Beef, ground, .*(?<!un)cooked",
    r"^Potatoes, french fried",
    r"^Snacks, popcorn",
    r"^Candies, marshmallows",
    r"^Candies, hard",
    r"^Frankfurter, ",
    r"^Grapefruit, raw",
    r"^Grapefruit juice",
    r"^Spinach, raw",
    r"^Kale, raw",
    r"^Cheese, cheddar",
    r"^Egg, whole, cooked",
    r"^Bananas, raw",
    r"^Apples, raw",
    r"^Tomatoes, red, ripe, raw",
    r"^Bread, whole.wheat",
    r"^Milk, whole",
    r"^Lentils, mature seeds, cooked",
    r"^Beans, .*mature seeds, cooked",
    r"^Oats$|^Cereals, oats",
    r"^Salmon, .*(?<!un)cooked",
    r"^Honey$|^Honey, ",
    r"^Chewing gum",
)

#: Descriptions matching these are dropped outright: alternate presentations of a
#: food we already keep, or entries that are not a food a person eats as served.
EXCLUDE: tuple[str, ...] = (
    r"\bunprepared\b",
    r"\bas purchased\b",
    r"\bdry mix\b",
    r"\bfrozen as packaged\b",
    r"prepared-by-recipe",
    r"\bformulated bar\b",
    r"\binfant formula\b",
    r"\bmeal replacement\b",
    r"\bnutritional supplement\b",
    r"\bUSDA Commodity\b",
    r"\bschool lunch\b",
    r"\bimitation\b",
    r"\bfat-free.*imitation\b",
    r"\bgiblets\b",
    r"\bmechanically separated\b",
    r"\bvariety meats\b",
    r"\bcarcass\b",
    r"\bhydrogenated\b.*\bindustrial\b",
)

#: Exclusions that only make sense inside one category. ``raw`` is essential for
#: fruit and vegetables and meaningless for a staple nobody eats uncooked, so the
#: rule cannot be global. ``dry-roasted`` is deliberately spared -- that is a
#: finished food, not an uncooked staple.
CATEGORY_SCOPED_EXCLUDE: dict[str, tuple[str, ...]] = {
    "grain": (r"\b(raw|uncooked|dehydrated)\b|\bdry\b(?![- ]roast)",),
    "cereal": (r"\b(raw|uncooked|dehydrated)\b|\bdry\b(?![- ]roast)",),
    "legume": (r"\braw\b|\buncooked\b|\bdry\b(?![- ]roast)",),
    "soup_sauce": (r"\bdry\b(?![- ]roast)(?!.*prepared with water)|\bgranules\b",),
    "beverage": (r"\bpowder\b|\bdry mix\b|\bconcentrate, undiluted\b",),
    # Raw muscle food is not a meal as served, and SR Legacy carries a raw row for
    # almost every cooked one. Dropping them halves the meat noise at no cost.
    "meat": (r"\b(raw|uncooked)\b",),
    "poultry": (r"\b(raw|uncooked)\b",),
    "fish": (r"\b(raw|uncooked)\b",),
    "processed_meat": (r"\b(raw|uncooked|unheated)\b",),
}

# ---------------------------------------------------------------------------
# Choking-hazard classes  (the "food_category" half of a (category, form) ban)
# ---------------------------------------------------------------------------

#: ``hazard_class`` -> (description regex, allowed FoodSense categories or None).
#: Order matters: the first match wins, so specific classes come before general
#: ones (nut butter before nut).
HAZARD_RULES: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("popcorn", r"\bpopcorn\b", None),
    ("marshmallow", r"\bmarshmallow", None),
    ("gum", r"\bchewing gum\b|\bgum, chewing\b", None),
    ("hard_candy", r"candies, hard|butterscotch|lollipop|\bhard candy\b|candies, .*brittle", None),
    ("nut_butter", r"\b(peanut|almond|cashew|nut|sunflower seed) butter\b", None),
    ("hot_dog", r"\bfrankfurter\b|\bhot dog\b", ("processed_meat", "prepared", "meat")),
    ("grape", r"^grapes?,|\bgrapes, (raw|american|red or green)\b", ("fruit",)),
    ("cherry_tomato", r"cherry tomato|tomatoes, .*cherry", ("vegetable",)),
    (
        "seed",
        r"\bseeds?,|\b(sunflower|pumpkin|sesame|flax|chia) seed",
        ("nut_seed",),
    ),
    (
        # ``legume`` is in scope because FDC files peanuts -- the canonical toddler
        # choking hazard -- under "Legumes and Legume Products", not under nuts.
        "nut",
        r"\b(peanuts?|almonds?|cashews?|walnuts?|pecans?|pistachios?|hazelnuts?|"
        r"macadamia|brazilnuts?|mixed nuts)\b",
        ("nut_seed", "snack", "legume"),
    ),
    (
        "hard_raw_vegetable",
        r"\b(carrots?|celery|broccoli|cauliflower|turnips?|radishes?|jicama|parsnips?|"
        r"rutabagas?|kohlrabi|beets?)\b.*\braw\b",
        ("vegetable",),
    ),
    (
        "meat_chunk",
        r"",  # assigned by category, not by keyword -- see _assign_hazard_class
        ("meat", "poultry", "fish"),
    ),
)

# ---------------------------------------------------------------------------
# Interaction / diet tags
# ---------------------------------------------------------------------------

#: tag -> description regex. Nutrient-derived tags are added separately in
#: :func:`_assign_tags` because a threshold on real data beats a keyword list.
TAG_PATTERNS: dict[str, str] = {
    "grapefruit": r"\bgrapefruit\b",
    "alcohol": (
        r"\b(beer|wine|liqueur|vodka|whiskey|whisky|rum|gin|tequila|brandy|"
        r"alcoholic beverage|cordial|sake)\b"
    ),
    "caffeine": r"\b(coffee|espresso|tea, (black|green|instant)|cola|energy drink|cocoa)\b",
    "honey": r"\bhoney\b(?!dew)",
    "aged_cheese": (
        r"cheese, (blue|brick|brie|camembert|cheddar|colby|edam|feta|gouda|gruyere|"
        r"limburger|monterey|muenster|parmesan|provolone|roquefort|swiss|romano)"
    ),
    "cured_meat": (
        r"\b(salami|pepperoni|prosciutto|bologna|pastrami|corned beef|bacon|"
        r"ham, (cured|sliced)|jerky|chorizo|sausage, dry|summer sausage|"
        r"luncheon meat|frankfurter)\b"
    ),
    "high_tyramine": (
        r"\b(sauerkraut|soy sauce|miso|tempeh|fermented|fava bean|broad bean|"
        r"yeast extract|marmite|vegemite|anchov|dried fish|shrimp paste)\b"
    ),
    # A *name* claim, not a measurement. USDA lists "Crackers, saltines, fat-free,
    # low-sodium" at 849 mg/100 g, so a food can legitimately carry both this tag
    # and ``high_sodium`` -- which is exactly the sort of gap Stage 4 exists to catch.
    "low_sodium_variant": (
        r"\b(low[ -]sodium|lower sodium|low salt|no salt added|without salt|"
        r"unsalted|salt[ -]free|reduced sodium)\b"
    ),
    "added_sugar_source": r"\b(candies|syrup|sweetened|frosting|sugars, granulated)\b",
}

#: Sweetened drinks, tagged only within the beverage category. They are a major
#: source of added sugar and the keyword list above misses almost all of them:
#: only 4 of 110 curated beverages carried ``added_sugar_source``, so colas,
#: sodas and lemonades were invisible to the added-sugar rule.
SWEETENED_BEVERAGE_RE = (
    r"\b(cola|soda|soft drink|lemonade|punch|energy drink|sport drink|"
    r"fruit drink|nectar|sweetened|malted|milkshake|shake|smoothie|"
    r"chocolate[- ]flavored|horchata|eggnog)\b"
)

#: ...but not when the name says otherwise. A diet cola is not a sugar source.
UNSWEETENED_RE = r"\b(unsweetened|sugar[- ]free|diet|low calorie|no sugar|zero)\b"

#: Nutrient-threshold tags, per 100 g. Values chosen to match the rules that
#: consume them: the warfarin rule caps vitamin K per meal, so a food only needs
#: the ``leafy_green_vitk`` tag if a normal portion could move that cap.
NUTRIENT_TAGS: tuple[tuple[str, str, float], ...] = (
    ("leafy_green_vitk", "vitamin_k_ug", 50.0),
    ("high_potassium", "potassium_mg", 400.0),
    ("high_sodium", "sodium_mg", 400.0),
)

# ---------------------------------------------------------------------------
# Preparation forms
# ---------------------------------------------------------------------------

#: ``Form.WHOLE`` means "as served, unmodified" -- it is the identity form, not a
#: claim that the food is literally whole. Every food therefore allows it.
FORMS_BY_CATEGORY: dict[str, tuple[Form, ...]] = {
    "fruit": (Form.WHOLE, Form.QUARTERED, Form.SLICED, Form.CHOPPED, Form.MASHED, Form.PUREED),
    "vegetable": (
        Form.WHOLE,
        Form.CHOPPED,
        Form.SLICED,
        Form.SOFT_COOKED,
        Form.MASHED,
        Form.PUREED,
    ),
    "meat": (Form.WHOLE, Form.SLICED, Form.CHOPPED, Form.MINCED, Form.GROUND, Form.SOFT_COOKED),
    "poultry": (Form.WHOLE, Form.SLICED, Form.CHOPPED, Form.MINCED, Form.GROUND, Form.SOFT_COOKED),
    "fish": (Form.WHOLE, Form.CHOPPED, Form.MINCED, Form.MASHED, Form.SOFT_COOKED),
    "processed_meat": (Form.WHOLE, Form.SLICED, Form.SLICED_ROUNDS, Form.CHOPPED, Form.MINCED),
    "dairy": (Form.WHOLE, Form.SLICED, Form.CHOPPED, Form.MASHED),
    "legume": (Form.WHOLE, Form.SOFT_COOKED, Form.MASHED, Form.PUREED),
    "grain": (Form.WHOLE, Form.SOFT_COOKED, Form.MASHED),
    "cereal": (Form.WHOLE, Form.SOFT_COOKED, Form.MASHED),
    "baked": (Form.WHOLE, Form.SLICED, Form.CHOPPED, Form.THIN_SPREAD),
    "nut_seed": (Form.WHOLE, Form.CHOPPED, Form.GROUND, Form.THIN_SPREAD),
    "fat_oil": (Form.WHOLE, Form.THIN_SPREAD),
    "soup_sauce": (Form.WHOLE, Form.PUREED),
    "sweets": (Form.WHOLE, Form.CHOPPED),
    "snack": (Form.WHOLE, Form.CHOPPED, Form.GROUND),
    "beverage": (Form.WHOLE,),
    "prepared": (Form.WHOLE, Form.SLICED, Form.CHOPPED, Form.MINCED, Form.MASHED),
    "herb_spice": (Form.WHOLE, Form.GROUND, Form.MINCED),
    "baby_food": (Form.PUREED, Form.MASHED, Form.WHOLE),
}

#: Hazard classes override the category default, because the hazard is precisely
#: about which forms are physically available.
FORMS_BY_HAZARD: dict[str, tuple[Form, ...]] = {
    # No SLICED: a grape cut into rounds is still an airway plug, and the AAP
    # guidance is specifically to quarter lengthwise. Leaving the form out of the
    # search space is stronger than banning it -- the optimiser cannot pick it.
    "grape": (Form.WHOLE, Form.QUARTERED, Form.MASHED, Form.PUREED),
    "cherry_tomato": (Form.WHOLE, Form.QUARTERED, Form.PUREED),
    "nut": (Form.WHOLE, Form.CHOPPED, Form.GROUND),
    "seed": (Form.WHOLE, Form.GROUND),
    "nut_butter": (Form.SPOONFUL, Form.THIN_SPREAD),
    "hot_dog": (Form.WHOLE, Form.SLICED_ROUNDS, Form.SLICED, Form.MINCED),
    "hard_raw_vegetable": (
        Form.WHOLE,
        Form.CHOPPED,
        Form.SLICED,
        Form.SOFT_COOKED,
        Form.MASHED,
        Form.PUREED,
    ),
    # No preparation makes these safe for a small child; the rules must remove
    # or substitute rather than re-form, so the search space offers no escape.
    "popcorn": (Form.WHOLE,),
    "marshmallow": (Form.WHOLE,),
    "hard_candy": (Form.WHOLE,),
    "gum": (Form.WHOLE,),
    "meat_chunk": (
        Form.WHOLE,
        Form.SLICED,
        Form.CHOPPED,
        Form.MINCED,
        Form.GROUND,
        Form.SOFT_COOKED,
    ),
}

#: Keyword overrides applied between the hazard map and the category default.
#: A category is too coarse for physical form: "dairy" covers both cheddar (which
#: slices) and yogurt (which does not), and slicing a soup is meaningless.
FORMS_BY_KEYWORD: tuple[tuple[str, tuple[Form, ...]], ...] = (
    (r"^Cheese,|^Cheese food|^Cheese product", (Form.WHOLE, Form.SLICED, Form.CHOPPED,
                                                Form.GROUND, Form.MASHED)),
    (r"^Egg, |^Eggs, ", (Form.WHOLE, Form.CHOPPED, Form.MASHED, Form.SOFT_COOKED)),
    (r"^Yogurt|^Kefir|^Sour cream|^Cream, |^Milk|^Buttermilk", (Form.WHOLE, Form.PUREED)),
    (r"^Soup, |^Fish broth|\bbroth\b|\bbouillon\b|^Gravy|^Sauce", (Form.WHOLE, Form.PUREED)),
    (r"^Oil, |^Butter, |^Margarine|^Shortening|^Salad dressing|^Fat, ",
     (Form.WHOLE, Form.THIN_SPREAD)),
    (r"^Beverages|^Water|^Alcoholic|juice\b|^Tea,|^Coffee", (Form.WHOLE,)),
    (r"^Syrups|^Honey|^Jams|^Jellies|^Molasses", (Form.WHOLE, Form.THIN_SPREAD)),
)

#: Foods whose description already states a preparation start in that form.
DEFAULT_FORM_HINTS: tuple[tuple[str, Form], ...] = (
    (r"\bpureed\b|\bstrained\b|, puree", Form.PUREED),
    (r"\bmashed\b", Form.MASHED),
    (r"\bground\b", Form.GROUND),
    (r"\bminced\b", Form.MINCED),
    (r"\bsliced\b", Form.SLICED),
    (r"\bchopped\b", Form.CHOPPED),
    (r"\bbutter\b.*\b(peanut|almond|cashew|nut)\b|\b(peanut|almond|cashew|nut) butter\b",
     Form.SPOONFUL),
    (r"\bcooked\b|\bboiled\b|\bbraised\b|\bsteamed\b", Form.SOFT_COOKED),
)


# ---------------------------------------------------------------------------
# Download / IO
# ---------------------------------------------------------------------------


@dataclass
class BuildReport:
    """What the build actually did -- printed, and stored in the database."""

    source_files: dict[str, str]
    n_sr_legacy_input: int
    n_foundation_input: int
    n_after_category_map: int
    n_after_exclude: int
    n_after_nutrient_filter: int
    n_must_include_kept: int
    n_final: int
    category_counts: dict[str, int]
    hazard_counts: dict[str, int]
    tag_counts: dict[str, int]
    built_at: str


def download_sources(force: bool = False) -> dict[str, Path]:
    """Fetch the FDC bundles into ``data/raw/fdc/``. Cached after the first run."""
    FDC_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in SOURCES.items():
        dest = FDC_DIR / name
        if dest.exists() and not force:
            print(f"  cached   {name} ({dest.stat().st_size / 1e6:.1f} MB)")
            paths[name] = dest
            continue
        print(f"  download {name} <- {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "foodsense/0.1"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                f.write(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SystemExit(
                f"\nCould not download {name}: {exc}\n"
                f"Download it manually from {url} and place it at {dest}, then re-run.\n"
                f"See data/README.md."
            ) from exc
        print(f"           {dest.stat().st_size / 1e6:.1f} MB")
        paths[name] = dest
    return paths


def _read_zip_csv(zip_path: Path, member: str) -> pd.DataFrame:
    """Read one CSV out of an FDC bundle without unpacking the whole archive."""
    with zipfile.ZipFile(zip_path) as z:
        root = z.infolist()[0].filename
        with z.open(root + member) as f:
            return pd.read_csv(io.BytesIO(f.read()), dtype=str, low_memory=False)


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


def _everydayness(description: str) -> float:
    """Rank how much a description looks like an everyday food. Lower is better.

    SR Legacy descriptions get longer and more comma-separated the more specific
    the cut, grade or trim is: "Beef, chuck, arm pot roast, separable lean only,
    trimmed to 1/8 inch fat, all grades, cooked, braised". Counting qualifiers is
    a crude but effective proxy for "would a person say this out loud".
    """
    commas = description.count(",")
    parens = description.count("(")
    length = len(description)
    # Brand names shout in SR Legacy ("OCEAN SPRAY", "KELLOGG'S").
    shouty = sum(1 for tok in description.split() if len(tok) > 2 and tok.isupper())
    return commas * 2.0 + parens * 1.5 + shouty * 3.0 + length / 40.0


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _pivot_nutrients(food_nutrient: pd.DataFrame, keep_ids: set[int]) -> pd.DataFrame:
    """Turn the long food_nutrient table into one column per FoodSense nutrient."""
    fn = food_nutrient[["fdc_id", "nutrient_id", "amount"]].copy()
    fn["nutrient_id"] = pd.to_numeric(fn["nutrient_id"], errors="coerce")
    fn["amount"] = pd.to_numeric(fn["amount"], errors="coerce")
    fn = fn[fn["nutrient_id"].isin(keep_ids) & fn["amount"].notna()]

    wide = fn.pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first")

    out = pd.DataFrame(index=wide.index)
    for nutrient in NUTRIENTS:
        ids = FDC_NUTRIENT_IDS[nutrient]
        # Added sugars must come only from its own id; letting it fall back to the
        # sugars chain would report total sugars as added sugars.
        if nutrient == "added_sugars_g":
            ids = (1235,)
        series = pd.Series(pd.NA, index=wide.index, dtype="Float64")
        for fdc_nutrient_id in ids:
            if fdc_nutrient_id in wide.columns:
                series = series.fillna(wide[fdc_nutrient_id].astype("Float64"))
        out[nutrient] = series
    return out.reset_index()


def _assign_hazard_class(description: str, category: str) -> str:
    """First matching hazard class wins; specific classes are listed first."""
    desc = description.lower()
    for hazard, pattern, categories in HAZARD_RULES:
        if categories is not None and category not in categories:
            continue
        if hazard == "meat_chunk":
            # Not keyword-driven: any solid muscle food that is not already
            # ground or minced presents as a chunk on the plate.
            if re.search(r"\bground\b|\bminced\b|\bpaste\b|\bpuree", desc):
                return ""
            return "meat_chunk"
        if pattern and re.search(pattern, desc, flags=re.IGNORECASE):
            # "peanut butter" must not be caught by the nut rule.
            if hazard in {"nut", "seed"} and re.search(r"\bbutter\b|\bflour\b|\boil\b|\bmilk\b",
                                                       desc):
                return ""
            return hazard
    return ""


def _assign_tags(description: str, category: str, hazard: str, row: pd.Series) -> list[str]:
    tags: list[str] = []
    for tag, pattern in TAG_PATTERNS.items():
        if re.search(pattern, description, flags=re.IGNORECASE):
            tags.append(tag)
    for tag, nutrient, threshold in NUTRIENT_TAGS:
        value = row.get(nutrient)
        if pd.notna(value) and float(value) >= threshold:
            tags.append(tag)
    if (
        category == "beverage"
        and re.search(SWEETENED_BEVERAGE_RE, description, flags=re.IGNORECASE)
        and not re.search(UNSWEETENED_RE, description, flags=re.IGNORECASE)
    ):
        tags.append("sweetened_beverage")

    # Hazard-derived tags named explicitly in the design brief.
    if hazard == "nut":
        tags.append("whole_nut")
    if hazard == "hard_raw_vegetable":
        tags.append("raw_hard_veg")
    if hazard == "nut_butter":
        tags.append("nut_butter")
    # Aged cheese and cured meat are the tyramine sources the MAOI rule targets.
    if "aged_cheese" in tags or "cured_meat" in tags:
        tags.append("high_tyramine")
    if category == "beverage" and "alcohol" in tags:
        tags.append("alcohol")
    return sorted(set(tags))


def _assign_forms(category: str, hazard: str, description: str) -> tuple[list[str], str]:
    """Resolve the allowed forms: hazard map first, then keyword, then category.

    Hazard wins outright because those form lists are the safety contract -- the
    optimiser must not be handed an escape hatch the rules do not know about.
    """
    forms = FORMS_BY_HAZARD.get(hazard)
    if forms is None:
        for pattern, keyword_forms in FORMS_BY_KEYWORD:
            if re.search(pattern, description, flags=re.IGNORECASE):
                forms = keyword_forms
                break
    if forms is None:
        forms = FORMS_BY_CATEGORY.get(category, (Form.WHOLE,))
    allowed = [f.value for f in forms]

    default = allowed[0]
    for pattern, form in DEFAULT_FORM_HINTS:
        if re.search(pattern, description, flags=re.IGNORECASE) and form.value in allowed:
            default = form.value
            break
    return allowed, default


def curate(report_only: bool = False) -> tuple[pd.DataFrame, BuildReport]:
    """Load, filter and annotate the FDC releases into the curated table."""
    paths = download_sources()

    print("\nReading FDC tables...")
    categories = _read_zip_csv(paths["sr_legacy.zip"], "food_category.csv")
    cat_lookup = dict(zip(categories["id"], categories["description"], strict=True))

    food_sr = _read_zip_csv(paths["sr_legacy.zip"], "food.csv")
    food_sr = food_sr[food_sr["data_type"] == "sr_legacy_food"].copy()
    food_fd = _read_zip_csv(paths["foundation.zip"], "food.csv")
    food_fd = food_fd[food_fd["data_type"] == "foundation_food"].copy()
    n_sr_in, n_fd_in = len(food_sr), len(food_fd)
    print(f"  sr_legacy_food   {n_sr_in:6d} rows")
    print(f"  foundation_food  {n_fd_in:6d} rows")

    foods = pd.concat([food_sr, food_fd], ignore_index=True)
    foods["fdc_category"] = foods["food_category_id"].map(cat_lookup)
    foods["category"] = foods["fdc_category"].map(CATEGORY_MAP)
    foods = foods[foods["category"].notna()].copy()
    n_after_cat = len(foods)
    print(f"  after category map        {n_after_cat:6d}")

    foods["must_include"] = foods["description"].apply(lambda d: _matches_any(d, MUST_INCLUDE))
    excluded = foods["description"].apply(lambda d: _matches_any(d, EXCLUDE))
    scoped = pd.Series(False, index=foods.index)
    for category, patterns in CATEGORY_SCOPED_EXCLUDE.items():
        in_category = foods["category"] == category
        scoped |= in_category & foods["description"].apply(
            lambda d, p=patterns: _matches_any(d, p)
        )
    foods = foods[~(excluded | scoped) | foods["must_include"]].copy()
    n_after_exclude = len(foods)
    print(f"  after exclusion patterns  {n_after_exclude:6d}")

    print("\nReading nutrients (this is the slow part)...")
    keep_ids = {i for ids in FDC_NUTRIENT_IDS.values() for i in ids}
    nut_frames = [
        _pivot_nutrients(_read_zip_csv(paths[z], "food_nutrient.csv"), keep_ids)
        for z in ("sr_legacy.zip", "foundation.zip")
    ]
    nutrients = pd.concat(nut_frames, ignore_index=True).drop_duplicates("fdc_id", keep="first")
    print(f"  nutrient rows for {len(nutrients)} foods")

    foods = foods.merge(nutrients, on="fdc_id", how="inner")

    # A food with no energy value cannot participate in any goal computation.
    foods = foods[foods["energy_kcal"].notna()].copy()
    n_after_nutrients = len(foods)
    print(f"  with energy value         {n_after_nutrients:6d}")

    # --- rank and cap per category ----------------------------------------
    foods["score"] = foods["description"].apply(_everydayness)
    # Foundation foods are the highest-quality, most generic entries in FDC;
    # give them a head start so they survive capping.
    foods.loc[foods["data_type"] == "foundation_food", "score"] -= 25.0
    foods.loc[foods["must_include"], "score"] -= 1000.0

    foods = foods.sort_values(["category", "score", "description"], kind="stable")
    # SR Legacy and Foundation both describe e.g. "Milk, whole, 3.25% milkfat, with
    # added vitamin D". Three fdc_ids for one food would let the optimiser "add" a
    # food it already has, so keep the best-scoring row per description.
    n_before_dedup = len(foods)
    foods = foods.drop_duplicates(subset="description", keep="first")
    print(f"  after name de-duplication {len(foods):6d}  (-{n_before_dedup - len(foods)})")

    kept = []
    for category, group in foods.groupby("category", sort=False):
        cap = CATEGORY_CAPS.get(category, 60)
        forced = group[group["must_include"]]
        rest = group[~group["must_include"]].head(max(cap - len(forced), 0))
        kept.append(pd.concat([forced, rest]))
    db = pd.concat(kept, ignore_index=True)
    n_must = int(db["must_include"].sum())
    print(f"  after per-category caps   {len(db):6d}  ({n_must} force-included)")

    # --- annotate ----------------------------------------------------------
    print("\nAnnotating hazard classes, tags and forms...")
    db["hazard_class"] = [
        _assign_hazard_class(d, c)
        for d, c in zip(db["description"], db["category"], strict=True)
    ]
    db["tags"] = [
        json.dumps(_assign_tags(row["description"], row["category"], row["hazard_class"], row))
        for _, row in db.iterrows()
    ]
    forms = [
        _assign_forms(row["category"], row["hazard_class"], row["description"])
        for _, row in db.iterrows()
    ]
    db["allowed_forms"] = [json.dumps(a) for a, _ in forms]
    db["default_form"] = [d for _, d in forms]

    # --- final shape -------------------------------------------------------
    for nutrient in NUTRIENTS:
        db[nutrient] = pd.to_numeric(db[nutrient], errors="coerce").fillna(0.0).astype(float)

    db = db.rename(columns={"description": "name"})
    db["fdc_id"] = db["fdc_id"].astype(str)
    columns = [
        "fdc_id",
        "name",
        "category",
        "fdc_category",
        "data_type",
        "hazard_class",
        "default_form",
        "allowed_forms",
        "tags",
        *NUTRIENTS,
    ]
    db = db[columns].sort_values("name", kind="stable").reset_index(drop=True)

    hazard_counts = Counter(h for h in db["hazard_class"] if h)
    tag_counts = Counter(t for tags in db["tags"] for t in json.loads(tags))

    report = BuildReport(
        source_files={k: v for k, v in SOURCES.items()},
        n_sr_legacy_input=n_sr_in,
        n_foundation_input=n_fd_in,
        n_after_category_map=n_after_cat,
        n_after_exclude=n_after_exclude,
        n_after_nutrient_filter=n_after_nutrients,
        n_must_include_kept=n_must,
        n_final=len(db),
        category_counts=dict(Counter(db["category"]).most_common()),
        hazard_counts=dict(hazard_counts.most_common()),
        tag_counts=dict(tag_counts.most_common()),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return db, report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(db: pd.DataFrame, report: BuildReport) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if FOOD_DB_SQLITE.exists():
        FOOD_DB_SQLITE.unlink()
    with sqlite3.connect(FOOD_DB_SQLITE) as conn:
        db.to_sql("foods", conn, index=False)
        conn.execute("CREATE UNIQUE INDEX idx_foods_fdc_id ON foods(fdc_id)")
        conn.execute("CREATE INDEX idx_foods_category ON foods(category)")
        conn.execute("CREATE INDEX idx_foods_hazard ON foods(hazard_class)")
        conn.execute("CREATE TABLE build_info (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO build_info (key, value) VALUES (?, ?)",
            [
                ("built_at", report.built_at),
                ("seed", str(SEED)),
                ("n_foods", str(report.n_final)),
                ("sources", json.dumps(report.source_files)),
                ("nutrients", json.dumps(list(NUTRIENTS))),
            ],
        )
        conn.commit()

    db.to_parquet(FOOD_DB_PARQUET, index=False, compression="zstd")

    print("\nWrote:")
    print(f"  {FOOD_DB_SQLITE}   {FOOD_DB_SQLITE.stat().st_size / 1e6:.2f} MB")
    print(f"  {FOOD_DB_PARQUET}  {FOOD_DB_PARQUET.stat().st_size / 1e6:.2f} MB")


def print_report(db: pd.DataFrame, report: BuildReport) -> None:
    print("\n" + "=" * 72)
    print("FOOD DATABASE BUILD REPORT")
    print("=" * 72)
    print(f"Built at        : {report.built_at}")
    print(f"SR Legacy input : {report.n_sr_legacy_input}")
    print(f"Foundation input: {report.n_foundation_input}")
    print(f"After category  : {report.n_after_category_map}")
    print(f"After exclusion : {report.n_after_exclude}")
    print(f"With nutrients  : {report.n_after_nutrient_filter}")
    print(f"FINAL FOODS     : {report.n_final}  ({report.n_must_include_kept} force-included)")

    print("\n--- category histogram ---")
    for name, count in report.category_counts.items():
        bar = "#" * max(1, round(count / 8))
        print(f"  {name:<16} {count:>5}  {bar}")

    print("\n--- choking-hazard classes ---")
    for name, count in report.hazard_counts.items():
        print(f"  {name:<20} {count:>5}")

    print("\n--- interaction / diet tags ---")
    for name, count in report.tag_counts.items():
        print(f"  {name:<20} {count:>5}")

    print("\n--- nutrient coverage (share of foods with a non-zero value) ---")
    for nutrient in NUTRIENTS:
        share = float((db[nutrient] > 0).mean())
        print(f"  {nutrient:<26} {share * 100:5.1f}%")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true", help="print the full build report")
    parser.add_argument(
        "--force-download", action="store_true", help="re-download the FDC bundles"
    )
    args = parser.parse_args(argv)

    if args.force_download:
        download_sources(force=True)

    db, report = curate()
    write_outputs(db, report)
    if args.stats:
        print_report(db, report)
    else:
        print(f"\n{report.n_final} foods across {len(report.category_counts)} categories.")
        print("Run with --stats for the full report.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
