"""Meal + profile -> Stage-1 feature vector.

The feature set is deliberately narrow: the nutrients and derived quantities the
guideline rules are actually written against, plus a description of *who* the
meal is for. That is enough to approximate the rule engine's numeric score, which
is all Stage 1 is asked to do.

Nothing here encodes a choking hazard or a medication exclusion, and that is a
design decision rather than an oversight -- see :mod:`foodsense.constraints.engine`.
Those are discrete facts about ``(hazard_class, form)`` pairs and about food tags;
a model given nutrient features could only fit them as noise, so they are enforced
explicitly in the Stage-2 objective and re-scanned in Stage 4.

This module is the hottest path in the project: differential evolution evaluates
thousands of candidate meals per run, and each evaluation starts here. Feature
construction therefore goes through the food database's vectorised nutrient
lookup rather than building a Pydantic model per item.
"""

from __future__ import annotations

import numpy as np

from foodsense.constraints.goals import (
    ADDED_SUGAR_PROXY_CATEGORIES,
    ADDED_SUGAR_PROXY_TAGS,
    DEFAULT_GI,
    GI_BY_CATEGORY,
)
from foodsense.data.fdc import FoodDB
from foodsense.schemas import NUTRIENTS, AgeGroup, Goal, HealthFlag, Meal, MealItem, UserProfile

__all__ = ["FEATURE_NUTRIENTS", "feature_names", "meal_features", "n_features"]

#: The nutrients the rules care about. Carrying all 33 would add columns that no
#: guideline references and that the model would have to learn to ignore.
FEATURE_NUTRIENTS: tuple[str, ...] = (
    "energy_kcal",
    "protein_g",
    "carbohydrate_g",
    "sugars_g",
    "fiber_g",
    "fat_g",
    "saturated_fat_g",
    "cholesterol_mg",
    "sodium_mg",
    "potassium_mg",
    "calcium_mg",
    "iron_mg",
    "zinc_mg",
    "vitamin_c_mg",
    "vitamin_d_ug",
    "vitamin_k_ug",
    "vitamin_b12_ug",
)

_DERIVED = (
    "glycemic_load",
    "added_sugars_g",
    "item_count",
    "total_mass_g",
    "max_item_mass_share",
    "share_protein",
    "share_carbohydrate",
    "share_fat",
)

_AGE_GROUPS = tuple(AgeGroup)
_GOALS = tuple(Goal)
_HEALTH_FLAGS = tuple(HealthFlag)

_ATWATER = (4.0, 4.0, 9.0)


def feature_names() -> list[str]:
    """Column names, in the exact order :func:`meal_features` produces them."""
    return [
        *FEATURE_NUTRIENTS,
        *_DERIVED,
        *(f"age_{a.value}" for a in _AGE_GROUPS),
        *(f"goal_{g.value}" for g in _GOALS),
        *(f"flag_{f.value}" for f in _HEALTH_FLAGS),
    ]


def n_features() -> int:
    return len(feature_names())


def meal_features(meal: Meal | list[MealItem], profile: UserProfile, db: FoodDB) -> np.ndarray:
    """One feature row for a meal and the person eating it."""
    items = meal.items if isinstance(meal, Meal) else meal

    totals = db.nutrient_totals(items)  # vectorised, canonical NUTRIENTS order
    nutrient_index = db.nutrient_index

    row = np.zeros(n_features(), dtype=np.float64)
    cursor = 0
    for nutrient in FEATURE_NUTRIENTS:
        row[cursor] = totals[nutrient_index[nutrient]]
        cursor += 1

    glycemic_load = 0.0
    added_sugars = 0.0
    masses: list[float] = []
    for item in items:
        record = db.find(item.food_id)
        if record is None:
            continue
        masses.append(item.quantity_g)
        factor = item.quantity_g / 100.0
        per_100g = record.nutrients_per_100g
        available_carb = max(per_100g.carbohydrate_g - per_100g.fiber_g, 0.0) * factor
        glycemic_load += GI_BY_CATEGORY.get(record.category, DEFAULT_GI) * available_carb / 100.0
        if per_100g.added_sugars_g > 0:
            added_sugars += per_100g.added_sugars_g * factor
        elif record.category in ADDED_SUGAR_PROXY_CATEGORIES or bool(
            ADDED_SUGAR_PROXY_TAGS & record.tags
        ):
            added_sugars += per_100g.sugars_g * factor

    total_mass = sum(masses)
    energy = totals[nutrient_index["energy_kcal"]]

    row[cursor] = glycemic_load
    row[cursor + 1] = added_sugars
    row[cursor + 2] = float(len(masses))
    row[cursor + 3] = total_mass
    # How concentrated the meal is in one food. A 400 g plate that is 90% chips
    # scores differently from one spread across five foods, and no nutrient total
    # captures that.
    row[cursor + 4] = (max(masses) / total_mass) if total_mass > 0 else 0.0

    if energy > 0:
        for offset, nutrient in enumerate(("protein_g", "carbohydrate_g", "fat_g")):
            grams = totals[nutrient_index[nutrient]]
            row[cursor + 5 + offset] = grams * _ATWATER[offset] / energy
    cursor += len(_DERIVED)

    for age_group in _AGE_GROUPS:
        row[cursor] = float(profile.age_group == age_group)
        cursor += 1
    for goal in _GOALS:
        row[cursor] = float(profile.goal == goal)
        cursor += 1
    active = set(profile.health_flags)
    for flag in _HEALTH_FLAGS:
        row[cursor] = float(flag in active)
        cursor += 1

    return row


# Guard against the feature list and the nutrient contract drifting apart.
assert set(FEATURE_NUTRIENTS) <= set(NUTRIENTS)
