"""Goal thresholds, the smooth-margin machinery, and derived meal metrics.

Three things live here, in dependency order:

1. :class:`Threshold` / :class:`Rule` and :func:`satisfaction` -- how a numeric
   guideline becomes a score in [0, 1]. Every rule in the system, goal or age or
   medication, is expressed with these, which is what lets one engine evaluate
   all of them.
2. :class:`GoalConfig` -- loads ``configs/goals/*.yaml``.
3. Derived metrics a nutrient vector does not carry directly: estimated glycemic
   load, the added-sugar proxy, and macronutrient energy shares.

**Why the margins are soft.** A guideline is a step: 550 kcal is fine and 551 is
not. A step function tells an optimiser nothing -- every infeasible meal looks
equally bad, so there is no direction to move in. Replacing each threshold with a
sigmoid centred on it keeps the meaning (exactly 0.5 at the threshold) while
making "620 mg of sodium is better than 900 mg" expressible. That is the surface
the Stage-1 surrogate learns and the Stage-2 optimiser climbs. Validity is still
decided by the threshold itself, never by the softened score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

import yaml

from foodsense import CONFIG_DIR
from foodsense.data.fdc import FoodDB
from foodsense.schemas import Goal, Meal, MealItem, NutrientVector

__all__ = [
    "ADDED_SUGAR_PROXY_CATEGORIES",
    "ADDED_SUGAR_PROXY_TAGS",
    "DEFAULT_GI",
    "GI_BY_CATEGORY",
    "GoalConfig",
    "Rule",
    "Severity",
    "Threshold",
    "energy_shares",
    "estimate_added_sugars",
    "estimate_glycemic_load",
    "load_goal_config",
    "meal_metrics",
    "satisfaction",
]

Severity = Literal["hard", "soft"]

GOALS_DIR = CONFIG_DIR / "goals"

#: Guards against division by zero when a threshold is itself zero.
_MIN_WIDTH = 1e-6

#: Beyond this many sigmoid widths from a threshold, saturate rather than call
#: ``math.exp`` on a large number.
_SATURATION = 40.0


# ---------------------------------------------------------------------------
# Thresholds and rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Threshold:
    """A numeric guideline: a floor, a ceiling, or a band between the two."""

    minimum: float | None = None
    maximum: float | None = None
    weight: float = 1.0

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> Threshold:
        raw = raw or {}
        return cls(
            minimum=_as_float(raw.get("min")),
            maximum=_as_float(raw.get("max")),
            weight=float(raw.get("weight", 1.0)),
        )

    def scaled(self, factor: float, floor_attainment: float = 1.0) -> Threshold:
        """Scale both bounds -- used to turn a daily target into a per-meal one.

        ``floor_attainment`` scales the *minimum* only, and exists because floors
        and ceilings do not divide across meals the same way. Exceeding a sodium
        ceiling in one meal genuinely matters, so ceilings stay proportional. A
        floor does not work that way: micronutrient intake is assessed over a day
        and is heavily skewed between meals, so requiring every meal to deliver
        its full proportional share of calcium or iron would mean rejecting every
        real meal. Measured on 400 Food.com meals, the median single serving
        supplies 0.22 of a proportional calcium floor and 0.29 of an iron one.
        """
        return Threshold(
            minimum=None if self.minimum is None else self.minimum * factor * floor_attainment,
            maximum=None if self.maximum is None else self.maximum * factor,
            weight=self.weight,
        )

    def is_satisfied(self, value: float) -> bool:
        """Hard truth, ignoring the softening. This is what decides validity."""
        if self.minimum is not None and value < self.minimum:
            return False
        return not (self.maximum is not None and value > self.maximum)

    def describe(self) -> str:
        if self.minimum is not None and self.maximum is not None:
            return f"{self.minimum:g}-{self.maximum:g}"
        if self.minimum is not None:
            return f">= {self.minimum:g}"
        if self.maximum is not None:
            return f"<= {self.maximum:g}"
        return "unbounded"


@dataclass(frozen=True, slots=True)
class Rule:
    """One numeric guideline attached to one measurable quantity of a meal."""

    rule_id: str
    quantity: str
    threshold: Threshold
    severity: Severity = "soft"
    message: str = ""
    source: str = ""

    @property
    def weight(self) -> float:
        return self.threshold.weight


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _sigmoid(x: float) -> float:
    if x >= _SATURATION:
        return 1.0
    if x <= -_SATURATION:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def satisfaction(value: float, threshold: Threshold, softness: float) -> float:
    """How well ``value`` satisfies ``threshold``, in [0, 1].

    Exactly 0.5 at a one-sided threshold, so the softening never changes where the
    boundary is -- only how sharply the score falls away from it. ``softness`` is a
    fraction of the threshold, so a 550 kcal ceiling and a 500 mg one are softened
    proportionally rather than by the same absolute amount.
    """
    if threshold.minimum is not None and threshold.maximum is not None:
        # For a band, soften by the width of the band rather than by the size of
        # each bound. Using each bound's own magnitude makes a narrow band
        # unsatisfiable: with a 25-35% fat-share band (the value this goal used to
        # carry), a meal sitting exactly in the middle at 30% scored 0.57, because
        # it is only 1.3 sigmoid-widths above the floor and 0.95 below the ceiling.
        # Scaling by the band width puts the centre near 1.0 and keeps each edge
        # at exactly 0.5.
        width = max((threshold.maximum - threshold.minimum) * softness, _MIN_WIDTH)
        return _sigmoid((value - threshold.minimum) / width) * _sigmoid(
            (threshold.maximum - value) / width
        )

    # One-sided thresholds are softened on a *ratio* scale rather than a linear
    # one: the argument is log(threshold / value) instead of (threshold - value).
    #
    # Both give exactly 0.5 at the threshold and behave almost identically near
    # it, but they differ completely once a rule is badly broken, and that is
    # where it matters. On the linear scale a 500 mg sodium ceiling scores
    # 5.2e-06 at 1413 mg and 3.1e-07 at 1623 mg -- numerically dead, so the
    # optimiser gets no signal that the second meal is worse than the first, and
    # the smooth surface this whole design rests on is flat exactly where it is
    # most needed. On the ratio scale those become 9.8e-04 and 3.9e-04: still
    # firmly rejected, but still ordered.
    score = 1.0
    if threshold.minimum is not None:
        score *= _one_sided(value, threshold.minimum, softness, is_floor=True)
    if threshold.maximum is not None:
        score *= _one_sided(value, threshold.maximum, softness, is_floor=False)
    return score


def _one_sided(value: float, threshold: float, softness: float, *, is_floor: bool) -> float:
    """Satisfaction of a single bound, softened on a ratio scale."""
    if threshold <= 0:
        # A zero-or-negative bound has no meaningful ratio; fall back to the
        # linear form, which is well defined there.
        width = max(abs(threshold) * softness, _MIN_WIDTH)
        delta = (value - threshold) if is_floor else (threshold - value)
        return _sigmoid(delta / width)
    if value <= 0:
        # Nothing at all: fully satisfies a ceiling, fully fails a floor.
        return 0.0 if is_floor else 1.0
    ratio = value / threshold if is_floor else threshold / value
    return _sigmoid(math.log(ratio) / max(softness, _MIN_WIDTH))


# ---------------------------------------------------------------------------
# Goal configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalConfig:
    """One file from ``configs/goals/``."""

    goal: Goal
    label: str
    description: str
    per_meal: dict[str, Threshold]
    energy_shares: dict[str, Threshold]

    def rules(self) -> list[Rule]:
        """Every numeric rule this goal imposes on a single meal."""
        rules = [
            Rule(
                rule_id=f"goal.{self.goal.value}.{quantity}",
                quantity=quantity,
                threshold=threshold,
                severity="soft",
                message=f"{quantity.replace('_', ' ')} outside {threshold.describe()} for "
                f"the {self.label.lower()} goal",
                source=f"configs/goals/{self.goal.value}.yaml",
            )
            for quantity, threshold in self.per_meal.items()
        ]
        rules += [
            Rule(
                rule_id=f"goal.{self.goal.value}.share.{quantity}",
                quantity=f"share:{quantity}",
                threshold=threshold,
                severity="soft",
                message=f"{quantity.replace('_', ' ')} energy share outside "
                f"{threshold.describe()} for the {self.label.lower()} goal",
                source=f"configs/goals/{self.goal.value}.yaml",
            )
            for quantity, threshold in self.energy_shares.items()
        ]
        return rules


@cache
def load_goal_config(goal: Goal) -> GoalConfig:
    """Load and cache ``configs/goals/<goal>.yaml``."""
    path = GOALS_DIR / f"{Goal(goal).value}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No goal configuration at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return GoalConfig(
        goal=Goal(raw["goal"]),
        label=raw.get("label", raw["goal"]),
        description=(raw.get("description") or "").strip(),
        per_meal={k: Threshold.from_config(v) for k, v in (raw.get("per_meal") or {}).items()},
        energy_shares={
            k: Threshold.from_config(v) for k, v in (raw.get("energy_shares") or {}).items()
        },
    )


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

#: Representative glycemic index by FoodSense category.
#:
#: The design brief specifies a per-food GI *class* looked up by category rather
#: than a per-food GI value, and that is what this is. Values follow the ranges in
#: Foster-Powell, Holt & Brand-Miller (2002), International table of glycemic
#: index and glycemic load values. Foods with essentially no available
#: carbohydrate are given 0 so they contribute nothing rather than a spurious
#: fraction.
GI_BY_CATEGORY: dict[str, float] = {
    "fruit": 45.0,
    "vegetable": 35.0,
    "legume": 30.0,
    "dairy": 35.0,
    "grain": 65.0,
    "cereal": 70.0,
    "baked": 70.0,
    "sweets": 75.0,
    "snack": 65.0,
    "beverage": 60.0,
    "soup_sauce": 45.0,
    "prepared": 60.0,
    "baby_food": 60.0,
    "nut_seed": 20.0,
    "herb_spice": 20.0,
    "meat": 0.0,
    "poultry": 0.0,
    "fish": 0.0,
    "processed_meat": 15.0,
    "fat_oil": 0.0,
}
DEFAULT_GI = 50.0

# ---------------------------------------------------------------------------
# The added-sugar proxy
# ---------------------------------------------------------------------------
#
# DOCUMENTED PROXY -- NOT A MEASURED VALUE.
#
# Neither USDA release FoodSense consumes (SR Legacy 2018, Foundation Foods
# 2025-04) reports FDC nutrient 1235, "Added Sugars". `added_sugars_g` is
# therefore 0.0 for every food in the curated database, and the added-sugar
# rules -- the AAP's "no added sugars below 24 months" and the DGA's daily
# ceiling -- would be unenforceable if taken literally.
#
# So in the categories below, and in sweetened beverages, a food's *total*
# sugars are counted as added sugars. In those categories the sugar is
# overwhelmingly added rather than intrinsic, which makes the substitution
# conservative in the protective direction. Fruit and dairy are deliberately
# excluded: their sugar is intrinsic, and counting it would penalise exactly the
# foods a toddler should be eating.
#
# Where a food does carry a measured added-sugar value, that value is used and
# the proxy is not applied. See data/README.md and docs/architecture.md.

#: Categories whose total sugars stand in for added sugars.
ADDED_SUGAR_PROXY_CATEGORIES = frozenset({"sweets", "snack", "baked", "cereal"})

#: Foods anywhere in the database carrying one of these tags are treated the same
#: way. ``sweetened_beverage`` is category-scoped at build time, so a diet cola
#: and an unsweetened almond milk do not carry it.
ADDED_SUGAR_PROXY_TAGS = frozenset({"added_sugar_source", "sweetened_beverage"})

#: kcal per gram, for energy shares.
_ATWATER = {"protein_g": 4.0, "carbohydrate_g": 4.0, "fat_g": 9.0}


def estimate_glycemic_load(items: Meal | list[MealItem], db: FoodDB) -> float:
    """Estimated glycemic load of a meal: ``sum(GI * available carbohydrate / 100)``.

    Available carbohydrate is total carbohydrate minus fibre. This is the metric
    that stands in for MetaPlate's predicted postprandial glucose response --
    computable from composition alone, which matters because we have no CGM data
    for the age groups FoodSense targets.
    """
    meal_items = items.items if isinstance(items, Meal) else items
    total = 0.0
    for item in meal_items:
        record = db.find(item.food_id)
        if record is None:
            continue
        nutrients = record.nutrients_for(item.quantity_g)
        available_carb = max(nutrients.carbohydrate_g - nutrients.fiber_g, 0.0)
        gi = GI_BY_CATEGORY.get(record.category, DEFAULT_GI)
        total += gi * available_carb / 100.0
    return total


def estimate_added_sugars(items: Meal | list[MealItem], db: FoodDB) -> float:
    """Added sugars in grams, using measured values where they exist and a proxy elsewhere.

    See :data:`ADDED_SUGAR_PROXY_CATEGORIES` for why a proxy is needed at all.
    """
    meal_items = items.items if isinstance(items, Meal) else items
    total = 0.0
    for item in meal_items:
        record = db.find(item.food_id)
        if record is None:
            continue
        nutrients = record.nutrients_for(item.quantity_g)
        if nutrients.added_sugars_g > 0:
            total += nutrients.added_sugars_g
        elif record.category in ADDED_SUGAR_PROXY_CATEGORIES or bool(
            ADDED_SUGAR_PROXY_TAGS & record.tags
        ):
            total += nutrients.sugars_g
    return total


def energy_shares(nutrients: NutrientVector) -> dict[str, float]:
    """Fraction of a meal's energy contributed by protein, carbohydrate and fat."""
    energy = nutrients.energy_kcal
    if energy <= 0:
        return dict.fromkeys(_ATWATER, 0.0)
    return {
        key: getattr(nutrients, key) * kcal_per_g / energy for key, kcal_per_g in _ATWATER.items()
    }


def meal_metrics(items: Meal | list[MealItem], db: FoodDB) -> dict[str, float]:
    """Every quantity a :class:`Rule` can be written against, in one flat mapping.

    Nutrient totals by name, plus ``glycemic_load``, the added-sugar proxy, and
    ``share:<macro>`` entries. Rules name a key in here, so adding a metric is
    enough to make it available to every config file.
    """
    meal_items = items.items if isinstance(items, Meal) else items
    nutrients = db.nutrients_for(meal_items)

    metrics: dict[str, float] = dict(nutrients.as_dict())
    metrics["glycemic_load"] = estimate_glycemic_load(meal_items, db)
    metrics["added_sugars_g"] = estimate_added_sugars(meal_items, db)
    metrics["item_count"] = float(len(meal_items))
    metrics["total_mass_g"] = float(sum(i.quantity_g for i in meal_items))
    for macro, share in energy_shares(nutrients).items():
        metrics[f"share:{macro}"] = share
    return metrics
