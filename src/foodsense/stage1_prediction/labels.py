"""Weak-supervision labels for the Stage-1 surrogate.

The label for ``(meal, profile)`` is the rule engine's **soft guideline score**
perturbed by ``N(0, sigma)`` and clipped to [0, 1]. Three things about that are
deliberate.

**Why the rule engine supplies the labels at all.** There is no dataset of
"how suitable was this meal for this toddler". There is, however, a large body of
published guidance, and it is already encoded in ``configs/``. Generating labels
from it is weak supervision in the standard sense: a programmatic labelling
function stands in for absent ground truth, and the model's job is to generalise
it rather than to discover it.

**Why noise is added.** Without it the target is a deterministic function of the
features and a boosted-tree model would simply memorise the rule boundaries,
reproducing the same step function the surrogate exists to smooth. The noise
forces it to fit the trend rather than the edges.

**Why the *soft* score.** ``RuleEvaluation.score`` folds in hard-safety
violations, which are facts about ``(hazard_class, form)`` pairs and food tags.
No nutrient feature vector encodes them, so including them in the label would
only add irreducible noise. Hard safety is enforced where it can be: as an
explicit penalty in the Stage-2 objective and as a scan in Stage 4.

Meals alone do not span the space the optimiser will search -- real recipes
cluster around plausible portions. Synthetic perturbations (rescaling, dropping
an item, adding one) push the training distribution out to the edges the
optimiser will actually visit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from foodsense import SEED
from foodsense.constraints.engine import RuleEngine
from foodsense.data.fdc import FoodDB
from foodsense.schemas import AgeGroup, Goal, HealthFlag, Meal, MealItem, UserProfile
from foodsense.stage1_prediction.features import feature_names, meal_features

__all__ = ["Dataset", "build_dataset", "perturb_meal", "sample_profile"]

#: Health flags worth sampling per life stage, with the probability of each being
#: set. Sampling a toddler on warfarin would waste capacity on a combination the
#: system will never see; sampling an older adult with one is the point.
FLAG_PRIORS: dict[AgeGroup, tuple[tuple[HealthFlag, float], ...]] = {
    AgeGroup.TODDLER: (
        (HealthFlag.IRON_FOCUS, 0.35),
        (HealthFlag.STRICT_NO_ADDED_SUGAR, 0.25),
    ),
    AgeGroup.ADULT: (
        (HealthFlag.HYPERTENSION, 0.20),
        (HealthFlag.DIABETES, 0.20),
        (HealthFlag.IRON_FOCUS, 0.10),
    ),
    AgeGroup.OLDER_ADULT: (
        (HealthFlag.HYPERTENSION, 0.35),
        (HealthFlag.DIABETES, 0.20),
        (HealthFlag.WARFARIN, 0.12),
        (HealthFlag.STATIN, 0.20),
        (HealthFlag.METFORMIN, 0.15),
        (HealthFlag.ACE_INHIBITOR_OR_K_SPARING_DIURETIC, 0.12),
        (HealthFlag.MAOI, 0.05),
        (HealthFlag.DYSPHAGIA, 0.10),
    ),
}

#: Age in months and body weight, sampled per life stage.
_AGE_MONTHS_RANGE: dict[AgeGroup, tuple[int, int]] = {
    AgeGroup.TODDLER: (12, 47),
    AgeGroup.ADULT: (18 * 12, 64 * 12),
    AgeGroup.OLDER_ADULT: (65 * 12, 92 * 12),
}
_WEIGHT_KG_RANGE: dict[AgeGroup, tuple[float, float]] = {
    AgeGroup.TODDLER: (9.0, 16.0),
    AgeGroup.ADULT: (50.0, 100.0),
    AgeGroup.OLDER_ADULT: (45.0, 95.0),
}

#: Quantity bounds used when a perturbation adds a food.
_ADDED_ITEM_G = (15.0, 180.0)

#: Categories a perturbation may pull an extra food from -- the ones a person
#: plausibly has to hand. Adding a random spice to a meal teaches nothing.
_ADDABLE_CATEGORIES = (
    "vegetable",
    "fruit",
    "grain",
    "legume",
    "dairy",
    "poultry",
    "meat",
    "fish",
    "nut_seed",
    "baked",
    "snack",
    "sweets",
    "soup_sauce",
    "cereal",
)


@dataclass(slots=True)
class Dataset:
    """A built training set, plus everything needed to split it honestly."""

    X: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    groups: np.ndarray
    feature_names: list[str] = field(default_factory=feature_names)
    #: The (meal, profile) pair behind each row, when ``keep_examples`` was set.
    #: Off by default -- training does not need them and 24k Pydantic meals is a
    #: lot of memory to hold for nothing.
    examples: list[tuple[Meal, UserProfile]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_groups(self) -> int:
        return len(np.unique(self.groups))


def sample_profile(rng: random.Random, age_group: AgeGroup | None = None) -> UserProfile:
    """Draw a plausible user profile.

    Age group and goal are uniform, so no life stage or goal is under-represented;
    flags follow :data:`FLAG_PRIORS`.
    """
    age_group = age_group or rng.choice(list(AgeGroup))
    low, high = _AGE_MONTHS_RANGE[age_group]
    weight_low, weight_high = _WEIGHT_KG_RANGE[age_group]
    flags = [flag for flag, p in FLAG_PRIORS.get(age_group, ()) if rng.random() < p]
    return UserProfile(
        age_group=age_group,
        age_months=rng.randint(low, high),
        weight_kg=round(rng.uniform(weight_low, weight_high), 1),
        goal=rng.choice(list(Goal)),
        health_flags=flags,
    )


def perturb_meal(meal: Meal, rng: random.Random, db: FoodDB) -> Meal:
    """Push a meal off its recipe-typical shape.

    One operation per call, chosen uniformly. The point is coverage: the optimiser
    will propose 15 g of one thing and 300 g of another, and the surrogate has to
    have seen that region to score it sensibly.
    """
    items = [i.model_copy() for i in meal.items]
    operation = rng.choice(("scale_all", "scale_one", "drop_one", "add_one"))

    if operation == "scale_all":
        factor = rng.uniform(0.5, 1.8)
        items = [
            i.model_copy(update={"quantity_g": round(i.quantity_g * factor, 1)}) for i in items
        ]

    elif operation == "scale_one" and items:
        index = rng.randrange(len(items))
        factor = rng.uniform(0.2, 3.0)
        items[index] = items[index].model_copy(
            update={"quantity_g": round(items[index].quantity_g * factor, 1)}
        )

    elif operation == "drop_one" and len(items) > 2:
        items.pop(rng.randrange(len(items)))

    elif operation == "add_one":
        candidates = [r for r in db.records if r.category in _ADDABLE_CATEGORIES]
        if candidates:
            record = rng.choice(candidates)
            items.append(record.as_item(round(rng.uniform(*_ADDED_ITEM_G), 1)))

    return Meal(items=[i for i in items if i.quantity_g > 0])


def build_dataset(
    meals: list[Meal] | list[list[MealItem]],
    engine: RuleEngine,
    n_profiles_per_meal: int = 3,
    perturbation_rate: float = 0.5,
    noise_sigma: float = 0.05,
    seed: int = SEED,
    keep_examples: bool = False,
) -> Dataset:
    """Cross meals with sampled profiles and label each pair with the rule engine.

    ``groups`` carries the index of the source meal so that a train/test split can
    keep every variant of one meal on the same side -- otherwise a perturbed copy
    of a training meal leaks into the test set and the metrics flatter the model.
    """
    rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    db = engine.db

    rows: list[np.ndarray] = []
    clean: list[float] = []
    groups: list[int] = []
    examples: list[tuple[Meal, UserProfile]] = []

    for meal_index, raw_meal in enumerate(meals):
        meal = raw_meal if isinstance(raw_meal, Meal) else Meal(items=list(raw_meal))
        for _ in range(n_profiles_per_meal):
            candidate = perturb_meal(meal, rng, db) if rng.random() < perturbation_rate else meal
            if len(candidate.items) < 1:
                continue
            profile = sample_profile(rng)
            evaluation = engine.evaluate(candidate, profile)
            rows.append(meal_features(candidate, profile, db))
            clean.append(evaluation.soft_score)
            groups.append(meal_index)
            if keep_examples:
                examples.append((candidate, profile))

    X = np.asarray(rows, dtype=np.float64)
    y_clean = np.asarray(clean, dtype=np.float64)
    y = np.clip(y_clean + numpy_rng.normal(0.0, noise_sigma, size=y_clean.shape), 0.0, 1.0)
    return Dataset(
        X=X,
        y=y,
        y_clean=y_clean,
        groups=np.asarray(groups, dtype=np.int64),
        examples=examples,
    )
