"""The counterfactual objective.

    lambda1 * max(0, target - f(x))          get the meal over the line
  + lambda2 * L1(x, x0) / scale              stay close to what was planned
  + lambda3 * (items changed)                change as few things as possible
  + big_penalty * (hard safety violations)   never trade safety for anything

Weights come from ``configs/pipeline.yaml``. Three things about the shape of this
are worth stating.

**The validity term is one-sided.** Once ``f(x)`` clears the target there is
nothing further to gain, so the optimiser stops improving nutrition and starts
minimising distance instead. Without the ``max(0, ...)`` it would keep pushing
toward a perfect meal and hand back a completely rewritten plate -- which is a
different product from the one this project is building.

**The safety term is not a trade-off.** ``big_penalty`` is large enough that no
combination of the other terms can buy a hazard back. Safety is lexicographic in
practice while remaining a single differentiable-ish scalar the search can read.

**Hard safety comes from the rule engine, not the surrogate.** Whether a meal
contains whole grapes is a fact about ``(hazard_class, form)`` that no nutrient
vector encodes, so the surrogate was never trained on it (see
``stage1_prediction/labels.py``). It is counted here directly instead.

The ablation switches exist so the Wachter-style baseline can be *this* optimiser
with terms removed, which isolates the contribution of each constraint rather
than confounding it with a different search algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from foodsense import CONFIG_DIR
from foodsense.constraints.engine import RuleEngine
from foodsense.schemas import UserProfile
from foodsense.stage1_prediction.features import meal_features
from foodsense.stage1_prediction.predict import SuitabilityModel
from foodsense.stage2_optimizer.space import SearchSpace

__all__ = ["CounterfactualObjective", "ObjectiveConfig", "ObjectiveTerms", "meal_diff"]

PIPELINE_CONFIG = CONFIG_DIR / "pipeline.yaml"

#: Objective value returned for a candidate with no food in it at all. Large
#: enough to be rejected outright, finite so the search surface stays defined.
EMPTY_MEAL_PENALTY = 1e4


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    """Weights and tolerances from ``configs/pipeline.yaml``."""

    lambda_validity: float = 1.0
    lambda_distance: float = 0.4
    lambda_sparsity: float = 0.15
    big_penalty: float = 1000.0
    distance_scale_g: float = 200.0
    quantity_epsilon_g: float = 2.0
    min_serving_g: float = 10.0
    lambda_form_preference: float = 0.05
    target_score: float = 0.70

    @classmethod
    def load(cls) -> ObjectiveConfig:
        raw = yaml.safe_load(PIPELINE_CONFIG.read_text(encoding="utf-8")) or {}
        stage2 = raw.get("stage2") or {}
        stage1 = raw.get("stage1") or {}
        return cls(
            lambda_validity=float(stage2.get("lambda_validity", 1.0)),
            lambda_distance=float(stage2.get("lambda_distance", 0.4)),
            lambda_sparsity=float(stage2.get("lambda_sparsity", 0.15)),
            big_penalty=float(stage2.get("big_penalty", 1000.0)),
            distance_scale_g=float(stage2.get("distance_scale_g", 200.0)),
            quantity_epsilon_g=float(stage2.get("quantity_epsilon_g", 2.0)),
            min_serving_g=float(stage2.get("min_serving_g", 10.0)),
            lambda_form_preference=float(stage2.get("lambda_form_preference", 0.05)),
            target_score=float(stage1.get("target_score", 0.70)),
        )


@dataclass(slots=True)
class ObjectiveTerms:
    """A candidate's score, broken into the parts that produced it."""

    total: float
    validity: float
    distance: float
    sparsity: float
    safety: float
    suitability: float
    l1_g: float
    l2_g: float
    n_changed: int
    n_hard_violations: int

    @property
    def is_safe(self) -> bool:
        return self.n_hard_violations == 0


def meal_diff(
    space: SearchSpace, x: np.ndarray, epsilon_g: float
) -> tuple[float, float, int, float]:
    """L1 grams, L2 grams and the number of items changed, against the planned meal.

    An item counts as changed if its quantity moved by more than ``epsilon_g`` *or*
    its form changed. Form matters even at identical grams -- re-quartering the
    grapes is an edit the user has to perform, and the sparsity term should say so.
    """
    l1 = 0.0
    squared = 0.0
    n_changed = 0
    form_cost = 0.0
    for i, variable in enumerate(space.variables):
        quantity = float(x[2 * i])
        if quantity < epsilon_g:
            quantity = 0.0
        delta = quantity - variable.planned_quantity_g
        l1 += abs(delta)
        squared += delta * delta

        form_index = min(max(int(x[2 * i + 1]), 0), max(len(variable.forms) - 1, 0))
        form_changed = quantity > 0 and form_index != variable.planned_form_index
        if quantity > 0 and variable.form_costs:
            form_cost += variable.form_costs[form_index]
        if abs(delta) > epsilon_g or form_changed:
            n_changed += 1
    return l1, float(np.sqrt(squared)), n_changed, form_cost


class CounterfactualObjective:
    """Scores candidate meals for one profile against one planned meal."""

    def __init__(
        self,
        space: SearchSpace,
        profile: UserProfile,
        model: SuitabilityModel,
        engine: RuleEngine,
        config: ObjectiveConfig | None = None,
        *,
        use_safety: bool = True,
        use_sparsity: bool = True,
        use_distance: bool = True,
    ) -> None:
        self.space = space
        self.profile = profile
        self.model = model
        self.engine = engine
        self.config = config or ObjectiveConfig.load()
        self.use_safety = use_safety
        self.use_sparsity = use_sparsity
        self.use_distance = use_distance
        self.n_evaluations = 0

    # -- single candidate ---------------------------------------------------

    def terms(self, x: np.ndarray) -> ObjectiveTerms:
        """Full breakdown for one decision vector."""
        return self.terms_many(np.atleast_2d(x))[0]

    def __call__(self, x: np.ndarray) -> float:
        return self.terms(x).total

    # -- population ---------------------------------------------------------

    def terms_many(self, population: np.ndarray) -> list[ObjectiveTerms]:
        """Score a whole population.

        Batched deliberately: the surrogate is by far the most expensive part of
        an evaluation, and differential evolution generates a full population at
        a time, so scoring them in one call rather than a Python loop is most of
        the optimiser's speed.
        """
        population = np.atleast_2d(population)
        self.n_evaluations += len(population)
        config = self.config

        meals = [self.space.decode(row, config.min_serving_g) for row in population]
        non_empty = [i for i, meal in enumerate(meals) if meal.items]

        suitability = np.zeros(len(meals), dtype=np.float64)
        if non_empty:
            features = np.vstack(
                [meal_features(meals[i], self.profile, self.space.db) for i in non_empty]
            )
            suitability[non_empty] = self.model.predict_features(features)

        out: list[ObjectiveTerms] = []
        for i, meal in enumerate(meals):
            if not meal.items:
                out.append(
                    ObjectiveTerms(
                        total=EMPTY_MEAL_PENALTY,
                        validity=EMPTY_MEAL_PENALTY,
                        distance=0.0,
                        sparsity=0.0,
                        safety=0.0,
                        suitability=0.0,
                        l1_g=0.0,
                        l2_g=0.0,
                        n_changed=0,
                        n_hard_violations=0,
                    )
                )
                continue

            l1, l2, n_changed, form_cost = meal_diff(
                self.space, population[i], config.quantity_epsilon_g
            )
            validity = config.lambda_validity * max(0.0, config.target_score - suitability[i])
            distance = (
                config.lambda_distance * l1 / config.distance_scale_g if self.use_distance else 0.0
            )
            sparsity = config.lambda_sparsity * n_changed if self.use_sparsity else 0.0
            sparsity += config.lambda_form_preference * form_cost if self.use_sparsity else 0.0

            n_hard = self.engine.count_hard_violations(meal, self.profile)
            safety = config.big_penalty * n_hard if self.use_safety else 0.0

            out.append(
                ObjectiveTerms(
                    total=validity + distance + sparsity + safety,
                    validity=validity,
                    distance=distance,
                    sparsity=sparsity,
                    safety=safety,
                    suitability=float(suitability[i]),
                    l1_g=l1,
                    l2_g=l2,
                    n_changed=n_changed,
                    n_hard_violations=n_hard,
                )
            )
        return out

    def values(self, population: np.ndarray) -> np.ndarray:
        """Just the scalars, for the optimiser's inner loop."""
        return np.asarray([t.total for t in self.terms_many(population)], dtype=np.float64)
