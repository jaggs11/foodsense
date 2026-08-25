"""Counterfactual baselines for the ablation table.

Four comparators, chosen so that each isolates one thing FoodSense does:

``wachter``
    *Our own* differential evolution with the availability, safety and sparsity
    terms removed -- validity and L1 distance only, in the spirit of Wachter et
    al. (2017). Same search algorithm, same budget, same surrogate. Any
    difference in the results is attributable to the constraints and to nothing
    else, which is what makes it an ablation rather than a different system.
``dice_random`` / ``dice_genetic``
    DiCE (Mothilal et al., 2020) via ``dice-ml``, model-agnostic and
    availability-blind. It is a genuinely different search, and it also cannot
    express preparation form -- a generic tabular counterfactual method has no
    notion of "quartered" -- so it can only ever repair a choking hazard by
    deleting the food.
``greedy``
    A single-edit hill climber over quantities and whole-food substitutions. The
    "obvious" thing to build, and the thing a reviewer will ask why you did not.

**How the baselines get to violate availability.** A method cannot be measured
as availability-blind if it is handed a space containing only available foods.
Every baseline therefore searches ``planned + pantry + K random foods from the
database``, while the audit set stays ``planned + pantry``. FoodSense searches
the audit set itself, which is the whole point: its zero availability-violation
rate is a property of the search space, not a tuned outcome.

All methods are held to the same evaluation budget and the actual count each
used is reported, so the comparison is on equal terms.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from foodsense import SEED
from foodsense.constraints.engine import RuleEngine
from foodsense.data.fdc import FoodDB
from foodsense.schemas import Meal, MealItem, UserProfile
from foodsense.stage1_prediction.predict import SuitabilityModel
from foodsense.stage2_optimizer.de_optimizer import DEConfig, differential_evolution
from foodsense.stage2_optimizer.objective import CounterfactualObjective, ObjectiveConfig
from foodsense.stage2_optimizer.space import SearchSpace, build_space

__all__ = [
    "METHODS",
    "CFResult",
    "build_unrestricted_space",
    "run_method",
]

#: How many foods the user does *not* have are added to a baseline's search
#: space. Enough that an availability-blind method will reach for one; small
#: enough that the search problem stays comparable in size.
N_UNAVAILABLE_FOODS = 8

#: Categories the extra foods are drawn from -- plausible ingredients, so the
#: baselines are not handicapped by being offered nothing but spices.
_POOL_CATEGORIES = (
    "vegetable",
    "fruit",
    "grain",
    "legume",
    "dairy",
    "poultry",
    "meat",
    "fish",
    "nut_seed",
    "cereal",
    "baked",
)

METHODS = ("foodsense_de", "wachter", "dice_random", "dice_genetic", "greedy")


class _BudgetExceeded(RuntimeError):
    """Raised inside the DiCE adapter when the shared evaluation budget runs out."""


@dataclass(slots=True)
class CFResult:
    """One method's answer for one case, with everything the table needs."""

    method: str
    meal: Meal
    valid: bool
    safe: bool
    score_before: float
    score_after: float
    l1_g: float
    l2_g: float
    n_changed: int
    n_hard_violations: int
    n_unavailable: int
    evaluations: int
    runtime_s: float
    note: str = ""
    planned_mass_g: float = 0.0

    @property
    def norm_l1(self) -> float:
        return self.l1_g / max(self.planned_mass_g, 1.0)

    @property
    def norm_l2(self) -> float:
        return self.l2_g / max(self.planned_mass_g, 1.0)

    @property
    def availability_violation(self) -> bool:
        return self.n_unavailable > 0

    @property
    def safety_violation(self) -> bool:
        return self.n_hard_violations > 0


def build_unrestricted_space(
    planned_meal: Meal,
    pantry: Meal | list[MealItem] | None,
    db: FoodDB,
    profile: UserProfile,
    rng: np.random.Generator,
    n_extra: int = N_UNAVAILABLE_FOODS,
) -> SearchSpace:
    """A search space that also contains foods the user does not have.

    The returned space's ``available_ids`` still lists only what the user has, so
    :meth:`SearchSpace.unavailable_items` measures the violation rather than
    hiding it.
    """
    restricted = build_space(planned_meal, pantry, db, profile=profile)
    available = frozenset(restricted.available_ids)

    pool = [r for r in db.records if r.category in _POOL_CATEGORIES and r.fdc_id not in available]
    extra = [pool[i] for i in rng.choice(len(pool), size=min(n_extra, len(pool)), replace=False)]
    padded_pantry = [
        *(pantry.items if isinstance(pantry, Meal) else list(pantry or [])),
        *(record.as_item(0.0) for record in extra),
    ]
    space = build_space(planned_meal, padded_pantry, db, profile=profile)
    space.available_ids = available
    return space


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def _measure(
    method: str,
    meal: Meal,
    space: SearchSpace,
    planned_meal: Meal,
    profile: UserProfile,
    engine: RuleEngine,
    config: ObjectiveConfig,
    score_before: float,
    evaluations: int,
    runtime_s: float,
    note: str = "",
) -> CFResult:
    """Score a finished candidate identically for every method."""
    evaluation = engine.evaluate(meal, profile)
    planned_by_id = {i.food_id: i.quantity_g for i in planned_meal.items}
    final_by_id = {i.food_id: i.quantity_g for i in meal.items}
    planned_forms = {i.food_id: i.form for i in planned_meal.items}

    l1 = 0.0
    squared = 0.0
    n_changed = 0
    for food_id in set(planned_by_id) | set(final_by_id):
        delta = final_by_id.get(food_id, 0.0) - planned_by_id.get(food_id, 0.0)
        l1 += abs(delta)
        squared += delta * delta
        form_changed = (
            food_id in final_by_id
            and food_id in planned_forms
            and next(i.form for i in meal.items if i.food_id == food_id) != planned_forms[food_id]
        )
        if abs(delta) > config.quantity_epsilon_g or form_changed:
            n_changed += 1

    return CFResult(
        method=method,
        meal=meal,
        valid=evaluation.is_safe and evaluation.score >= config.target_score,
        safe=evaluation.is_safe,
        score_before=score_before,
        score_after=evaluation.score,
        l1_g=l1,
        l2_g=float(np.sqrt(squared)),
        n_changed=n_changed,
        n_hard_violations=len(evaluation.hard_violations),
        n_unavailable=len(space.unavailable_items(meal)),
        evaluations=evaluations,
        runtime_s=runtime_s,
        note=note,
        planned_mass_g=sum(i.quantity_g for i in planned_meal.items),
    )


def _run_de(
    method: str,
    space: SearchSpace,
    planned_meal: Meal,
    profile: UserProfile,
    model: SuitabilityModel,
    engine: RuleEngine,
    config: ObjectiveConfig,
    de_config: DEConfig,
    score_before: float,
    budget: int,
    *,
    use_safety: bool,
    use_sparsity: bool,
    seed: int,
) -> CFResult:
    objective = CounterfactualObjective(
        space, profile, model, engine, config, use_safety=use_safety, use_sparsity=use_sparsity
    )
    result = differential_evolution(
        objective, space, profile, engine, de_config, seed=seed, max_evaluations=budget
    )
    return _measure(
        method,
        result.meal,
        space,
        planned_meal,
        profile,
        engine,
        config,
        score_before,
        result.evaluations,
        result.runtime_s,
        result.converged_reason,
    )


def _run_greedy(
    space: SearchSpace,
    planned_meal: Meal,
    profile: UserProfile,
    model: SuitabilityModel,
    engine: RuleEngine,
    config: ObjectiveConfig,
    score_before: float,
    budget: int,
) -> CFResult:
    """Single-edit hill climbing over quantities and whole-food substitutions.

    Availability-blind and form-blind, like the other baselines: it can only add,
    remove or resize a food, so a hazard that needs re-forming is beyond it.
    """
    started = time.perf_counter()
    objective = CounterfactualObjective(
        space, profile, model, engine, config, use_safety=False, use_sparsity=False
    )

    x = space.encode_planned()
    best = float(objective.values(x[None, :])[0])
    steps_g = (-50.0, -25.0, 25.0, 50.0)

    while objective.n_evaluations < budget:
        candidates: list[np.ndarray] = []
        for i, variable in enumerate(space.variables):
            for step in steps_g:
                trial = x.copy()
                trial[2 * i] = float(np.clip(trial[2 * i] + step, 0.0, variable.max_quantity_g))
                candidates.append(trial)
            drop = x.copy()
            drop[2 * i] = 0.0
            candidates.append(drop)
            if x[2 * i] < config.min_serving_g:
                add = x.copy()
                add[2 * i] = min(60.0, variable.max_quantity_g)
                candidates.append(add)

        if not candidates:
            break
        scores = objective.values(np.vstack(candidates))
        index = int(np.argmin(scores))
        if scores[index] >= best - 1e-6:
            break
        best = float(scores[index])
        x = candidates[index]

    meal = space.decode(x, config.min_serving_g)
    return _measure(
        "greedy",
        meal,
        space,
        planned_meal,
        profile,
        engine,
        config,
        score_before,
        objective.n_evaluations,
        time.perf_counter() - started,
    )


def _caused_by_budget(exc: BaseException) -> bool:
    """dice-ml wraps exceptions raised inside the model, so unwrap before judging."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, _BudgetExceeded):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def _run_dice(
    method: str,
    space: SearchSpace,
    planned_meal: Meal,
    profile: UserProfile,
    model: SuitabilityModel,
    engine: RuleEngine,
    config: ObjectiveConfig,
    score_before: float,
    budget: int,
    seed: int,
) -> CFResult:
    """DiCE over the quantity columns, via ``dice-ml``.

    DiCE works on a tabular feature vector, so the decision space given to it is
    one column per candidate food holding that food's quantity in grams. Forms
    are held at their planned values: a generic tabular counterfactual method has
    no way to express "quartered", which is itself part of what the comparison
    shows.
    """
    started = time.perf_counter()
    try:
        import dice_ml
        import pandas as pd
    except ImportError:  # pragma: no cover - exercised only without the extra
        return _measure(
            method,
            planned_meal,
            space,
            planned_meal,
            profile,
            engine,
            config,
            score_before,
            0,
            time.perf_counter() - started,
            "dice-ml not installed",
        )

    rng = np.random.default_rng(seed)
    columns = [f"q{i}" for i in range(len(space.variables))]
    if not columns:
        return _measure(
            method,
            planned_meal,
            space,
            planned_meal,
            profile,
            engine,
            config,
            score_before,
            0,
            time.perf_counter() - started,
            "empty_search_space",
        )

    planned_x = space.encode_planned()
    forms = planned_x[1::2].copy()

    def to_full(quantities: np.ndarray) -> np.ndarray:
        full = np.empty(space.n_dims, dtype=np.float64)
        full[0::2] = quantities
        full[1::2] = forms
        return full

    counter = {"n": 0, "cap": budget}

    class _Adapter:
        """Presents the surrogate to DiCE as a plain regressor over quantities.

        Also enforces the shared evaluation budget. That is not bookkeeping: DiCE's
        genetic initialiser is
        ``while kx < num_inits: sample uniformly over every feature; keep it if it
        is already a valid counterfactual``. On this problem the feasible region is
        sparse -- a uniform draw puts a positive amount of all eighteen candidate
        foods on one plate, which scores 0.36-0.43 against a 0.70 target -- so the
        condition is essentially never met and the loop does not terminate. Capping
        model calls here turns that into an honest "exhausted its budget" result
        under exactly the same allowance every other method gets, rather than a
        hang. The behaviour is a property of DiCE's assumptions meeting a
        structured domain, not a defect in this harness.
        """

        def predict(self, frame):
            values = np.atleast_2d(np.asarray(frame, dtype=np.float64))
            counter["n"] += len(values)
            if counter["n"] > counter["cap"]:
                raise _BudgetExceeded
            population = np.vstack([to_full(row) for row in values])
            meals = space.decode_many(population, config.min_serving_g)
            out = np.zeros(len(meals))
            live = [i for i, m in enumerate(meals) if m.items]
            if live:
                out[live] = model.predict_many([meals[i] for i in live], profile)
            return out

    # DiCE needs a reference sample to learn each feature's range and, for the
    # genetic search, to know what outcomes are reachable at all.
    #
    # This has to be *sparse*. Sampling every column uniformly puts a positive
    # amount of all eighteen candidate foods on the plate at once -- a three-
    # kilogram meal -- and every such row scores badly, so the reference outcomes
    # span only 0.36-0.43 and a target of 0.70 looks unreachable. DiCE's genetic
    # method then searches indefinitely for a region its reference data says does
    # not exist. Real meals have a handful of foods in them, so the reference
    # sample is built that way, which is also the fairer comparison: the baseline
    # is not handicapped by a degenerate view of the space.
    upper = np.asarray([v.max_quantity_g for v in space.variables])
    n_reference = max(budget // 8, 200)
    sample = np.zeros((n_reference, len(columns)), dtype=np.float64)
    for row in range(n_reference):
        k = int(rng.integers(2, min(7, len(columns) + 1)))
        chosen = rng.choice(len(columns), size=k, replace=False)
        sample[row, chosen] = rng.uniform(20.0, np.minimum(upper[chosen], 250.0))
    frame = pd.DataFrame(sample, columns=columns)
    frame["outcome"] = _Adapter().predict(sample)
    counter["n"] = 0  # the reference sample is setup, not search

    query = pd.DataFrame([planned_x[0::2]], columns=columns)
    note = ""
    meal = planned_meal
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = dice_ml.Data(
                dataframe=frame, continuous_features=columns, outcome_name="outcome"
            )
            wrapped = dice_ml.Model(model=_Adapter(), backend="sklearn", model_type="regressor")
            explainer = dice_ml.Dice(data, wrapped, method=method.removeprefix("dice_"))
            extra: dict = {}
            if method == "dice_genetic":
                # Left at its defaults, DiCE's genetic search runs 500 generations,
                # initialises from a KD-tree over the reference sample and then
                # does a binary post-hoc sparsity pass -- each of which calls the
                # model many times over. Against a surrogate as slow as ours that
                # is minutes per case. These settings bring its budget into the
                # same range as every other method's; the search itself is
                # untouched.
                extra = {
                    "maxiterations": 15,
                    "initialization": "random",
                    "posthoc_sparsity_param": None,
                }
            generated = explainer.generate_counterfactuals(
                query,
                total_CFs=4,
                desired_range=[config.target_score, 1.0],
                verbose=False,
                **extra,
            )
        candidates = generated.cf_examples_list[0].final_cfs_df
        if candidates is None or not len(candidates):
            note = "no_counterfactual_found"
        else:
            # DiCE returns several; take the one closest to the planned meal, which
            # is the choice its own proximity objective is aiming at.
            values = candidates[columns].to_numpy(dtype=np.float64)
            distances = np.abs(values - planned_x[0::2]).sum(axis=1)
            meal = space.decode(to_full(values[int(np.argmin(distances))]), config.min_serving_g)
    except _BudgetExceeded:
        note = "evaluation_budget_exhausted"
    except Exception as exc:
        note = (
            "evaluation_budget_exhausted"
            if _caused_by_budget(exc)
            else (f"dice_failed: {type(exc).__name__}")
        )

    return _measure(
        method,
        meal,
        space,
        planned_meal,
        profile,
        engine,
        config,
        score_before,
        counter["n"],
        time.perf_counter() - started,
        note,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MethodContext:
    """Everything a method needs, built once per case so the comparison is fair."""

    planned_meal: Meal
    pantry: Meal
    profile: UserProfile
    db: FoodDB
    model: SuitabilityModel
    engine: RuleEngine
    config: ObjectiveConfig = field(default_factory=ObjectiveConfig.load)
    de_config: DEConfig = field(default_factory=DEConfig.load)
    budget: int = 6000
    seed: int = SEED


def run_method(method: str, context: MethodContext) -> CFResult:
    """Run one counterfactual method on one case."""
    engine, config = context.engine, context.config
    score_before = engine.evaluate(context.planned_meal, context.profile).score
    rng = np.random.default_rng(context.seed)

    if method == "foodsense_de":
        space = build_space(
            context.planned_meal, context.pantry, context.db, profile=context.profile
        )
        return _run_de(
            method,
            space,
            context.planned_meal,
            context.profile,
            context.model,
            engine,
            config,
            context.de_config,
            score_before,
            context.budget,
            use_safety=True,
            use_sparsity=True,
            seed=context.seed,
        )

    space = build_unrestricted_space(
        context.planned_meal, context.pantry, context.db, context.profile, rng
    )

    if method == "wachter":
        return _run_de(
            method,
            space,
            context.planned_meal,
            context.profile,
            context.model,
            engine,
            config,
            context.de_config,
            score_before,
            context.budget,
            use_safety=False,
            use_sparsity=False,
            seed=context.seed,
        )
    if method == "greedy":
        return _run_greedy(
            space,
            context.planned_meal,
            context.profile,
            context.model,
            engine,
            config,
            score_before,
            context.budget,
        )
    if method in ("dice_random", "dice_genetic"):
        return _run_dice(
            method,
            space,
            context.planned_meal,
            context.profile,
            context.model,
            engine,
            config,
            score_before,
            context.budget,
            context.seed,
        )
    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
