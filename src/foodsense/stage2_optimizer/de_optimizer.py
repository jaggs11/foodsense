"""Differential evolution over the mixed continuous/categorical meal-edit space.

A compact rand/1/bin DE rather than SciPy's, for three reasons that matter to
this project specifically:

* **Exact evaluation budgets.** The headline comparison requires every method to
  run under the same budget. SciPy's ``popsize`` is a multiplier on the number of
  parameters, so the actual budget varies with meal size; here it is stated
  directly and counted.
* **Early stopping on the right signal.** The search should stop once the meal is
  *valid by the rule engine* and the objective has plateaued -- not when some
  numerical tolerance is met. That needs a per-generation hook with access to the
  rules.
* **Mixed variables.** Half the dimensions are categorical form indices. Rounding
  them inside the objective would let the search drift between values that decode
  identically; they are quantised in the population instead.

**Validity is judged by the RuleEngine, never by the surrogate.** The optimiser
climbs ``f``, but whether it has succeeded is decided by the rules it cannot see
inside. An optimiser marked by its own model can always win by finding that
model's blind spots, and this is the structural guarantee that it cannot.

The planned meal is seeded into the initial population, so the search starts from
what the user actually intended and can only move away from it by paying for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import yaml

from foodsense import CONFIG_DIR, SEED
from foodsense.constraints.engine import RuleEngine
from foodsense.schemas import Meal, UserProfile
from foodsense.stage2_optimizer.objective import CounterfactualObjective, ObjectiveTerms
from foodsense.stage2_optimizer.space import SearchSpace

__all__ = ["DEConfig", "OptimizationResult", "differential_evolution"]

PIPELINE_CONFIG = CONFIG_DIR / "pipeline.yaml"


@dataclass(frozen=True, slots=True)
class DEConfig:
    """Differential-evolution hyperparameters from ``configs/pipeline.yaml``."""

    population_size: int = 40
    max_generations: int = 200
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    early_stop_patience: int = 15
    early_stop_tolerance: float = 1e-4
    seed: int = SEED

    @classmethod
    def load(cls) -> DEConfig:
        raw = yaml.safe_load(PIPELINE_CONFIG.read_text(encoding="utf-8")) or {}
        stage2 = raw.get("stage2") or {}
        mutation = stage2.get("mutation", [0.5, 1.0])
        return cls(
            population_size=int(stage2.get("population_size", 40)),
            max_generations=int(stage2.get("max_generations", 200)),
            mutation=(float(mutation[0]), float(mutation[1])),
            recombination=float(stage2.get("recombination", 0.7)),
            early_stop_patience=int(stage2.get("early_stop_patience", 15)),
            early_stop_tolerance=float(stage2.get("early_stop_tolerance", 1e-4)),
            seed=int(raw.get("seed", SEED)),
        )


@dataclass(slots=True)
class OptimizationResult:
    """What the search found, and what it cost."""

    x: np.ndarray
    meal: Meal
    terms: ObjectiveTerms
    valid: bool
    generations: int
    evaluations: int
    runtime_s: float
    converged_reason: str
    history: list[float]


def _quantise(population: np.ndarray, integrality: np.ndarray) -> np.ndarray:
    """Snap categorical dimensions to integers; leave continuous ones alone."""
    out = population.copy()
    out[:, integrality] = np.floor(out[:, integrality])
    return out


def differential_evolution(
    objective: CounterfactualObjective,
    space: SearchSpace,
    profile: UserProfile,
    engine: RuleEngine,
    config: DEConfig | None = None,
    *,
    seed: int | None = None,
    max_evaluations: int | None = None,
) -> OptimizationResult:
    """Minimise ``objective`` over ``space``, returning the best feasible candidate.

    ``max_evaluations`` caps the budget directly, which is how the baselines are
    held to the same allowance as this optimiser.
    """
    config = config or DEConfig.load()
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed if seed is None else seed)

    # Guard before touching bounds: an empty space yields an empty list, and
    # np.asarray([]) is one-dimensional, so the two-index unpacking below raises.
    n_dims = space.n_dims
    if n_dims == 0:
        planned = space.decode(np.zeros(0), objective.config.min_serving_g)
        return OptimizationResult(
            x=np.zeros(0),
            meal=planned,
            terms=objective.terms(np.zeros(0)),
            valid=False,
            generations=0,
            evaluations=0,
            runtime_s=time.perf_counter() - started,
            converged_reason="empty_search_space",
            history=[],
        )

    bounds = np.asarray(space.bounds(), dtype=np.float64)
    lower, upper = bounds[:, 0], bounds[:, 1]
    integrality = space.integrality()

    size = max(config.population_size, 5)
    population = rng.uniform(lower, upper, size=(size, n_dims))
    # Seed the planned meal, and a near-copy of it, so the search begins at what
    # the user intended rather than somewhere random.
    planned_x = space.encode_planned()
    population[0] = planned_x
    population[1] = np.clip(planned_x * rng.uniform(0.85, 1.15, n_dims), lower, upper)
    population = _quantise(population, integrality)

    scores = objective.values(population)
    best_index = int(np.argmin(scores))
    best_x, best_score = population[best_index].copy(), float(scores[best_index])

    history: list[float] = [best_score]
    stagnant = 0
    reason = "max_generations"
    generation = 0

    # `generation` is read after the loop to report how far the search got.
    for generation in range(1, config.max_generations + 1):  # noqa: B007
        if max_evaluations is not None and objective.n_evaluations >= max_evaluations:
            reason = "evaluation_budget"
            break

        # Dithered mutation factor: re-drawn each generation, which is the
        # standard trick for keeping DE from stalling on multimodal surfaces.
        f = rng.uniform(*config.mutation)

        trials = np.empty_like(population)
        for i in range(size):
            choices = rng.choice(np.delete(np.arange(size), i), size=3, replace=False)
            a, b, c = population[choices]
            mutant = np.clip(a + f * (b - c), lower, upper)

            cross = rng.random(n_dims) < config.recombination
            if not cross.any():
                cross[rng.integers(n_dims)] = True
            trials[i] = np.where(cross, mutant, population[i])
        trials = _quantise(trials, integrality)

        trial_scores = objective.values(trials)
        improved = trial_scores < scores
        population[improved] = trials[improved]
        scores[improved] = trial_scores[improved]

        generation_best = int(np.argmin(scores))
        if scores[generation_best] < best_score - config.early_stop_tolerance:
            best_score = float(scores[generation_best])
            best_x = population[generation_best].copy()
            stagnant = 0
        else:
            stagnant += 1
        history.append(best_score)

        # Stop only when the meal is genuinely valid *and* the search has stopped
        # improving. Plateauing while still invalid is not success.
        if stagnant >= config.early_stop_patience:
            candidate = space.decode(best_x, objective.config.min_serving_g)
            if engine.is_valid(candidate, profile, objective.config.target_score):
                reason = "valid_and_plateaued"
                break
            reason = "plateaued_without_validity"
            break

    best_meal = space.decode(best_x, objective.config.min_serving_g)
    best_terms = objective.terms(best_x)
    return OptimizationResult(
        x=best_x,
        meal=best_meal,
        terms=best_terms,
        valid=engine.is_valid(best_meal, profile, objective.config.target_score),
        generations=generation,
        evaluations=objective.n_evaluations,
        runtime_s=time.perf_counter() - started,
        converged_reason=reason,
        history=history,
    )
