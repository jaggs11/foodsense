"""End-to-end orchestration of the four stages.

    run_pipeline(profile, planned_meal, pantry) -> PipelineTrace

The trace is the product, not a debugging aid: the API returns it verbatim, the UI
renders its stepper from it, and the acceptance tests assert against it. Anything a
stage decides that a reader might reasonably want to check is recorded there rather
than discarded, which is what makes the "verification-guided" claim inspectable
instead of merely asserted.

    Stage 1  score the planned meal            -> Stage1Result
    Stage 2  edit it, minimally and safely     -> Stage2Result
    Stage 3  say what changed, in words        -> Stage3Result
    Stage 4  check every word against USDA     -> VerificationReport

Each stage degrades rather than raising. A missing model, an LLM outage or an
un-optimisable meal produces a trace with a warning in it and the best answer the
remaining stages could reach -- because a demo that shows a partial result is worth
more than one that shows a stack trace.
"""

from __future__ import annotations

import time

from foodsense.constraints.engine import RuleEngine
from foodsense.data.fdc import FoodDB, get_food_db
from foodsense.schemas import (
    Meal,
    MealItem,
    PipelineTrace,
    Stage1Result,
    Stage2Result,
    UserProfile,
)
from foodsense.stage1_prediction.predict import ModelMissingError, SuitabilityModel
from foodsense.stage2_optimizer.de_optimizer import DEConfig, differential_evolution
from foodsense.stage2_optimizer.objective import CounterfactualObjective, ObjectiveConfig
from foodsense.stage2_optimizer.space import build_space
from foodsense.stage3_rag.providers import LLMProvider, TemplateProvider
from foodsense.stage3_rag.translate import build_diff, translate
from foodsense.stage4_verification.verifier import verify

__all__ = ["run_pipeline", "run_scenario"]


def run_pipeline(
    profile: UserProfile,
    planned_meal: Meal,
    pantry: Meal | list[MealItem] | None = None,
    *,
    provider: LLMProvider | None = None,
    db: FoodDB | None = None,
    engine: RuleEngine | None = None,
    model: SuitabilityModel | None = None,
    objective_config: ObjectiveConfig | None = None,
    de_config: DEConfig | None = None,
    scenario: str | None = None,
    seed: int | None = None,
) -> PipelineTrace:
    """Run all four stages and return the full trace."""
    started = time.perf_counter()
    db = db or get_food_db()
    engine = engine or RuleEngine(db=db)
    objective_config = objective_config or ObjectiveConfig.load()
    de_config = de_config or DEConfig.load()
    pantry_meal = pantry if isinstance(pantry, Meal) else Meal(items=list(pantry or []))

    trace = PipelineTrace(
        profile=profile,
        planned_meal=planned_meal,
        pantry=pantry_meal,
        scenario=scenario,
        seed=de_config.seed if seed is None else seed,
    )

    if model is None:
        try:
            from foodsense.stage1_prediction.predict import get_suitability_model

            model = get_suitability_model()
        except ModelMissingError as exc:
            # Without the surrogate there is no smooth objective to climb, so
            # Stage 2 cannot run. Everything else still can: the rules alone can
            # score, describe and verify the planned meal.
            trace.warnings.append(f"Stage 1 unavailable: {exc}")

    # --- Stage 1 -----------------------------------------------------------
    stage1_started = time.perf_counter()
    planned_evaluation = engine.evaluate(planned_meal, profile)
    suitability = (
        model.predict(planned_meal, profile) if model is not None else planned_evaluation.soft_score
    )
    trace.stage1 = Stage1Result(
        suitability=suitability,
        nutrients=db.nutrients_for(planned_meal),
        glycemic_load=_glycemic_load(planned_meal, db),
        rule_evaluation=planned_evaluation,
        model_name=model.backend if model is not None else "rule_engine_fallback",
        runtime_s=round(time.perf_counter() - stage1_started, 4),
    )

    # --- Stage 2 -----------------------------------------------------------
    optimized_meal = planned_meal
    if model is not None:
        space = build_space(planned_meal, pantry_meal, db, profile=profile)
        objective = CounterfactualObjective(space, profile, model, engine, objective_config)
        result = differential_evolution(objective, space, profile, engine, de_config, seed=seed)
        optimized_meal = result.meal
        after_evaluation = engine.evaluate(optimized_meal, profile)
        trace.stage2 = Stage2Result(
            optimized_meal=optimized_meal,
            diff=build_diff(
                planned_meal,
                optimized_meal,
                planned_evaluation.violations,
                objective_config.quantity_epsilon_g,
            ),
            suitability_before=suitability,
            suitability_after=result.terms.suitability,
            objective_value=round(result.terms.total, 5),
            valid=result.valid,
            rule_evaluation_after=after_evaluation,
            n_generations=result.generations,
            n_evaluations=result.evaluations,
            runtime_s=round(result.runtime_s, 4),
            method="foodsense_de",
            search_space_size=len(space),
        )
        if not result.valid:
            trace.warnings.append(
                f"Stage 2 did not reach the target score "
                f"({after_evaluation.score:.2f} < {objective_config.target_score:.2f}); "
                f"returning the best safe edit it found."
            )

    # --- Stage 3 -----------------------------------------------------------
    trace.stage3 = translate(
        planned_meal,
        optimized_meal,
        profile,
        provider=provider or TemplateProvider(),
        engine=engine,
        db=db,
        diff=trace.stage2.diff if trace.stage2 else None,
    )
    if trace.stage3.fallback_used:
        trace.warnings.append(
            f"Stage 3 provider {trace.stage3.provider!r} failed; "
            f"used the deterministic template instead."
        )

    # --- Stage 4 -----------------------------------------------------------
    final_meal, report = verify(
        trace.stage3.items,
        profile,
        claimed_nutrients=trace.stage3.claimed_nutrients,
        db=db,
        engine=engine,
    )
    trace.stage4 = report
    trace.final_meal = final_meal
    trace.final_rule_evaluation = engine.evaluate(final_meal, profile)
    if not report.final_pass:
        trace.warnings.append("Stage 4 could not make the final meal safe; see stage4.flagged.")

    trace.total_runtime_s = round(time.perf_counter() - started, 4)
    return trace


def run_scenario(key: str, **kwargs) -> PipelineTrace:
    """Run one of the built-in demo scenarios end to end."""
    from foodsense.scenarios import load_scenario

    scenario = load_scenario(key)
    db = kwargs.pop("db", None) or get_food_db()
    return run_pipeline(
        scenario.profile,
        scenario.planned_meal(db),
        scenario.pantry_meal(db),
        db=db,
        scenario=key,
        **kwargs,
    )


def _glycemic_load(meal: Meal, db: FoodDB) -> float:
    from foodsense.constraints.goals import estimate_glycemic_load

    return round(estimate_glycemic_load(meal, db), 2)
