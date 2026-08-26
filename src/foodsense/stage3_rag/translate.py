"""Stage 3: optimised meal + diff -> a recommendation a person can act on.

Assembles the diff between the planned and optimised meals, retrieves real USDA
candidates to ground the generator, attaches age-appropriate texture phrasing, and
asks a provider to write it up.

The diff is computed here rather than inside a provider on purpose. "What changed
and why" is a fact about the optimisation, and a generative step should be
describing that fact, not deciding it. Each change carries a one-line ``reason``
drawn from the rule engine, so the explanation the user reads traces back to a
specific guideline rather than to the model's impression of one.
"""

from __future__ import annotations

import time

from foodsense.constraints.age_rules import load_age_config
from foodsense.constraints.engine import RuleEngine
from foodsense.data.fdc import FoodDB, get_food_db
from foodsense.schemas import (
    ItemChange,
    Meal,
    MealDiff,
    Stage3Result,
    UserProfile,
    Violation,
)
from foodsense.stage3_rag.providers import (
    LLMProvider,
    TemplateProvider,
    TranslationRequest,
    phrase_for,
)
from foodsense.stage3_rag.retriever import FoodRetriever, get_retriever

__all__ = ["build_diff", "translate"]


def build_diff(
    planned: Meal, optimized: Meal, violations: list[Violation], epsilon_g: float = 2.0
) -> MealDiff:
    """What changed between two meals, with a reason attached where one exists."""
    planned_by_id = {i.food_id: i for i in planned.items}
    optimized_by_id = {i.food_id: i for i in optimized.items}

    # A violation names the food it was about, so a repair can be explained in
    # terms of the rule that caused it rather than as an unexplained edit.
    reason_by_food: dict[str, str] = {}
    for violation in violations:
        for food_id in violation.offending_items:
            reason_by_food.setdefault(food_id, violation.message)

    changes: list[ItemChange] = []
    l1 = 0.0
    squared = 0.0
    n_changed = 0

    for food_id in dict.fromkeys([*planned_by_id, *optimized_by_id]):
        before = planned_by_id.get(food_id)
        after = optimized_by_id.get(food_id)
        old_q = before.quantity_g if before else 0.0
        new_q = after.quantity_g if after else 0.0
        delta = new_q - old_q
        l1 += abs(delta)
        squared += delta * delta

        name = (after or before).name  # type: ignore[union-attr]
        reason = reason_by_food.get(food_id)

        if before is None and after is not None:
            change_type = "added"
        elif after is None and before is not None:
            change_type = "removed"
            reason = reason or "removed to make room for a better fit"
        elif before and after and (abs(delta) > epsilon_g or before.form != after.form):
            change_type = "modified"
        else:
            change_type = "unchanged"

        if change_type != "unchanged":
            n_changed += 1

        changes.append(
            ItemChange(
                change_type=change_type,
                food_id=food_id,
                name=name,
                old_quantity_g=old_q if before else None,
                new_quantity_g=new_q if after else None,
                old_form=before.form if before else None,
                new_form=after.form if after else None,
                reason=reason,
            )
        )

    return MealDiff(
        changes=changes,
        l1_distance_g=round(l1, 1),
        l2_distance_g=round(squared**0.5, 1),
        n_items_changed=n_changed,
    )


def translate(
    planned: Meal,
    optimized: Meal,
    profile: UserProfile,
    *,
    provider: LLMProvider | None = None,
    engine: RuleEngine | None = None,
    db: FoodDB | None = None,
    retriever: FoodRetriever | None = None,
    diff: MealDiff | None = None,
    retrieval_top_k: int = 5,
) -> Stage3Result:
    """Describe the optimised meal for this profile."""
    started = time.perf_counter()
    db = db or get_food_db()
    engine = engine or RuleEngine(db=db)
    provider = provider or TemplateProvider()
    retriever = retriever or get_retriever()

    planned_evaluation = engine.evaluate(planned, profile)
    diff = diff or build_diff(planned, optimized, planned_evaluation.violations)

    # Retrieval is over the *edited* foods: those are the ones the write-up has to
    # name, and the ones a generator would otherwise be tempted to invent.
    edited_names = [c.name for c in diff.edits] or [i.name for i in optimized.items]
    candidates = retriever.candidates_for(edited_names, k=retrieval_top_k)

    age_config = load_age_config(profile.age_group)
    texture_notes = dict(age_config.texture_notes)

    fixed = [
        v.message
        for v in planned_evaluation.hard_violations
        if not any(
            i.food_id in v.offending_items and i.form == _planned_form(planned, i.food_id)
            for i in optimized.items
        )
    ]

    request = TranslationRequest(
        profile=profile,
        planned_meal=planned,
        optimized_meal=optimized,
        changes=[c.model_dump(mode="json") for c in diff.edits],
        candidates=candidates,
        texture_notes=texture_notes,
        violations_fixed=fixed,
    )
    response = provider.generate(request)

    return Stage3Result(
        items=response.items,
        text=response.text,
        rationale=response.rationale,
        provider=response.provider,
        retrieved_candidates=candidates,
        claimed_nutrients=db.nutrients_for(response.items),
        fallback_used=response.fallback_used,
        runtime_s=round(time.perf_counter() - started, 4),
    )


def _planned_form(planned: Meal, food_id: str):
    for item in planned.items:
        if item.food_id == food_id:
            return item.form
    return None


def describe_form(form, texture_notes: dict[str, str]) -> str:
    """Public wrapper so the API and UI can phrase a form the same way Stage 3 does."""
    return phrase_for(form, texture_notes)
