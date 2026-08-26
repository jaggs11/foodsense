"""Stage 4: verify what Stage 3 produced against USDA ground truth.

This is extension #3, and the premise is simple: **nothing a generator says is
trusted**. Whatever Stage 3 hands over -- template or LLM -- is re-checked from
scratch against the curated database before a person sees it.

Five steps, in order:

1. **Match.** Each generated item's name is fuzzy-matched to a real food. At or
   above the threshold the match is accepted; below it, the item is recorded as
   ``unmatched`` and replaced by the retriever's best real candidate rather than
   silently dropped or silently kept.
2. **Recompute.** Nutrients come from the database row times the quantity. The
   generator's own numbers are never used, only compared against.
3. **Compare.** Claims outside the tolerance are corrected to the database value
   and the correction is recorded, with what was claimed and what is true.
4. **Safety scan.** The same ``RuleEngine`` that produced the labels and judged
   Stage-2 validity re-runs on the final item list. A hazard that survived to
   here is a hazard the earlier stages missed, which is exactly what this is for.
5. **Repair.** A surviving hard violation is fixed by moving to the nearest safe
   form, or by removing the item when no safe form exists. Repairs are logged.

The counts in :class:`VerificationReport` are the project's headline metric, so
the report distinguishes *what was wrong* from *what was fixed*: a run that
corrected three quantities and re-formed one hazard is a run where four things
would otherwise have reached a user.
"""

from __future__ import annotations

import time

from foodsense.constraints.age_rules import nearest_safe_form
from foodsense.constraints.engine import RuleEngine
from foodsense.data.fdc import DEFAULT_MATCH_THRESHOLD, FoodDB, get_food_db
from foodsense.schemas import (
    ItemCorrection,
    Meal,
    MealItem,
    NutrientVector,
    SafetyFix,
    UserProfile,
    VerificationReport,
)
from foodsense.stage3_rag.retriever import FoodRetriever, get_retriever

__all__ = ["DEFAULT_TOLERANCE", "verify"]

#: Relative gap beyond which a claimed value is corrected to the database value.
DEFAULT_TOLERANCE = 0.10

#: Nutrients compared against a provider's claims. The macros a person or a label
#: would actually state -- comparing all 33 would report corrections for trace
#: minerals nobody claimed in the first place.
CHECKED_NUTRIENTS = (
    "energy_kcal",
    "protein_g",
    "carbohydrate_g",
    "fat_g",
    "fiber_g",
    "sugars_g",
    "sodium_mg",
)


def verify(
    items: list[MealItem],
    profile: UserProfile,
    *,
    claimed_nutrients: NutrientVector | None = None,
    db: FoodDB | None = None,
    engine: RuleEngine | None = None,
    retriever: FoodRetriever | None = None,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[Meal, VerificationReport]:
    """Verify and repair a generated item list. Returns the final meal and the report."""
    started = time.perf_counter()
    db = db or get_food_db()
    engine = engine or RuleEngine(db=db)
    retriever = retriever or get_retriever()

    report = VerificationReport(checked=len(items))
    verified: list[MealItem] = []

    # --- 1-3: match, recompute, correct ------------------------------------
    for item in items:
        record = db.find(item.food_id)
        if record is None:
            # The id is not real. Fall back to matching on the name, then to the
            # retriever, so a hallucinated id degrades to the nearest real food
            # rather than to a silent zero-nutrient item.
            record, score = db.match(item.name, threshold=match_threshold)
            if record is None:
                grounded = retriever.ground(item.name)
                report.unmatched.append(item.name)
                if grounded is None:
                    continue
                record = grounded
                report.safety_fixes.append(
                    SafetyFix(
                        rule_id="verification.unmatched_food",
                        food_id=item.food_id,
                        name=item.name,
                        action="substitute",
                        replacement_food_id=record.fdc_id,
                        replacement_name=record.name,
                        message=(
                            f"{item.name!r} does not exist in the USDA database "
                            f"(best name match scored {score:.0f}); substituted the closest "
                            f"real food."
                        ),
                    )
                )
        else:
            report.matched += 1

        form = item.form if record.permits(item.form) else record.default_form
        if form != item.form:
            report.corrected.append(
                ItemCorrection(
                    food_id=record.fdc_id,
                    name=record.name,
                    field="form",
                    claimed=0.0,
                    corrected=0.0,
                    relative_error=0.0,
                    note=f"{item.form.value} is not a form this food takes; used {form.value}",
                )
            )

        quantity = max(item.quantity_g, 0.0)
        verified.append(
            MealItem(
                food_id=record.fdc_id,
                name=record.name,
                quantity_g=round(quantity, 1),
                form=form,
            )
        )

    # Compare the provider's nutrient claims against the recomputed truth.
    truth = db.nutrients_for(verified)
    if claimed_nutrients is not None:
        claimed = claimed_nutrients.as_dict()
        actual = truth.as_dict()
        for nutrient in CHECKED_NUTRIENTS:
            claim, real = claimed[nutrient], actual[nutrient]
            if real <= 0 and claim <= 0:
                continue
            denominator = max(abs(real), 1e-6)
            error = (claim - real) / denominator
            if abs(error) > tolerance:
                report.corrected.append(
                    ItemCorrection(
                        food_id="",
                        name="meal total",
                        field=nutrient,
                        claimed=round(claim, 2),
                        corrected=round(real, 2),
                        relative_error=round(error, 4),
                        note="corrected to the USDA recomputation",
                    )
                )

    # --- 4-5: safety scan and repair ---------------------------------------
    evaluation = engine.evaluate(verified, profile)
    for violation in evaluation.hard_violations:
        report.flagged.append(violation)

    if evaluation.hard_violations:
        verified = _repair(verified, profile, db, engine, report)

    final_evaluation = engine.evaluate(verified, profile)
    report.verified_nutrients = db.nutrients_for(verified)
    report.final_pass = final_evaluation.is_safe
    report.runtime_s = round(time.perf_counter() - started, 4)
    return Meal(items=verified), report


def _repair(
    items: list[MealItem],
    profile: UserProfile,
    db: FoodDB,
    engine: RuleEngine,
    report: VerificationReport,
) -> list[MealItem]:
    """Re-form or remove whatever is still unsafe.

    Re-forming first is not a preference, it is the cheaper repair: quartering the
    grapes keeps the food on the plate, removing them does not. Removal is what
    happens when no form is safe.
    """
    offending: set[str] = set()
    for violation in engine.evaluate(items, profile).hard_violations:
        offending.update(violation.offending_items)

    repaired: list[MealItem] = []
    for item in items:
        if item.food_id not in offending:
            repaired.append(item)
            continue

        safe_form = nearest_safe_form(item, profile, db)
        if safe_form is not None and safe_form != item.form:
            repaired.append(item.model_copy(update={"form": safe_form}))
            report.safety_fixes.append(
                SafetyFix(
                    rule_id="verification.reformed",
                    food_id=item.food_id,
                    name=item.name,
                    action="reform",
                    old_form=item.form,
                    new_form=safe_form,
                    message=(
                        f"{item.name} was unsafe as {item.form.value}; "
                        f"served {safe_form.value} instead."
                    ),
                )
            )
            continue

        report.safety_fixes.append(
            SafetyFix(
                rule_id="verification.removed",
                food_id=item.food_id,
                name=item.name,
                action="remove",
                old_form=item.form,
                message=(f"{item.name} has no safe preparation for this profile and was removed."),
            )
        )

    # A repair can uncover a second violation -- re-forming one item does not make
    # another safe. One more pass catches that; beyond two the meal is genuinely
    # unfixable and `final_pass` will say so rather than looping.
    if repaired != items and engine.evaluate(repaired, profile).hard_violations:
        still_offending: set[str] = set()
        for violation in engine.evaluate(repaired, profile).hard_violations:
            still_offending.update(violation.offending_items)
        survivors = []
        for item in repaired:
            if item.food_id in still_offending:
                report.safety_fixes.append(
                    SafetyFix(
                        rule_id="verification.removed",
                        food_id=item.food_id,
                        name=item.name,
                        action="remove",
                        old_form=item.form,
                        message=f"{item.name} remained unsafe after re-forming; removed.",
                    )
                )
                continue
            survivors.append(item)
        repaired = survivors

    return repaired
