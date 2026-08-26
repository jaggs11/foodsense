"""Stage-4 impact -> ``results/verification_eval.md``.

Two studies, because one of them alone would be misleading.

**A. Observed rates by provider.** Run the pipeline over sampled cases with each
available Stage-3 provider and count how many outputs contain a quantity that
disagrees with the database, a food that does not exist, or a surviving safety
violation -- before Stage 4 and after it. The deterministic template provider
passes the optimiser's own items straight through, so its "before" rate is
near zero *by construction*. That is a real result and it is reported as one, but
on its own it would make Stage 4 look like it does nothing, when what it actually
shows is that the offline path has nothing to correct.

**B. Fault injection.** The verifier's job is to catch what a generator gets
wrong, so it is measured against generators that get things wrong. Faults of the
kinds an LLM actually produces -- a drifted quantity, a hallucinated food, a
preparation the food cannot take, an inflated nutrient claim, a re-introduced
hazard -- are injected into Stage-3 output and Stage 4 is scored on how many it
catches and repairs.

Every injected fault is labelled as injected. Part B measures a capability under
controlled conditions; it is not a claim about how often real LLMs err.

    python experiments/run_verification_eval.py
    python experiments/run_verification_eval.py --cases 40 --providers template anthropic
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodsense import RESULTS_DIR, SEED
from foodsense.constraints.engine import RuleEngine
from foodsense.data.corpora import load_meals
from foodsense.data.fdc import FoodDB, get_food_db
from foodsense.pipeline import run_pipeline
from foodsense.schemas import AgeGroup, Form, Meal, MealItem, UserProfile
from foodsense.stage1_prediction.labels import sample_profile
from foodsense.stage3_rag.providers import PROVIDERS, get_provider
from foodsense.stage4_verification.verifier import verify

PANTRY_SIZE = 8
PANTRY_CATEGORIES = (
    "vegetable",
    "fruit",
    "grain",
    "legume",
    "dairy",
    "poultry",
    "meat",
    "fish",
    "cereal",
)

FAULTS = (
    "quantity_drift",
    "hallucinated_food",
    "impossible_form",
    "inflated_claim",
    "reintroduced_hazard",
)


@dataclass(slots=True)
class ProviderOutcome:
    """One provider's observed error rates, before and after verification."""

    provider: str
    n: int = 0
    available: bool = True
    reason: str = ""
    before_bad_quantity: int = 0
    before_unmatched: int = 0
    before_unsafe: int = 0
    after_unsafe: int = 0
    corrections: int = 0
    safety_fixes: int = 0
    fallbacks: int = 0
    runtimes: list[float] = field(default_factory=list)

    @property
    def before_any(self) -> int:
        return self.before_bad_quantity + self.before_unmatched + self.before_unsafe


@dataclass(slots=True)
class FaultOutcome:
    """How Stage 4 handled one class of injected fault."""

    fault: str
    n: int = 0
    detected: int = 0
    repaired: int = 0
    reached_user: int = 0


# ---------------------------------------------------------------------------
# Study A: observed rates
# ---------------------------------------------------------------------------


def _cases(db: FoodDB, n: int, seed: int = SEED, stratify: bool = False):
    """Sample cases. ``stratify`` gives each age group an equal share.

    Study B needs that: choking bans only exist for toddlers, so an unstratified
    sample yields very few meals in which a hazard can be re-introduced at all,
    and the most important fault class ends up measured on a handful of cases.
    """
    meals = [m.meal for m in load_meals("foodcom", limit=n * 2, rows=n * 12)]
    pool = [r for r in db.records if r.category in PANTRY_CATEGORIES]
    groups = list(AgeGroup)
    rng = random.Random(seed)
    for i in range(n):
        age_group = groups[i % len(groups)] if stratify else None
        yield (
            sample_profile(random.Random(seed + i), age_group),
            meals[i % len(meals)],
            Meal(items=[rng.choice(pool).as_item(0.0) for _ in range(PANTRY_SIZE)]),
        )


def study_observed(
    db: FoodDB, engine: RuleEngine, providers: list[str], n: int
) -> list[ProviderOutcome]:
    outcomes: list[ProviderOutcome] = []
    for name in providers:
        provider = get_provider(name)
        outcome = ProviderOutcome(provider=name, available=provider.available)
        if not provider.available:
            outcome.reason = provider.unavailable_reason()
            outcomes.append(outcome)
            print(f"  {name}: unavailable ({outcome.reason}) -- skipped")
            continue

        print(f"  {name}: running {n} cases")
        for profile, planned, pantry in _cases(db, n):
            trace = run_pipeline(profile, planned, pantry, provider=provider, db=db, engine=engine)
            stage3, report = trace.stage3, trace.stage4
            if stage3 is None or report is None:
                continue
            outcome.n += 1
            outcome.fallbacks += int(stage3.fallback_used)

            # "Before": what Stage 3 handed over, judged on its own.
            before = engine.evaluate(stage3.items, profile)
            outcome.before_unsafe += int(bool(before.hard_violations))
            outcome.before_unmatched += int(bool(report.unmatched))
            outcome.before_bad_quantity += int(any(c.field != "form" for c in report.corrected))
            outcome.after_unsafe += int(not report.final_pass)
            outcome.corrections += len(report.corrected)
            outcome.safety_fixes += len(report.safety_fixes)
            outcome.runtimes.append(report.runtime_s)
        outcomes.append(outcome)
    return outcomes


# ---------------------------------------------------------------------------
# Study B: fault injection
# ---------------------------------------------------------------------------


def inject(
    fault: str, items: list[MealItem], profile: UserProfile, db: FoodDB, rng: random.Random
) -> tuple[list[MealItem], bool]:
    """Corrupt a Stage-3 output the way a generator plausibly would.

    Returns the corrupted items and whether the fault could actually be applied --
    not every meal admits every fault (a meal with no re-formable hazard cannot
    have one re-introduced), and counting an unapplied fault as "caught" would
    inflate the result.
    """
    items = [i.model_copy() for i in items]
    if not items:
        return items, False

    if fault == "quantity_drift":
        index = rng.randrange(len(items))
        items[index] = items[index].model_copy(
            update={"quantity_g": round(items[index].quantity_g * rng.uniform(1.6, 3.0), 1)}
        )
        return items, True

    if fault == "hallucinated_food":
        index = rng.randrange(len(items))
        items[index] = MealItem(
            food_id="9999999",
            name="Artisanal quinoa-kale power blend, house recipe",
            quantity_g=items[index].quantity_g,
            form=Form.WHOLE,
        )
        return items, True

    if fault == "impossible_form":
        for index, item in enumerate(items):
            record = db.find(item.food_id)
            if record is None:
                continue
            options = [f for f in Form if f not in record.allowed_forms]
            if options:
                items[index] = item.model_copy(update={"form": rng.choice(options)})
                return items, True
        return items, False

    if fault == "inflated_claim":
        # Handled by the caller, which controls the claimed nutrient vector.
        return items, True

    if fault == "reintroduced_hazard":
        from foodsense.constraints.age_rules import load_age_config

        bans = {b.hazard_class: b for b in load_age_config(profile.age_group).active_bans()}
        if not bans:
            return items, False
        for index, item in enumerate(items):
            record = db.find(item.food_id)
            if record is None or record.hazard_class not in bans:
                continue
            ban = bans[record.hazard_class]
            unsafe = [f for f in record.allowed_forms if ban.bans(f, profile.age_months)]
            if unsafe:
                items[index] = item.model_copy(update={"form": unsafe[0]})
                return items, True

        # This meal holds nothing that can be made unsafe -- usually because the
        # optimiser already took the hazard out. Add one instead: the failure
        # under test is "a generative step put an unsafe food on the plate", and
        # it does not matter whether that food was there beforehand. Without this
        # the most important fault class is measured on a dozen cases.
        for hazard_class, ban in bans.items():
            for record in db.by_hazard(hazard_class):
                unsafe = [f for f in record.allowed_forms if ban.bans(f, profile.age_months)]
                if unsafe:
                    items.append(record.as_item(30.0, unsafe[0]))
                    return items, True
        return items, False

    raise ValueError(f"Unknown fault {fault!r}")


def study_faults(db: FoodDB, engine: RuleEngine, n: int) -> list[FaultOutcome]:
    outcomes = {fault: FaultOutcome(fault=fault) for fault in FAULTS}
    provider = get_provider("template")
    rng = random.Random(SEED)

    for profile, planned, pantry in _cases(db, n, seed=SEED + 977, stratify=True):
        trace = run_pipeline(profile, planned, pantry, provider=provider, db=db, engine=engine)
        if trace.stage3 is None or not trace.stage3.items:
            continue
        clean_items = trace.stage3.items
        clean_nutrients = db.nutrients_for(clean_items)

        for fault in FAULTS:
            corrupted, applied = inject(fault, clean_items, profile, db, rng)
            if not applied:
                continue
            claimed = clean_nutrients.scaled(1.9) if fault == "inflated_claim" else clean_nutrients
            final, report = verify(
                corrupted, profile, claimed_nutrients=claimed, db=db, engine=engine
            )

            outcome = outcomes[fault]
            outcome.n += 1
            detected = bool(report.corrected or report.safety_fixes or report.unmatched)
            outcome.detected += int(detected)
            outcome.repaired += int(report.final_pass and detected)
            # Did the fault survive all the way to the user?
            survived = (
                (fault == "hallucinated_food" and any(i.food_id == "9999999" for i in final.items))
                or (fault == "reintroduced_hazard" and not report.final_pass)
                or (
                    fault == "impossible_form"
                    and any(
                        (r := db.find(i.food_id)) is not None and not r.permits(i.form)
                        for i in final.items
                    )
                )
                or (fault == "quantity_drift" and not detected)
                or (fault == "inflated_claim" and not detected)
            )
            outcome.reached_user += int(survived)
    return list(outcomes.values())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    return "n/a" if not denominator else f"{numerator / denominator * 100:.0f}%"


def render(observed: list[ProviderOutcome], faults: list[FaultOutcome], n: int) -> str:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# Stage-4 verification: what it catches",
        "",
        f"Generated by `experiments/run_verification_eval.py` at {stamp}, {n} cases per study.",
        "Every number came from a real run. Regenerate with `make eval`.",
        "",
        "## A. Observed error rates by Stage-3 provider",
        "",
        "Share of pipeline runs whose Stage-3 output contained at least one problem of",
        "each kind, before verification, and the share still unsafe after it.",
        "",
        "| Provider | Runs | Bad quantity | Non-existent food | Unsafe before | **Unsafe after** | Corrections | Safety fixes | Fell back |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for outcome in observed:
        if not outcome.available:
            lines.append(f"| {outcome.provider} | - | - | - | - | - | - | - | _{outcome.reason}_ |")
            continue
        lines.append(
            f"| {outcome.provider} | {outcome.n} "
            f"| {_pct(outcome.before_bad_quantity, outcome.n)} "
            f"| {_pct(outcome.before_unmatched, outcome.n)} "
            f"| {_pct(outcome.before_unsafe, outcome.n)} "
            f"| **{_pct(outcome.after_unsafe, outcome.n)}** "
            f"| {outcome.corrections} | {outcome.safety_fixes} "
            f"| {_pct(outcome.fallbacks, outcome.n)} |"
        )

    lines += [
        "",
        "**Reading this honestly.** The template provider emits the optimiser's own items",
        "unchanged, so there is nothing for Stage 4 to correct and its rates are near zero.",
        "That is not evidence that verification is unnecessary -- it is evidence that the",
        "offline path is already exact. Verification exists for the path where a generative",
        "model rewrites the meal, which is what study B measures directly.",
        "",
        "## B. Fault injection: what Stage 4 catches when something *is* wrong",
        "",
        "Faults of the kinds a language model actually produces are injected into Stage-3",
        "output and the verifier is scored on them. **Every fault here is deliberately",
        "injected**; these rates describe a capability, not the observed error rate of any",
        "real model.",
        "",
        "| Injected fault | Cases | Detected | Repaired | **Reached the user** |",
        "|---|---|---|---|---|",
    ]
    total_n = total_detected = total_reached = 0
    for outcome in faults:
        if not outcome.n:
            lines.append(f"| `{outcome.fault}` | 0 | n/a | n/a | n/a |")
            continue
        total_n += outcome.n
        total_detected += outcome.detected
        total_reached += outcome.reached_user
        lines.append(
            f"| `{outcome.fault}` | {outcome.n} | {_pct(outcome.detected, outcome.n)} "
            f"| {_pct(outcome.repaired, outcome.n)} "
            f"| **{_pct(outcome.reached_user, outcome.n)}** |"
        )
    if total_n:
        lines.append(
            f"| **all faults** | {total_n} | **{_pct(total_detected, total_n)}** | - "
            f"| **{_pct(total_reached, total_n)}** |"
        )

    lines += [
        "",
        "### What each fault is",
        "",
        "| Fault | What is injected | Why a generator would do it |",
        "|---|---|---|",
        "| `quantity_drift` | An item's quantity multiplied by 1.6-3.0x | Models restate amounts from memory and drift |",
        "| `hallucinated_food` | An item replaced by a plausible name with a fake id | The classic failure: a food that sounds real and is not |",
        "| `impossible_form` | A preparation the food cannot take | Forms are text to a model; nothing stops it writing 'pureed steak' |",
        "| `inflated_claim` | Nutrient totals overstated by 1.9x | Models assert nutrition figures confidently and wrongly |",
        "| `reintroduced_hazard` | A repaired item put back in its unsafe form | A rewrite can undo the optimiser's safety fix without noticing |",
        "",
        "`reintroduced_hazard` is the one that matters most: it is the case where a",
        "generative step silently undoes a safety decision the optimiser made. Stage 4",
        "re-runs the same `RuleEngine` on the final list precisely so that a rewrite cannot",
        "be the last word.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=90)
    parser.add_argument("--providers", nargs="*", default=list(PROVIDERS))
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db, engine = get_food_db(), RuleEngine()

    print("Study A: observed rates by provider")
    observed = study_observed(db, engine, args.providers, args.cases)
    print("\nStudy B: fault injection")
    faults = study_faults(db, engine, args.cases)

    markdown = render(observed, faults, args.cases)
    (RESULTS_DIR / "verification_eval.md").write_text(markdown, encoding="utf-8")
    (RESULTS_DIR / "verification_eval.json").write_text(
        json.dumps(
            {
                "observed": [
                    dataclasses.asdict(o)
                    | {
                        "runtime_mean_s": (
                            round(statistics.fmean(o.runtimes), 5) if o.runtimes else None
                        )
                    }
                    for o in observed
                ],
                "faults": [dataclasses.asdict(f) for f in faults],
                "n_cases": args.cases,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\n{'fault':<22} {'cases':>6} {'detected':>9} {'repaired':>9} {'reached user':>13}")
    for outcome in faults:
        print(
            f"{outcome.fault:<22} {outcome.n:>6} {_pct(outcome.detected, outcome.n):>9} "
            f"{_pct(outcome.repaired, outcome.n):>9} {_pct(outcome.reached_user, outcome.n):>13}"
        )
    print(f"\nwrote {RESULTS_DIR / 'verification_eval.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
