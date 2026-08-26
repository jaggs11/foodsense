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

Part B reports its faults in two blocks, because pooling them would overstate the
result. Three of the six are caught *by construction* -- an id absent from the
database fails a dictionary lookup, a form drawn from the complement of a food's
allowed forms fails a membership test, a 1.9x claim against a 10% tolerance is
outside it by arithmetic -- and their 100% rates measure that the guards are
wired up, not that the verifier is capable. The other three require independent
re-derivation and are the honest headline.

    python experiments/run_verification_eval.py
    python experiments/run_verification_eval.py --cases 40 --providers template anthropic
    python experiments/run_verification_eval.py --fault-cases 300
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


@dataclass(frozen=True, slots=True)
class Fault:
    """One injected fault, and an honest statement of how hard it is to catch."""

    key: str
    #: ``construction`` -- the check and the injection consult the same fact, so
    #: detection cannot fail and a 100% rate carries no information about
    #: capability. ``rederivation`` -- the verifier has to independently recompute
    #: or re-derive something to notice, and could genuinely miss it.
    kind: str
    injected: str
    why: str
    #: Why it sits in that class. Written down because the classification is the
    #: part a reader should be able to disagree with.
    because: str


#: Splitting these is the difference between a headline that survives scrutiny and
#: one that does not. Two of the five faults are caught by definition: an id that
#: is not in the database fails a dictionary lookup, and a 1.9x claim against a
#: 10% tolerance is outside it by arithmetic. Reporting those in one pooled
#: "100% detected" number with the two that require real work would be padding.
FAULT_SPECS = (
    Fault(
        key="hallucinated_food",
        kind="construction",
        injected='An item replaced by a plausible name with the fake id "9999999"',
        why="The classic failure: a food that sounds real and is not",
        because=(
            "`db.find` is a dictionary lookup on a fixed key set. The id is absent, "
            "so the lookup fails every time. No recomputation is involved."
        ),
    ),
    Fault(
        key="impossible_form",
        kind="construction",
        injected="A preparation drawn from the complement of the food's allowed forms",
        why="Forms are text to a model; nothing stops it writing 'pureed steak'",
        because=(
            "The injector picks a form *because* it is absent from "
            "`record.allowed_forms`, and the verifier's check is membership in that "
            "same tuple. It is one table read against its own complement."
        ),
    ),
    Fault(
        key="inflated_claim",
        kind="construction",
        injected="Nutrient totals overstated by 1.9x against a 10% tolerance",
        why="Models assert nutrition figures confidently and wrongly",
        because=(
            "90% over a 10% bound cannot land inside it. The magnitude is chosen "
            "far outside the tolerance, so the comparison is decided before it runs."
        ),
    ),
    Fault(
        key="quantity_drift_marginal",
        kind="rederivation",
        injected="One item's quantity multiplied by 1.10-1.15x",
        why="The floor of the measurement: a drift at or just past the tolerance itself",
        because=(
            "Reported separately because much of this band cannot be caught and it "
            "would otherwise drag the honest number down. The tolerance is 10% of "
            "the *meal total*; a 10-15% drift on one item moves the total by that "
            "fraction times the item's share of the plate, which is below 10% for "
            "every item that is not almost the whole meal. Detections here are the "
            "cases where the drifted item dominated the plate."
        ),
    ),
    Fault(
        key="quantity_drift_near",
        kind="rederivation",
        injected="One item's quantity multiplied by 1.15-1.3x",
        why="The realistic version: a small restatement, not an obvious one",
        because=(
            "Claimed nutrients are computed *before* the drift, so catching it "
            "requires recomputing the meal from the database and comparing. This is "
            "the band where detection genuinely turns over: large enough that a "
            "drifted item of ordinary size can push the meal total past the "
            "tolerance, small enough that it often does not."
        ),
    ),
    Fault(
        key="quantity_drift_far",
        kind="rederivation",
        injected="One item's quantity multiplied by 1.6-3.0x",
        why="Models restate amounts from memory and drift",
        because=(
            "Same mechanism as the near band. Larger drifts move the meal total "
            "further outside the tolerance, so this is the easy end of the same "
            "measurement, reported separately rather than pooled with it."
        ),
    ),
    Fault(
        key="reintroduced_hazard",
        kind="rederivation",
        injected="A repaired item put back in its unsafe form, or an unsafe food added",
        why="A rewrite can undo the optimiser's safety fix without noticing",
        because=(
            "Nothing in the item marks it as unsafe. The verifier has to re-run the "
            "full `RuleEngine` -- age config, hazard class, form, age in months -- "
            "and re-derive the violation from scratch. This is the fault the whole "
            "extension exists for."
        ),
    ),
)

FAULTS = tuple(f.key for f in FAULT_SPECS)
FAULT_BY_KEY = {f.key: f for f in FAULT_SPECS}

#: Multipliers for the two drift bands, so the near band's boundary is stated in
#: one place rather than buried in the injector.
DRIFT_BANDS = {
    "quantity_drift_marginal": (1.10, 1.15),
    "quantity_drift_near": (1.15, 1.30),
    "quantity_drift_far": (1.60, 3.00),
}


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
    #: Mean relative shift the fault caused in the meal's *total* energy. Recorded
    #: because that -- not the per-item multiplier -- is the quantity the 10%
    #: tolerance is applied to, and it is the reason a visible per-item drift can
    #: be invisible at the meal level.
    total_shifts: list[float] = field(default_factory=list)

    @property
    def mean_total_shift(self) -> float:
        return statistics.fmean(self.total_shifts) if self.total_shifts else 0.0


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

    if fault in DRIFT_BANDS:
        low, high = DRIFT_BANDS[fault]
        index = rng.randrange(len(items))
        items[index] = items[index].model_copy(
            update={"quantity_g": round(items[index].quantity_g * rng.uniform(low, high), 1)}
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
    """Inject each fault into a clean Stage-3 output and score the verifier on it.

    ``n`` is larger than study A's on purpose. Choking bans exist only for
    toddlers, so under the equal-share stratification only a third of the cases
    can carry a re-introduced hazard at all -- and that is the fault class this
    extension exists to catch, so it is the one that must not be thin.
    """
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
            if fault in DRIFT_BANDS:
                clean_energy = clean_nutrients.as_dict()["energy_kcal"]
                drifted_energy = db.nutrients_for(corrupted).as_dict()["energy_kcal"]
                if clean_energy > 0:
                    outcome.total_shifts.append(abs(drifted_energy - clean_energy) / clean_energy)
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
                or (fault in DRIFT_BANDS and not detected)
                or (fault == "inflated_claim" and not detected)
            )
            outcome.reached_user += int(survived)
    return list(outcomes.values())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    return "n/a" if not denominator else f"{numerator / denominator * 100:.0f}%"


def render(
    observed: list[ProviderOutcome], faults: list[FaultOutcome], n: int, n_faults: int
) -> str:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# Stage-4 verification: what it catches",
        "",
        f"Generated by `experiments/run_verification_eval.py` at {stamp}: {n} cases in",
        f"study A, {n_faults} age-stratified cases in study B.",
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
        "## B. Fault injection: what Stage 4 catches when something *is* wrong",
        "",
        "Faults of the kinds a language model actually produces are injected into Stage-3",
        "output and the verifier is scored on them. **Every fault here is deliberately",
        "injected**; these rates describe a capability, not the observed error rate of any",
        "real model.",
        "",
        "The faults are reported in two blocks, because a single pooled detection rate",
        "would overstate what this measures. Some faults are caught *by construction* --",
        "the check and the injection consult the same fact, so detection cannot fail and",
        "the resulting 100% says nothing about capability. The rest require the verifier",
        "to independently recompute or re-derive something, and could genuinely miss.",
        "**The second block is the real headline.** It is the smaller number and it is the",
        "one that survives scrutiny.",
        "",
    ]

    header = "| Injected fault | Cases | Detected | Repaired | **Reached the user** |"

    def _block(kind: str) -> tuple[int, int, int]:
        """Render one block and return its (cases, detected, reached-user) subtotal."""
        block_n = block_detected = block_reached = 0
        lines.append(header)
        lines.append("|---|---|---|---|---|")
        for outcome in faults:
            if FAULT_BY_KEY[outcome.fault].kind != kind:
                continue
            if not outcome.n:
                lines.append(f"| `{outcome.fault}` | 0 | n/a | n/a | n/a |")
                continue
            block_n += outcome.n
            block_detected += outcome.detected
            block_reached += outcome.reached_user
            lines.append(
                f"| `{outcome.fault}` | {outcome.n} | {_pct(outcome.detected, outcome.n)} "
                f"| {_pct(outcome.repaired, outcome.n)} "
                f"| **{_pct(outcome.reached_user, outcome.n)}** |"
            )
        return block_n, block_detected, block_reached

    lines += ["### B1. Detected by construction -- not evidence of capability", ""]
    c_n, c_detected, c_reached = _block("construction")
    if c_n:
        lines.append(
            f"| **subtotal** | {c_n} | **{_pct(c_detected, c_n)}** | - "
            f"| **{_pct(c_reached, c_n)}** |"
        )
    lines += [
        "",
        "These three are worth running -- a regression that broke the id lookup or the",
        "form check would show up here immediately -- but their detection rates are",
        "arithmetic, not measurement. Read them as assertions that the guards are wired",
        "up, and read the next block for what verification is actually worth.",
        "",
        "### B2. Detected by re-derivation -- this is the real number",
        "",
        "Here the verifier has to do independent work: recompute the meal's nutrients",
        "from the database and compare against a claim made *before* the corruption, or",
        "re-run the whole `RuleEngine` to re-derive a hazard from the food, its form and",
        "the profile's age in months. Nothing in the corrupted item announces itself.",
        "",
    ]
    r_n, r_detected, r_reached = _block("rederivation")
    if r_n:
        lines.append(
            f"| **subtotal** | {r_n} | **{_pct(r_detected, r_n)}** | - "
            f"| **{_pct(r_reached, r_n)}** |"
        )
        lines += [
            "",
            "**Read that subtotal with the same suspicion as B1's.** It is a pooled figure",
            "over faults of deliberately different difficulty, so its value is partly a",
            "statement about how many rows were put in each band rather than about the",
            "verifier. The marginal drift band is in there precisely because much of it",
            "cannot be caught at a 10% meal-level tolerance, and it drags the pool down the",
            "same way the by-construction faults dragged B1's up.",
            "",
            "The numbers that actually mean something are the per-fault rows: the detection",
            "curve across the three drift magnitudes below, and `reintroduced_hazard`, which",
            "is the fault this extension exists to catch and is not a pooled number at all.",
        ]

    marginal = next((o for o in faults if o.fault == "quantity_drift_marginal"), None)
    near = next((o for o in faults if o.fault == "quantity_drift_near"), None)
    far = next((o for o in faults if o.fault == "quantity_drift_far"), None)
    hazard = next((o for o in faults if o.fault == "reintroduced_hazard"), None)

    lines += ["", "### Where quantity detection begins", ""]
    bands = [b for b in (marginal, near, far) if b is not None and b.n]
    if bands:
        lines += [
            "The drift bands are one mechanism at three magnitudes. Splitting them is not",
            "presentational: the tolerance is 10% of the **meal total**, while the fault is",
            "a multiplier on **one item**, so the quantity that actually decides detection",
            "is the multiplier scaled by that item's share of the plate. A 1.1x drift on a",
            "garnish is a fraction of a percent at the meal level and cannot be caught by a",
            "10% bound at all. Reporting 1.1-1.3x as one number therefore mixes a regime",
            "that is undetectable by construction into one that is genuinely being measured.",
            "",
            "| Band | Per-item multiplier | Mean shift in meal total | Cases | Detected |",
            "|---|---|---|---|---|",
        ]
        for band in bands:
            low, high = DRIFT_BANDS[band.fault]
            lines.append(
                f"| `{band.fault.rsplit('_', 1)[-1]}` | {low:.2f}-{high:.2f}x "
                f"| {band.mean_total_shift * 100:.1f}% | {band.n} "
                f"| {_pct(band.detected, band.n)} |"
            )
        lines += [
            "",
            "The **mean shift in meal total** column is the one to read against the 10%",
            "tolerance. Where it sits below 10%, a miss is the tolerance doing its job, not",
            "the verifier failing: a sub-tolerance discrepancy is by definition one this",
            "layer has decided not to flag, because flagging it would mean flagging",
            "legitimate rounding too.",
            "",
            "Raising detection in the low bands means tightening `nutrient_tolerance` or",
            "moving the check per item rather than per meal. Both trade directly against",
            "false positives, and neither is obviously worth it for a discrepancy smaller",
            "than the error in the food match itself.",
            "",
        ]
    else:
        lines += ["Not measured in this run.", ""]

    if hazard and hazard.n:
        lines += [
            "### `reintroduced_hazard`, the fault the extension exists for",
            "",
            f"Measured over {hazard.n} cases: {_pct(hazard.detected, hazard.n)} detected,",
            f"{_pct(hazard.reached_user, hazard.n)} reaching the user. This is the case where a",
            "generative step silently undoes a safety decision the optimiser already made.",
            "Stage 4 re-runs the same `RuleEngine` on the final item list precisely so that a",
            "rewrite cannot be the last word.",
            "",
            "It is measured only on toddler cases, and that is not a sampling accident: the",
            "choking bans are toddler rules, so no other profile admits the fault at all.",
            "Study B stratifies by age group and runs more cases than study A specifically to",
            "keep this cell from being thin.",
            "",
        ]

    lines += [
        "### What each fault is, and why it lands in the block it does",
        "",
        "| Fault | Block | What is injected | Why a generator would do it |",
        "|---|---|---|---|",
    ]
    for spec in FAULT_SPECS:
        block = "construction" if spec.kind == "construction" else "**re-derivation**"
        lines.append(f"| `{spec.key}` | {block} | {spec.injected} | {spec.why} |")

    lines += ["", "**Why each classification:**", ""]
    for spec in FAULT_SPECS:
        lines.append(f"- `{spec.key}` -- {spec.because}")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=90, help="cases for study A")
    parser.add_argument(
        "--fault-cases",
        type=int,
        default=150,
        help=(
            "cases for study B. Larger than study A on purpose: the sample is "
            "stratified by age group and only the toddler third can carry a "
            "re-introduced hazard, so 150 keeps that cell at 50 rather than 30."
        ),
    )
    parser.add_argument("--providers", nargs="*", default=list(PROVIDERS))
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db, engine = get_food_db(), RuleEngine()

    print("Study A: observed rates by provider")
    observed = study_observed(db, engine, args.providers, args.cases)
    print("\nStudy B: fault injection")
    faults = study_faults(db, engine, args.fault_cases)

    markdown = render(observed, faults, args.cases, args.fault_cases)
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
                "n_fault_cases": args.fault_cases,
                "fault_kinds": {f.key: f.kind for f in FAULT_SPECS},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{'fault':<22} {'kind':<13} {'cases':>6} {'detected':>9} {'repaired':>9} {'reached user':>13}"
    )
    for outcome in faults:
        print(
            f"{outcome.fault:<22} {FAULT_BY_KEY[outcome.fault].kind:<13} {outcome.n:>6} "
            f"{_pct(outcome.detected, outcome.n):>9} "
            f"{_pct(outcome.repaired, outcome.n):>9} {_pct(outcome.reached_user, outcome.n):>13}"
        )
    print(f"\nwrote {RESULTS_DIR / 'verification_eval.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
