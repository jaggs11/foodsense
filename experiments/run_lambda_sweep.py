"""Sensitivity of validity to ``lambda_validity`` -> ``results/lambda_sweep.{md,csv}``.

The comparison table reports FoodSense's validity against five baselines and the
number is lower than most of them. This experiment is the answer to the obvious
question about that number, and the answer is not "we could have tuned it higher"
offered as an excuse -- it is the measurement, run across the whole range, showing
exactly what moves and what does not.

What moves: validity, edit count and distance, all together, because they are the
same trade-off seen from three sides. A larger ``lambda_validity`` buys guideline
compliance with edits the user has to actually perform.

What does not move: safety. The hard-safety penalty is a separate term with a
weight three orders of magnitude larger, and structural availability is not a term
at all -- an unavailable food has no decision variable. Those are properties of
the formulation, so no setting of this weight can trade them away. That is the
claim this sweep exists to support, and it is falsifiable: a run that showed a
safety violation at any setting would refute it.

**Provenance.** ``lambda_validity: 5.0`` was chosen in Phase 3 from a sweep of the
same shape, before any comparison table against the baselines existed, on the
grounds that it was the smallest weight that materially improved the meal while
keeping edits near the two-edit scale of the proposal's worked example. It has not
been re-tuned since -- deliberately, because tuning a hyperparameter after seeing
the evaluation it will be reported against is fitting to the evaluation set. This
script re-measures the sweep on the shipped code so the artefact is reproducible
rather than a table transcribed into a comment.

    python experiments/run_lambda_sweep.py
    python experiments/run_lambda_sweep.py --cases 5 --lambdas 1 5 8
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodsense import RESULTS_DIR, SEED
from foodsense.constraints.engine import RuleEngine
from foodsense.data.corpora import load_meals
from foodsense.data.fdc import FoodDB, get_food_db
from foodsense.schemas import AgeGroup, Meal
from foodsense.stage1_prediction.labels import sample_profile
from foodsense.stage1_prediction.predict import get_suitability_model
from foodsense.stage2_optimizer.de_optimizer import DEConfig, differential_evolution
from foodsense.stage2_optimizer.objective import CounterfactualObjective, ObjectiveConfig
from foodsense.stage2_optimizer.space import build_space

#: The sweep points. 5.0 is the shipped default and is marked as such in the
#: output; the rest bracket it by roughly an order of magnitude either way.
DEFAULT_LAMBDAS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)

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
    "nut_seed",
    "cereal",
)


@dataclass(slots=True)
class SweepRow:
    """One ``lambda_validity`` setting, aggregated over every case."""

    lambda_validity: float
    n: int
    validity: float
    safe_rate: float
    availability_violation: float
    score_before: float
    score_after: float
    soft_before: float
    soft_after: float
    edits: float
    norm_l1: float
    runtime_s: float

    @property
    def delta(self) -> float:
        return self.soft_after - self.soft_before


def build_cases(db: FoodDB, n_per_group: int, seed: int = SEED):
    """Same construction as ``run_cf_eval.py``: a corpus meal, a profile, a pantry.

    Seeded identically so the cases are drawn from the same distribution the
    comparison table is measured on, and reproducible across invocations.
    """
    meals = [m.meal for m in load_meals("foodcom", limit=n_per_group * 2, rows=n_per_group * 12)]
    if not meals:
        raise RuntimeError("No corpus meals available; run `make data` first.")
    pool = [r for r in db.records if r.category in PANTRY_CATEGORIES]
    rng = random.Random(seed)
    for age_group in AgeGroup:
        for i in range(n_per_group):
            yield (
                sample_profile(random.Random(seed + i), age_group),
                meals[i % len(meals)],
                Meal(items=[rng.choice(pool).as_item(0.0) for _ in range(PANTRY_SIZE)]),
            )


def run_setting(
    lambda_validity: float,
    cases: list,
    db: FoodDB,
    engine: RuleEngine,
    model,
    base: ObjectiveConfig,
    de_config: DEConfig,
) -> SweepRow:
    """Optimise every case at one weight and aggregate."""
    config = dataclasses.replace(base, lambda_validity=lambda_validity)
    started = time.perf_counter()

    valid = safe = unavailable = 0
    before, after, soft_before, soft_after, edits, norm_l1 = [], [], [], [], [], []

    for index, (profile, planned, pantry) in enumerate(cases, start=1):
        space = build_space(planned, pantry, db, profile=profile)
        if not space.variables:
            continue
        objective = CounterfactualObjective(space, profile, model, engine, config)
        result = differential_evolution(
            objective, space, profile, engine, de_config, seed=SEED + index
        )

        planned_eval = engine.evaluate(planned, profile)
        final_eval = engine.evaluate(result.meal, profile)
        planned_mass = max(sum(i.quantity_g for i in planned.items), 1.0)

        valid += int(final_eval.is_safe and final_eval.score >= config.target_score)
        safe += int(final_eval.is_safe)
        unavailable += int(bool(space.unavailable_items(result.meal)))
        before.append(planned_eval.score)
        after.append(final_eval.score)
        soft_before.append(planned_eval.soft_score)
        soft_after.append(final_eval.soft_score)
        edits.append(result.terms.n_changed)
        norm_l1.append(result.terms.l1_g / planned_mass)

    n = len(after)
    return SweepRow(
        lambda_validity=lambda_validity,
        n=n,
        validity=valid / n if n else 0.0,
        safe_rate=safe / n if n else 0.0,
        availability_violation=unavailable / n if n else 0.0,
        score_before=statistics.fmean(before) if n else 0.0,
        score_after=statistics.fmean(after) if n else 0.0,
        soft_before=statistics.fmean(soft_before) if n else 0.0,
        soft_after=statistics.fmean(soft_after) if n else 0.0,
        edits=statistics.fmean(edits) if n else 0.0,
        norm_l1=statistics.fmean(norm_l1) if n else 0.0,
        runtime_s=time.perf_counter() - started,
    )


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render(rows: list[SweepRow], shipped: float, n_per_group: int) -> str:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    low, high = rows[0], rows[-1]
    best = max(rows, key=lambda r: r.validity)
    unsafe = [r for r in rows if r.safe_rate < 1.0]
    unavailable = [r for r in rows if r.availability_violation > 0.0]

    lines = [
        "# Sensitivity analysis: `lambda_validity`",
        "",
        f"Generated by `experiments/run_lambda_sweep.py` at {stamp}, "
        f"{n_per_group} cases per age group ({rows[0].n} total per setting).",
        "Every figure came from a real run. Regenerate with",
        "`python experiments/run_lambda_sweep.py`.",
        "",
        "## The claim",
        "",
        f"**Validity moves from {_pct(low.validity)} to {_pct(best.validity)} across this sweep.",
        f"Safety is {_pct(min(r.safe_rate for r in rows))} at every setting.**",
        "",
        "Validity is a tunable trade-off. Safety is not a setting.",
        "",
        f"`lambda_validity: {shipped:g}` is the shipped default in `configs/pipeline.yaml`.",
        "It was chosen in Phase 3 from a sweep of this shape, **before** any comparison",
        "against the baselines had been run, as the smallest weight that materially",
        "improves the meal while keeping the edit count near the two-edit scale of the",
        "proposal's worked example. It has not been re-tuned since the comparison table",
        "was produced, and deliberately so: choosing a hyperparameter after seeing the",
        "evaluation it will be reported against is fitting to the evaluation set. The",
        "row is marked below rather than being the row the sweep was built to flatter.",
        "",
        "## Results",
        "",
        "| `lambda_validity` | Validity | Safe | Avail. viol. | Soft score before -> after | Delta | Edits | norm-L1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        mark = " **(shipped)**" if row.lambda_validity == shipped else ""
        lines.append(
            f"| **{row.lambda_validity:g}**{mark} | {_pct(row.validity)} "
            f"| {_pct(row.safe_rate)} | {_pct(row.availability_violation)} "
            f"| {row.soft_before:.3f} -> {row.soft_after:.3f} | +{row.delta:.3f} "
            f"| {row.edits:.2f} | {row.norm_l1:.3f} |"
        )

    lines += [
        "",
        "## Reading it",
        "",
        "### The dial",
        "",
        f"At {low.lambda_validity:g} the validity term is bounded by the target score (0.70) while each",
        "edit costs `lambda_sparsity` outright, so the optimiser cannot afford the edits a",
        f"meal needs: {low.edits:.2f} edits and a soft-score gain of +{low.delta:.3f}. It repairs safety,",
        "which is priced separately and far higher, and then stops. As the weight rises the",
        f"same search buys more compliance with more edits, reaching {_pct(best.validity)} validity at",
        f"{best.lambda_validity:g} for {best.edits:.2f} edits and {best.norm_l1:.3f} normalised grams moved.",
        "",
        "Neither end is obviously right. A recommendation that rebuilds the plate is",
        "correct by the guidelines and is not a counterfactual explanation; one that",
        "changes almost nothing is minimal and does not help. The shipped setting is a",
        "judgement about that trade-off, and this table is what it was judged against.",
        "",
        "### What does not move",
        "",
    ]
    if unsafe:
        lines += [
            "**A safety violation appeared in this sweep**, at "
            + ", ".join(f"`{r.lambda_validity:g}` ({_pct(1 - r.safe_rate)})" for r in unsafe)
            + ". That contradicts the claim above and needs investigating before this",
            "artefact is cited.",
            "",
        ]
    else:
        lines += [
            f"Safety holds at {_pct(min(r.safe_rate for r in rows))} across every setting, and this is not luck. The",
            "hard-safety penalty is a separate term weighted `big_penalty` (1000) against a",
            f"validity term weighted at most {high.lambda_validity:g} here, so no achievable validity gain can",
            "pay for a hazard. The sweep is the falsifiable form of that argument: if the",
            "terms were genuinely commensurable, a large enough `lambda_validity` would",
            "eventually buy a violation. None does.",
            "",
        ]
    if unavailable:
        lines += [
            "**An availability violation appeared**, at "
            + ", ".join(f"`{r.lambda_validity:g}`" for r in unavailable)
            + ", which should be impossible by construction. Investigate `build_space`.",
            "",
        ]
    else:
        lines += [
            "Availability violations stay at 0% for a stronger reason than weighting: they",
            "are not a term at all. A food the user does not have has no decision variable,",
            "so no point in the search space contains one, and no weight in",
            "`configs/pipeline.yaml` can change that. This column is here to make the",
            "distinction visible -- safety is *priced* out, availability is *designed* out.",
            "",
        ]

    lines += [
        "### Caveat on scale",
        "",
        f"This sweep runs {rows[0].n} cases per setting against the comparison table's 300, because",
        "it re-optimises every case once per setting. It is sized to show the shape of the",
        "trade-off, not to give a validity figure precise to the percentage point. The",
        "headline validity number for FoodSense is the one in",
        "[`cf_comparison.md`](cf_comparison.md), measured at the shipped setting.",
        "",
    ]
    return "\n".join(lines)


def write_figure(rows: list[SweepRow], shipped: float) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    x = [r.lambda_validity for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(x, [r.validity * 100 for r in rows], "o-", color="#2a9d8f", label="Validity %")
    ax.plot(x, [r.safe_rate * 100 for r in rows], "s--", color="#264653", label="Safe %")
    ax.set_xlabel("lambda_validity")
    ax.set_ylabel("% of cases")
    ax.set_ylim(0, 105)
    ax.axvline(shipped, color="#c1121f", lw=1, ls=":", label=f"shipped ({shipped:g})")

    twin = ax.twinx()
    twin.plot(x, [r.edits for r in rows], "^-", color="#e76f51", label="Edits per meal")
    twin.set_ylabel("edits per meal")

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, frameon=False, loc="center right")
    ax.set_title("Validity is a dial; safety is not a setting")
    ax.spines[["top"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = RESULTS_DIR / "figures" / "lambda_sweep.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=10, help="cases per age group")
    parser.add_argument("--lambdas", type=float, nargs="*", default=list(DEFAULT_LAMBDAS))
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db, engine, model = get_food_db(), RuleEngine(), get_suitability_model()
    base, de_config = ObjectiveConfig.load(), DEConfig.load()
    shipped = base.lambda_validity

    cases = list(build_cases(db, args.cases))
    print(f"{len(cases)} cases x {len(args.lambdas)} settings (shipped default {shipped:g})")

    rows: list[SweepRow] = []
    for value in sorted(args.lambdas):
        row = run_setting(value, cases, db, engine, model, base, de_config)
        rows.append(row)
        print(
            f"  lambda={value:>5g}  valid {_pct(row.validity):>4}  safe {_pct(row.safe_rate):>4}  "
            f"edits {row.edits:>4.2f}  soft {row.soft_before:.3f}->{row.soft_after:.3f}  "
            f"({row.runtime_s:.0f}s)"
        )

    (RESULTS_DIR / "lambda_sweep.md").write_text(
        render(rows, shipped, args.cases), encoding="utf-8"
    )

    import pandas as pd

    pd.DataFrame([dataclasses.asdict(r) | {"delta_soft": round(r.delta, 4)} for r in rows]).to_csv(
        RESULTS_DIR / "lambda_sweep.csv", index=False
    )

    figure = write_figure(rows, shipped)
    print(f"\nwrote {RESULTS_DIR / 'lambda_sweep.md'} and lambda_sweep.csv")
    if figure:
        print(f"wrote {figure}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
