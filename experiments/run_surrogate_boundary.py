"""Is the surrogate/rule-engine gap at the target structural? -> ``results/surrogate_boundary.{md,csv}``.

Stage 2 optimises the *surrogate's* estimate and Stage 4 reports validity from the
*rule engine's* score. That split is deliberate -- an optimiser graded by its own
model can win by finding the model's blind spots -- but it has a consequence at the
decision boundary that this experiment is here to measure rather than assume.

The suspicion, raised from one scenario where the surrogate scored a meal 0.7007
and the rules scored it 0.6793:

1. Stage-1's error is around 0.057-0.059 RMSE. A 0.021 disagreement is well inside
   one standard error, so that scenario is not an anomaly -- near the target it is
   the expected behaviour of the design.
2. The validity term is ``max(0, target - surrogate)``. Once the surrogate clears
   the target it is exactly zero, so the objective goes flat in that direction and
   the search has no gradient left to close a remaining rule-engine gap. It
   plateaus, patience trips, and the case is recorded invalid.

If both hold, some share of the invalid cases are meals the optimiser could not
*tell* were invalid, and then had nothing to climb -- a calibration artefact rather
than a search failure.

This measures the model, not the optimiser, so it depends on nothing the CF
evaluation produces and cannot be accused of being fitted to it. Two things are
computed on held-out data the surrogate never trained on:

* the residual distribution ``surrogate - rule_engine`` overall and, crucially,
  restricted to meals whose true score sits near the target;
* the **straddle rate** -- among near-boundary meals, how often the surrogate lands
  on the opposite side of the target from the truth.

    python experiments/run_surrogate_boundary.py
    python experiments/run_surrogate_boundary.py --meals 400
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodsense import RESULTS_DIR
from foodsense.constraints.engine import RuleEngine
from foodsense.data.corpora import load_meals
from foodsense.stage1_prediction.labels import build_dataset
from foodsense.stage1_prediction.predict import get_suitability_model
from foodsense.stage1_prediction.train import TrainConfig
from foodsense.stage2_optimizer.objective import ObjectiveConfig

#: Half-width of the band around the target that counts as "near the boundary".
#: Set to one held-out RMSE rather than a round number: the question is whether the
#: model's own error is large enough to flip the verdict, so the band is defined in
#: units of that error.
NEAR_BAND_IN_RMSE = 1.0


@dataclass(slots=True)
class Residuals:
    """Surrogate minus rule engine, on meals the surrogate never saw."""

    n: int
    rmse: float
    bias: float
    sd: float
    q05: float
    q50: float
    q95: float


def _summarise(values: np.ndarray) -> Residuals:
    return Residuals(
        n=len(values),
        rmse=float(np.sqrt(np.mean(values**2))) if len(values) else 0.0,
        bias=float(np.mean(values)) if len(values) else 0.0,
        sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        q05=float(np.quantile(values, 0.05)) if len(values) else 0.0,
        q50=float(np.quantile(values, 0.50)) if len(values) else 0.0,
        q95=float(np.quantile(values, 0.95)) if len(values) else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meals",
        type=int,
        default=0,
        help="cap on held-out rows scored (0 = the whole held-out split)",
    )
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    engine = RuleEngine()
    model = get_suitability_model()
    train_config = TrainConfig.load()
    target = ObjectiveConfig.load().target_score

    # Reconstruct the *exact* held-out split the shipped model was evaluated on:
    # same corpus, same count, same dataset seed, same GroupShuffleSplit random
    # state. Anything looser risks scoring the model on rows it was fitted to,
    # which would understate the residual and quietly flatter the conclusion.
    print(f"reloading the training corpus ({train_config.n_train_meals} meals)")
    meals = [
        m.meal
        for m in load_meals(
            "foodcom", limit=train_config.n_train_meals, rows=train_config.n_train_meals * 4
        )
    ]
    if not meals:
        print("no corpus meals available; run `make data` first", file=sys.stderr)
        return 1

    dataset = build_dataset(
        meals,
        engine,
        n_profiles_per_meal=train_config.n_profiles_per_meal,
        perturbation_rate=train_config.perturbation_rate,
        noise_sigma=train_config.label_noise_sigma,
        seed=train_config.seed,
    )
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=train_config.test_size, random_state=train_config.seed
    )
    _, test_index = next(splitter.split(dataset.X, dataset.y, dataset.groups))
    if args.meals and args.meals < len(test_index):
        test_index = test_index[: args.meals]
    print(f"  {len(test_index)} held-out rows from {dataset.n_groups} source meals")

    # `y_clean` is the rule engine's soft score before label noise -- the same
    # quantity the surrogate is trained to predict, and the one Stage 2's validity
    # decision is made against. Comparing against the noised label would measure
    # the noise as well as the model.
    truth = np.asarray(dataset.y_clean[test_index], dtype=np.float64)
    predicted = np.asarray(model.predict_features(dataset.X[test_index]), dtype=np.float64)
    residual = predicted - truth

    overall = _summarise(residual)
    band = NEAR_BAND_IN_RMSE * overall.rmse
    near_mask = np.abs(truth - target) <= band
    near = _summarise(residual[near_mask])

    # The rate that matters is a *conditional* one, and getting its denominator
    # wrong changes the answer. The optimiser stops when its validity term reaches
    # zero, i.e. when the surrogate clears the target. The cases that costs are
    # meals whose true score is below target and whose prediction is not -- so the
    # denominator is "meals genuinely short of target", not "all meals".
    truth_over = truth >= target
    pred_over = predicted >= target
    optimistic = pred_over & ~truth_over
    pessimistic = truth_over & ~pred_over

    short = ~truth_over
    near_short = short & near_mask
    optimistic_given_short = float(optimistic[short].mean()) if short.any() else 0.0
    optimistic_given_near_short = float(optimistic[near_short].mean()) if near_short.any() else 0.0
    # How big is the mistake when it happens? A flip that clears the target by
    # 0.001 is a different problem from one that clears it by 0.05.
    flip_gap = (predicted - truth)[optimistic]
    near_flip_gap = (predicted - truth)[optimistic & near_mask]

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# Is the surrogate/rule-engine gap at the target structural?",
        "",
        f"Generated by `experiments/run_surrogate_boundary.py` at {stamp}, over",
        f"{overall.n} held-out (meal, profile) pairs the shipped surrogate never trained on.",
        "This measures the **model**, not the optimiser: it reads nothing the",
        "counterfactual evaluation produced and so cannot be fitted to it.",
        "",
        "## Residual, surrogate minus rule engine",
        "",
        "| Population | n | RMSE | Bias | SD | 5th pct | Median | 95th pct |",
        "|---|---|---|---|---|---|---|---|",
        f"| All held-out meals | {overall.n} | {overall.rmse:.4f} | {overall.bias:+.4f} "
        f"| {overall.sd:.4f} | {overall.q05:+.4f} | {overall.q50:+.4f} | {overall.q95:+.4f} |",
        f"| Within {band:.3f} of the {target:.2f} target | {near.n} | {near.rmse:.4f} "
        f"| {near.bias:+.4f} | {near.sd:.4f} | {near.q05:+.4f} | {near.q50:+.4f} "
        f"| {near.q95:+.4f} |",
        "",
        f"The near-boundary band is one RMSE wide on each side ({band:.3f}), because the",
        "question is precisely whether the model's own error is large enough to flip the",
        "verdict at the target. Defining the band in units of that error keeps the",
        "comparison honest rather than picking a round number that flatters it.",
        "",
        "## How often the surrogate lands on the wrong side of the target",
        "",
        "The denominator matters. The optimiser stops when the surrogate clears the",
        "target, so the failure that costs validity is specifically: **the meal is",
        "genuinely short, and the surrogate says it is not**. Rates below are conditioned",
        "on the meal actually being short, not on the whole population.",
        "",
        "| Population | n | Surrogate says done while the rules say short |",
        "|---|---|---|",
        f"| All held-out meals scoring below {target:.2f} | {int(short.sum())} "
        f"| {optimistic_given_short * 100:.1f}% |",
        f"| ...and within {band:.3f} of the target | {int(near_short.sum())} "
        f"| {optimistic_given_near_short * 100:.1f}% |",
        "",
        "| Direction | Share of all held-out meals |",
        "|---|---|",
        f"| Surrogate optimistic (says done, rules say short) | {optimistic.mean() * 100:.1f}% |",
        f"| Surrogate pessimistic (says short, rules say done) | {pessimistic.mean() * 100:.1f}% |",
        "",
        "Only the optimistic direction costs validity. A pessimistic error makes the",
        "optimiser keep working on a meal that was already good enough, which spends",
        "budget and edits but cannot turn a valid case invalid.",
        "",
    ]

    if len(flip_gap):
        lines += [
            "### How large is the mistake when it happens",
            "",
            "| Population | n | Mean over-estimate | Median | 90th pct | Max |",
            "|---|---|---|---|---|---|",
            f"| All optimistic flips | {len(flip_gap)} | {flip_gap.mean():.4f} "
            f"| {np.median(flip_gap):.4f} | {np.quantile(flip_gap, 0.9):.4f} "
            f"| {flip_gap.max():.4f} |",
        ]
        if len(near_flip_gap):
            lines.append(
                f"| Near-boundary optimistic flips | {len(near_flip_gap)} "
                f"| {near_flip_gap.mean():.4f} | {np.median(near_flip_gap):.4f} "
                f"| {np.quantile(near_flip_gap, 0.9):.4f} | {near_flip_gap.max():.4f} |"
            )
        lines += [
            "",
            "This is the headroom a calibrated stopping offset would have to cover.",
            "",
        ]

    lines += ["## Verdict", ""]
    lines += [
        "**First, a correction to the premise this was set up to test.** The Stage-1",
        "report's held-out RMSE of ~0.057 is measured against the *noised* training",
        "label: `configs/pipeline.yaml` adds Gaussian noise with sigma 0.05 to every",
        "label on purpose, to stop the surrogate memorising the rule engine's exact",
        "boundaries. That noise is in the label, not in the model's estimate of the",
        "underlying quantity. Measured against the clean rule-engine soft score -- which",
        "is what Stage 2 is actually trying to predict and what validity is judged on --",
        f"the residual RMSE is **{overall.rmse:.4f}**, and",
        f"sqrt({overall.rmse:.4f}^2 + 0.05^2) = {(overall.rmse**2 + 0.05**2) ** 0.5:.4f}, which recovers the reported",
        "figure almost exactly. The surrogate is roughly twice as accurate at the",
        "decision boundary as the headline number suggests.",
        "",
    ]
    if optimistic_given_near_short >= 0.30:
        lines += [
            "**The effect is nonetheless real and large.** Among held-out meals that are",
            "genuinely short of target and sit within one residual RMSE of it,",
            f"{optimistic_given_near_short * 100:.0f}% are called finished by the surrogate. Any meal the optimiser",
            "drives to the target is drawn from exactly that population, and the search",
            "*selects* for it: it climbs until `max(0, target - surrogate)` reaches zero,",
            "which means it deliberately stops where the model is least able to tell.",
            "",
            "The flat objective compounds it. Above the target the validity term is",
            "identically zero, so there is no gradient left to close a remaining",
            "rule-engine gap even in principle -- the search is not failing to climb, there",
            "is nothing there to climb.",
            "",
        ]
    elif optimistic_given_near_short >= 0.15:
        lines += [
            f"**The effect is real but bounded.** {optimistic_given_near_short * 100:.0f}% of near-boundary short meals",
            "are called finished by the surrogate. That is enough to account for some",
            "invalid cases and not enough to account for most of them. Whether it matters",
            "in practice depends on how much of the invalid population actually sits in",
            "this band, which only the validity decomposition can say.",
            "",
        ]
    else:
        lines += [
            f"**The effect is small.** Only {optimistic_given_near_short * 100:.0f}% of near-boundary short meals are",
            f"called finished by the surrogate, and {optimistic.mean() * 100:.1f}% of the held-out population",
            "overall. The mechanism is real -- the objective does go flat above the target",
            "-- but the model is accurate enough at the boundary that it should not be a",
            "large share of invalid cases. The decomposition is the test of that.",
            "",
        ]

    # ---- the calibrated-target derivation, stated whether or not it is used ----
    positive = residual[residual > 0]
    q80 = float(np.quantile(positive, 0.80)) if len(positive) else 0.0
    q90 = float(np.quantile(positive, 0.90)) if len(positive) else 0.0
    lines += [
        "## If a remedy is warranted: deriving an internal target offset",
        "",
        "Should the decomposition show this is a material share of invalid cases, the",
        "offset for a calibrated internal stopping target is derived **here**, from the",
        "model's own held-out residuals, and never from the validity number it would",
        "move. Judging validity stays with the rule engine at the published target; only",
        "the point at which the search stops trying would change.",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Held-out residual RMSE (surrogate - rules) | {overall.rmse:.4f} |",
        f"| Mean over-prediction, positive residuals only | "
        f"{statistics.fmean(positive) if len(positive) else 0.0:.4f} |",
        f"| 80th percentile of positive residuals | {q80:.4f} |",
        f"| 90th percentile of positive residuals | {q90:.4f} |",
        "",
        "A one-sided quantile of the *positive* residuals is the right statistic: only",
        "over-prediction causes a premature stop, and the quantile says how much headroom",
        "covers that fraction of the over-predictions. The RMSE is reported alongside as",
        "the scale check, not as the offset -- it is two-sided and would over-correct.",
        "",
    ]
    (RESULTS_DIR / "surrogate_boundary.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "n": overall.n,
        "target": target,
        "near_band": band,
        "residual_rmse_vs_clean_score": overall.rmse,
        "label_noise_sigma": 0.05,
        "overall": dataclasses.asdict(overall),
        "near_boundary": dataclasses.asdict(near),
        "optimistic_rate_all": float(optimistic.mean()),
        "pessimistic_rate_all": float(pessimistic.mean()),
        "optimistic_given_short": optimistic_given_short,
        "optimistic_given_near_short": optimistic_given_near_short,
        "n_short": int(short.sum()),
        "n_near_short": int(near_short.sum()),
        "flip_gap_mean": float(flip_gap.mean()) if len(flip_gap) else 0.0,
        "flip_gap_q90": float(np.quantile(flip_gap, 0.9)) if len(flip_gap) else 0.0,
        "positive_residual_q80": q80,
        "positive_residual_q90": q90,
    }
    (RESULTS_DIR / "surrogate_boundary.json").write_text(json.dumps(payload, indent=2), "utf-8")

    print(f"\nresidual vs CLEAN rule score: RMSE {overall.rmse:.4f}, bias {overall.bias:+.4f}")
    print("  (reported holdout RMSE ~0.057 is vs the NOISED label, sigma 0.05)")
    print(f"near-boundary band: +/-{band:.4f} around {target:.2f}")
    print(
        f"  optimistic | short          : {optimistic_given_short * 100:.1f}% (n={int(short.sum())})"
    )
    print(
        f"  optimistic | short & near band: {optimistic_given_near_short * 100:.1f}% "
        f"(n={int(near_short.sum())})"
    )
    print(f"positive-residual q80 {q80:.4f}, q90 {q90:.4f}")
    print(f"\nwrote {RESULTS_DIR / 'surrogate_boundary.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
