"""Train the Stage-1 meal-suitability surrogate.

    python -m foodsense.stage1_prediction.train           # LightGBM + XGBoost
    python -m foodsense.stage1_prediction.train --quick   # small run, for tests

Trains on meals derived from Food.com crossed with sampled profiles and synthetic
perturbations, and evaluates on two held-out sets:

* **Food.com held-out** -- meals from the same corpus, split by source meal so no
  perturbed variant of a training meal appears in the test set.
* **Nutrition5k** -- a different corpus entirely, built by different people from
  cafeteria plates rather than home recipes. The gap between the two is the
  honest measure of whether the surrogate learned the guidelines or the corpus.

Writes ``models/stage1_lightgbm.txt``, ``models/stage1_xgboost.json``,
``models/stage1_features.json`` and ``models/stage1_metrics.json``. Every number
reported in ``results/`` traces back to that metrics file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from foodsense import CONFIG_DIR, MODELS_DIR, SEED
from foodsense.constraints.engine import RuleEngine
from foodsense.data.corpora import load_meals
from foodsense.schemas import Meal
from foodsense.stage1_prediction.features import feature_names
from foodsense.stage1_prediction.labels import Dataset, build_dataset

__all__ = ["TrainConfig", "evaluate_predictions", "main", "train"]

PIPELINE_CONFIG = CONFIG_DIR / "pipeline.yaml"

LIGHTGBM_PATH = MODELS_DIR / "stage1_lightgbm.txt"
XGBOOST_PATH = MODELS_DIR / "stage1_xgboost.json"
FEATURES_PATH = MODELS_DIR / "stage1_features.json"
METRICS_PATH = MODELS_DIR / "stage1_metrics.json"


@dataclass(slots=True)
class TrainConfig:
    """Stage-1 settings from ``configs/pipeline.yaml``."""

    label_noise_sigma: float = 0.05
    target_score: float = 0.70
    auc_thresholds: tuple[float, ...] = (0.45, 0.55, 0.70)
    n_train_meals: int = 8000
    n_profiles_per_meal: int = 3
    perturbation_rate: float = 0.5
    test_size: float = 0.2
    seed: int = SEED
    lightgbm: dict[str, Any] = None  # type: ignore[assignment]
    xgboost: dict[str, Any] = None  # type: ignore[assignment]

    @classmethod
    def load(cls) -> TrainConfig:
        raw = yaml.safe_load(PIPELINE_CONFIG.read_text(encoding="utf-8")) or {}
        section = raw.get("stage1") or {}
        return cls(
            label_noise_sigma=float(section.get("label_noise_sigma", 0.05)),
            target_score=float(section.get("target_score", 0.70)),
            auc_thresholds=tuple(section.get("auc_thresholds") or (0.45, 0.55, 0.70)),
            n_train_meals=int(section.get("n_train_meals", 8000)),
            n_profiles_per_meal=int(section.get("n_profiles_per_meal", 3)),
            perturbation_rate=float(section.get("perturbation_rate", 0.5)),
            test_size=float(section.get("test_size", 0.2)),
            seed=int(raw.get("seed", SEED)),
            lightgbm=dict(section.get("lightgbm") or {}),
            xgboost=dict(section.get("xgboost") or {}),
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_score: float,
    thresholds: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """RMSE, MAE, R^2, and AUC for the thresholded "meets target" decision.

    AUC is reported because the number that matters downstream is not the exact
    score but whether the optimiser can tell an acceptable meal from an
    unacceptable one. It is ``None`` when the held-out set happens to be all one
    class, rather than a fabricated 0.5.
    """
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": len(y_true),
    }
    labels = (y_true >= target_score).astype(int)
    metrics["positive_rate"] = float(labels.mean())
    metrics["auc"] = _auc(labels, y_pred)

    # The positive class at the target is deliberately small -- most unedited
    # meals are not suitable, which is the premise of the whole system. Reporting
    # AUC at lower cut-offs as well shows whether the model ranks meals correctly
    # across the range, rather than only near the boundary.
    metrics["auc_at"] = {
        f"{t:.2f}": _auc((y_true >= t).astype(int), y_pred) for t in thresholds or ()
    }
    return metrics


def _auc(labels: np.ndarray, y_pred: np.ndarray) -> float | None:
    """ROC AUC, or ``None`` when the split is single-class -- never a faked 0.5."""
    return float(roc_auc_score(labels, y_pred)) if 0 < labels.sum() < len(labels) else None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _fit_lightgbm(X: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(random_state=seed, verbose=-1, **params)
    model.fit(X, y, feature_name=feature_names())
    return model


def _fit_xgboost(X: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int):
    from xgboost import XGBRegressor

    model = XGBRegressor(random_state=seed, verbosity=0, **params)
    model.fit(X, y)
    return model


def _cross_validate(
    dataset: Dataset, params: dict[str, Any], seed: int, target_score: float, n_splits: int = 5
) -> dict[str, float]:
    """Grouped K-fold CV, so a meal's variants never straddle a fold boundary."""
    n_splits = min(n_splits, dataset.n_groups)
    if n_splits < 2:
        return {"rmse_mean": float("nan"), "rmse_std": float("nan"), "n_splits": 0}
    scores = []
    for train_index, test_index in GroupKFold(n_splits=n_splits).split(
        dataset.X, dataset.y, dataset.groups
    ):
        model = _fit_lightgbm(dataset.X[train_index], dataset.y[train_index], params, seed)
        prediction = np.clip(model.predict(dataset.X[test_index]), 0.0, 1.0)
        scores.append(evaluate_predictions(dataset.y[test_index], prediction, target_score)["rmse"])
    return {
        "rmse_mean": float(np.mean(scores)),
        "rmse_std": float(np.std(scores)),
        "n_splits": n_splits,
        "fold_rmse": [round(s, 5) for s in scores],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _load_corpus_meals(source: str, limit: int, rows: int) -> list[Meal]:
    return [m.meal for m in load_meals(source, limit=limit, rows=rows)]


def train(config: TrainConfig | None = None, quick: bool = False) -> dict[str, Any]:
    """Build the dataset, fit both models, evaluate, and write everything to ``models/``."""
    config = config or TrainConfig.load()
    if quick:
        config.n_train_meals = 250
        config.n_profiles_per_meal = 2

    started = time.perf_counter()
    engine = RuleEngine()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Food.com meals (target {config.n_train_meals})...")
    train_meals = _load_corpus_meals(
        "foodcom", limit=config.n_train_meals, rows=config.n_train_meals * 4
    )
    print(f"  {len(train_meals)} meals")

    print("Loading Nutrition5k meals (out-of-corpus evaluation)...")
    try:
        n5k_meals = _load_corpus_meals(
            "nutrition5k", limit=max(config.n_train_meals // 4, 200), rows=20_000
        )
        print(f"  {len(n5k_meals)} meals")
    except FileNotFoundError as exc:
        print(f"  unavailable: {exc}")
        n5k_meals = []

    print("Building the training set...")
    dataset = build_dataset(
        train_meals,
        engine,
        n_profiles_per_meal=config.n_profiles_per_meal,
        perturbation_rate=config.perturbation_rate,
        noise_sigma=config.label_noise_sigma,
        seed=config.seed,
    )
    print(
        f"  {len(dataset)} instances from {dataset.n_groups} meals, {dataset.X.shape[1]} features"
    )

    # Split by source meal, not by row: perturbed variants of one meal must not
    # appear on both sides or the held-out metrics are meaningless.
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    train_index, test_index = next(splitter.split(dataset.X, dataset.y, dataset.groups))
    X_train, y_train = dataset.X[train_index], dataset.y[train_index]
    X_test, y_test = dataset.X[test_index], dataset.y[test_index]
    print(f"  train {len(y_train)} / held-out {len(y_test)}")

    results: dict[str, Any] = {
        "config": asdict(config),
        "n_features": dataset.X.shape[1],
        "n_instances": len(dataset),
        "n_source_meals": dataset.n_groups,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "models": {},
    }

    print("\nTraining LightGBM...")
    t0 = time.perf_counter()
    lightgbm_model = _fit_lightgbm(X_train, y_train, config.lightgbm, config.seed)
    lightgbm_seconds = time.perf_counter() - t0
    lightgbm_model.booster_.save_model(str(LIGHTGBM_PATH))

    print("Training XGBoost...")
    t0 = time.perf_counter()
    xgboost_model = _fit_xgboost(X_train, y_train, config.xgboost, config.seed)
    xgboost_seconds = time.perf_counter() - t0
    xgboost_model.save_model(str(XGBOOST_PATH))

    for name, model, seconds in (
        ("lightgbm", lightgbm_model, lightgbm_seconds),
        ("xgboost", xgboost_model, xgboost_seconds),
    ):
        entry: dict[str, Any] = {"train_seconds": round(seconds, 2)}
        entry["foodcom_holdout"] = evaluate_predictions(
            y_test,
            np.clip(model.predict(X_test), 0.0, 1.0),
            config.target_score,
            config.auc_thresholds,
        )
        entry["foodcom_train"] = evaluate_predictions(
            y_train,
            np.clip(model.predict(X_train), 0.0, 1.0),
            config.target_score,
            config.auc_thresholds,
        )
        results["models"][name] = entry

    print("\nCross-validating LightGBM (grouped 5-fold)...")
    results["models"]["lightgbm"]["cv"] = _cross_validate(
        Dataset(X_train, y_train, y_train, dataset.groups[train_index]),
        config.lightgbm,
        config.seed,
        config.target_score,
    )

    if n5k_meals:
        print("Evaluating on Nutrition5k (different corpus)...")
        n5k = build_dataset(
            n5k_meals,
            engine,
            n_profiles_per_meal=config.n_profiles_per_meal,
            perturbation_rate=config.perturbation_rate,
            noise_sigma=config.label_noise_sigma,
            seed=config.seed + 1,
        )
        for name, model in (("lightgbm", lightgbm_model), ("xgboost", xgboost_model)):
            results["models"][name]["nutrition5k"] = evaluate_predictions(
                n5k.y,
                np.clip(model.predict(n5k.X), 0.0, 1.0),
                config.target_score,
                config.auc_thresholds,
            )
        results["n_nutrition5k_instances"] = len(n5k)

    importance = sorted(
        zip(feature_names(), lightgbm_model.feature_importances_, strict=True),
        key=lambda kv: -kv[1],
    )
    results["lightgbm_feature_importance"] = [
        {"feature": f, "gain": int(g)} for f, g in importance[:20]
    ]
    results["total_seconds"] = round(time.perf_counter() - started, 1)

    FEATURES_PATH.write_text(
        json.dumps(
            {"feature_names": feature_names(), "target_score": config.target_score}, indent=2
        ),
        encoding="utf-8",
    )
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    _print_summary(results)
    return results


def _print_summary(results: dict[str, Any]) -> None:
    print("\n" + "=" * 74)
    print("STAGE-1 SUITABILITY SURROGATE")
    print("=" * 74)
    print(
        f"instances {results['n_instances']} from {results['n_source_meals']} meals "
        f"| {results['n_features']} features | {results['total_seconds']}s"
    )

    header = (
        f"{'model':<10} {'split':<20} {'RMSE':>8} {'R2':>8} "
        f"{'AUC@.45':>8} {'AUC@.55':>8} {'AUC@tgt':>8} {'n':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, entry in results["models"].items():
        for split in ("foodcom_train", "foodcom_holdout", "nutrition5k"):
            metrics = entry.get(split)
            if not metrics:
                continue

            def fmt(value):
                return f"{value:.4f}" if value is not None else "n/a"

            at = metrics.get("auc_at", {})
            print(
                f"{name:<10} {split:<20} {metrics['rmse']:>8.4f} {metrics['r2']:>8.4f} "
                f"{fmt(at.get('0.45')):>8} {fmt(at.get('0.55')):>8} "
                f"{fmt(metrics.get('auc')):>8} {metrics['n']:>7}"
            )

    cv = results["models"]["lightgbm"].get("cv", {})
    if cv.get("n_splits"):
        print(
            f"\nlightgbm grouped {cv['n_splits']}-fold CV RMSE: "
            f"{cv['rmse_mean']:.4f} +/- {cv['rmse_std']:.4f}"
        )

    print("\ntop features by gain:")
    for row in results["lightgbm_feature_importance"][:10]:
        print(f"  {row['feature']:<26} {row['gain']}")
    print(f"\nwrote {LIGHTGBM_PATH.name}, {XGBOOST_PATH.name}, {METRICS_PATH.name}")
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small run for smoke tests")
    args = parser.parse_args(argv)
    train(quick=args.quick)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
