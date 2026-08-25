"""Load a trained surrogate and score meals.

    model = SuitabilityModel.load()
    model.predict(meal, profile)          # -> float in [0, 1]

This is the function the Stage-2 optimiser climbs, so it is called thousands of
times per recommendation. Both the model and the food database are cached
process-wide, and :meth:`SuitabilityModel.predict_many` exists so differential
evolution can score a whole population in one batched call rather than a Python
loop over single rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from foodsense import MODELS_DIR
from foodsense.data.fdc import FoodDB, get_food_db
from foodsense.schemas import Meal, MealItem, UserProfile
from foodsense.stage1_prediction.features import feature_names, meal_features

__all__ = ["ModelMissingError", "SuitabilityModel", "get_suitability_model"]

LIGHTGBM_PATH = MODELS_DIR / "stage1_lightgbm.txt"
XGBOOST_PATH = MODELS_DIR / "stage1_xgboost.json"
FEATURES_PATH = MODELS_DIR / "stage1_features.json"


class ModelMissingError(RuntimeError):
    """Raised when the Stage-1 model has not been trained yet."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Stage-1 model not found at {path}.\n"
            f"Train it with:  python -m foodsense.stage1_prediction.train\n"
            f"(or `make train` / `./make.ps1 train`)."
        )


class _Predictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class SuitabilityModel:
    """A trained surrogate plus the feature contract it was fitted against."""

    booster: Any
    backend: str
    columns: list[str]
    target_score: float
    db: FoodDB

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, backend: str = "lightgbm", db: FoodDB | None = None) -> SuitabilityModel:
        db = db or get_food_db()
        columns, target_score = cls._load_contract()

        if backend == "lightgbm":
            if not LIGHTGBM_PATH.exists():
                raise ModelMissingError(LIGHTGBM_PATH)
            import lightgbm

            booster = lightgbm.Booster(model_file=str(LIGHTGBM_PATH))
        elif backend == "xgboost":
            if not XGBOOST_PATH.exists():
                raise ModelMissingError(XGBOOST_PATH)
            from xgboost import XGBRegressor

            booster = XGBRegressor()
            booster.load_model(str(XGBOOST_PATH))
        else:
            raise ValueError(f"Unknown backend {backend!r}; expected 'lightgbm' or 'xgboost'")

        return cls(
            booster=booster,
            backend=backend,
            columns=columns,
            target_score=target_score,
            db=db,
        )

    @staticmethod
    def _load_contract() -> tuple[list[str], float]:
        """Feature order and target threshold recorded at training time.

        Read from disk rather than recomputed so that a model trained against an
        older feature set fails loudly instead of being fed mismatched columns.
        """
        if FEATURES_PATH.exists():
            raw = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
            return list(raw["feature_names"]), float(raw.get("target_score", 0.75))
        return feature_names(), 0.75

    # -- prediction ---------------------------------------------------------

    def _raw(self, X: np.ndarray) -> np.ndarray:
        prediction = self.booster.predict(X)
        return np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)

    def predict(self, meal: Meal | list[MealItem], profile: UserProfile) -> float:
        """Predicted guideline suitability of one meal for one profile, in [0, 1]."""
        row = meal_features(meal, profile, self.db).reshape(1, -1)
        return float(self._raw(row)[0])

    def predict_many(
        self, meals: list[Meal] | list[list[MealItem]], profile: UserProfile
    ) -> np.ndarray:
        """Score a whole population of candidate meals against one profile."""
        if not meals:
            return np.zeros(0, dtype=np.float64)
        X = np.vstack([meal_features(m, profile, self.db) for m in meals])
        return self._raw(X)

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        """Score pre-built feature rows, for callers that cache their own features."""
        return self._raw(np.atleast_2d(X))

    def meets_target(self, meal: Meal | list[MealItem], profile: UserProfile) -> bool:
        """Whether the surrogate thinks this meal clears the target.

        A *prediction*, not a verdict. Stage 2 decides validity with the rule
        engine so the optimiser cannot win by exploiting its own model.
        """
        return self.predict(meal, profile) >= self.target_score


@lru_cache(maxsize=2)
def get_suitability_model(backend: str = "lightgbm") -> SuitabilityModel:
    """Process-wide cached model, sharing the cached food database."""
    return SuitabilityModel.load(backend)
