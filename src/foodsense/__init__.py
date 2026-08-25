"""FoodSense -- availability-aware, verification-guided counterfactual food recommendation.

Package-level constants only; importing this module must stay cheap so that the CLI
starts fast and the test suite does not pay for LightGBM at collection time.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

#: Global random seed. Every stochastic component (label noise, DE, sampling,
#: train/test splits) seeds from this so runs are reproducible.
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FOOD_DB_SQLITE = PROCESSED_DIR / "food_db.sqlite"
FOOD_DB_PARQUET = PROCESSED_DIR / "food_db.parquet"

__all__ = [
    "CONFIG_DIR",
    "DATA_DIR",
    "FOOD_DB_PARQUET",
    "FOOD_DB_SQLITE",
    "MODELS_DIR",
    "PROCESSED_DIR",
    "PROJECT_ROOT",
    "RAW_DIR",
    "RESULTS_DIR",
    "SAMPLES_DIR",
    "SEED",
    "__version__",
]
