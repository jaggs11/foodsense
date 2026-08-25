"""Repository-shape checks.

Cheap guards that catch a broken install or a missing scaffold file long before a
stage-level test would, and that keep the layout honest as the project grows.
"""

from __future__ import annotations

import importlib

import pytest

from foodsense import CONFIG_DIR, PROJECT_ROOT, SEED, __version__

EXPECTED_MODULES = [
    "foodsense.schemas",
    "foodsense.cli",
    "foodsense.pipeline",
    "foodsense.data.build_food_db",
    "foodsense.data.fdc",
    "foodsense.data.corpora",
    "foodsense.constraints.engine",
    "foodsense.constraints.age_rules",
    "foodsense.constraints.goals",
    "foodsense.stage1_prediction.features",
    "foodsense.stage1_prediction.labels",
    "foodsense.stage1_prediction.train",
    "foodsense.stage1_prediction.predict",
    "foodsense.stage2_optimizer.space",
    "foodsense.stage2_optimizer.objective",
    "foodsense.stage2_optimizer.de_optimizer",
    "foodsense.stage2_optimizer.baselines",
    "foodsense.stage3_rag.retriever",
    "foodsense.stage3_rag.providers",
    "foodsense.stage3_rag.translate",
    "foodsense.stage4_verification.verifier",
]

EXPECTED_PATHS = [
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    "make.ps1",
    "Dockerfile",
    "docker-compose.yml",
    "LICENSE",
    "README.md",
    ".env.example",
    ".github/workflows/ci.yml",
    "api/main.py",
    "data/README.md",
    "docs/architecture.md",
]


@pytest.mark.parametrize("module", EXPECTED_MODULES)
def test_every_pipeline_module_imports(module):
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize("relpath", EXPECTED_PATHS)
def test_scaffold_file_exists(relpath):
    assert (PROJECT_ROOT / relpath).exists(), f"missing scaffold file: {relpath}"


def test_seed_is_pinned():
    """Reproducibility is a stated requirement; the seed must not drift."""
    assert SEED == 42


def test_version_is_set():
    assert __version__


def test_config_dir_exists():
    assert CONFIG_DIR.is_dir()


def test_api_health_endpoint():
    from api.main import app
    from fastapi.testclient import TestClient

    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
