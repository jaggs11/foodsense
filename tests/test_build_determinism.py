"""`make data` must be reproducible (P1.2).

Its own module on purpose. A module-scoped fixture holding an open connection to
``food_db.sqlite`` anywhere in the same file stops the rebuild from replacing it on
Windows, so this file deliberately opens nothing until the builds are done.

Marked slow and excluded from CI, which has neither the raw archives nor the time.
Skipped -- never silently passed -- when the archives are absent, so a green run
cannot be mistaken for evidence that determinism was checked.
"""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys

import pytest

from foodsense import FOOD_DB_SQLITE, RAW_DIR

CONTENT_TABLES = (
    "foods",
    "food_aliases",
    "food_nutrients_extra",
    "food_toddler_meta",
    "recipe_derivations",
    "sources",
)


def _content_hash() -> str:
    """Hash of every content table. Excludes ``build_info``, which records when the
    build ran rather than what it produced -- ``built_at`` is wall-clock by design."""
    h = hashlib.sha256()
    conn = sqlite3.connect(FOOD_DB_SQLITE)
    try:
        for table in CONTENT_TABLES:
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1"):
                h.update(repr(row).encode())
    finally:
        conn.close()
    return h.hexdigest()


@pytest.mark.slow
def test_two_rebuilds_produce_identical_content():
    if not (RAW_DIR / "fdc" / "sr_legacy.zip").exists():
        pytest.skip("raw USDA archives absent; cannot rebuild (determinism unchecked)")

    build = [sys.executable, "-m", "foodsense.data.build_food_db"]

    first_run = subprocess.run(build, capture_output=True, text=True)
    assert first_run.returncode == 0, "first build failed: " + first_run.stderr[-2000:]
    first = _content_hash()

    second_run = subprocess.run(build, capture_output=True, text=True)
    assert second_run.returncode == 0, "second build failed: " + second_run.stderr[-2000:]

    assert _content_hash() == first, "make data is not reproducible"
