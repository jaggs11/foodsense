"""Tests for the schema and the USDA migration (P1.1, P1.2).

These assert invariants on the committed database, which is what CI actually loads.
The full-rebuild determinism check lives in ``test_build_determinism.py``, in its own
module because the module-scoped connection below would stop the rebuild from
replacing the file on Windows.
"""

from __future__ import annotations

import sqlite3

import pytest

from foodsense import FOOD_DB_SQLITE
from foodsense.data.schema import (
    FOODS_COLUMNS,
    MIGRATION_TIMESTAMP,
    NUTRIENT_COLUMNS,
    USDA_PROVENANCE,
    ddl_statements,
    namespaced_id,
)

EXPECTED_ROWS = 2590


@pytest.fixture(scope="module")
def conn():
    # Explicitly closed. `with sqlite3.connect(...)` commits on exit but does NOT
    # close the connection, and a leaked handle stops `make data` from replacing the
    # database on Windows -- the same trap that was live in FoodDB._read.
    c = sqlite3.connect(FOOD_DB_SQLITE)
    try:
        yield c
    finally:
        c.close()


class TestSchemaShape:
    def test_every_expected_table_exists(self, conn):
        got = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "foods",
            "food_aliases",
            "food_nutrients_extra",
            "food_toddler_meta",
            "recipe_derivations",
            "sources",
            "build_info",
        } <= got

    def test_foods_has_every_declared_column(self, conn):
        cols = {d[1] for d in conn.execute("PRAGMA table_info(foods)")}
        assert set(FOODS_COLUMNS) <= cols
        assert set(NUTRIENT_COLUMNS) <= cols

    def test_the_nutrient_columns_are_nullable(self, conn):
        """The whole of R8 rests on this."""
        notnull = {d[1] for d in conn.execute("PRAGMA table_info(foods)") if d[3]}
        assert not (set(NUTRIENT_COLUMNS) & notnull)

    def test_provenance_columns_are_not_nullable(self, conn):
        """A row without provenance is a row that cannot be cited."""
        notnull = {d[1] for d in conn.execute("PRAGMA table_info(foods)") if d[3]}
        assert {"source_type", "confidence", "verification_status"} <= notnull

    def test_the_ddl_is_valid_against_a_fresh_database(self):
        scratch = sqlite3.connect(":memory:")
        for statement in ddl_statements():
            scratch.execute(statement)
        tables = {
            r[0] for r in scratch.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "foods" in tables

    def test_name_key_is_uniquely_indexed_among_active_rows(self, conn):
        """Duplicate prevention enforced by the database, not by review."""
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_foods_name_key_active'"
        ).fetchone()
        assert sql is not None
        assert "UNIQUE" in sql[0].upper()
        assert "is_active" in sql[0]


class TestCheckConstraintsBite:
    """An out-of-vocabulary value must fail the write, not reach a user."""

    @pytest.fixture
    def fresh(self):
        c = sqlite3.connect(":memory:")
        for statement in ddl_statements():
            c.execute(statement)
        return c

    def _insert(self, c, **overrides):
        row = {
            "id": "x:1",
            "canonical_name": "X",
            "name_key": "x",
            "cuisine": "other",
            "veg_status": "unknown",
            "default_form": "whole",
            "allowed_forms": "[]",
            "texture": "unknown",
            "tags": "[]",
            "basis": "per_100g_edible_portion",
            "source_type": "usda",
            "confidence": "unknown",
            "verification_status": "pending",
            "is_active": 1,
            "created_at": MIGRATION_TIMESTAMP,
            "updated_at": MIGRATION_TIMESTAMP,
        }
        row.update(overrides)
        cols = ",".join(row)
        marks = ",".join("?" * len(row))
        c.execute(f"INSERT INTO foods ({cols}) VALUES ({marks})", list(row.values()))

    def test_a_valid_row_inserts(self, fresh):
        self._insert(fresh)

    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("cuisine", "martian"),
            ("veg_status", "sometimes"),
            ("source_type", "a_friend_told_me"),
            ("confidence", "quite_sure"),
            ("verification_status", "probably_fine"),
        ],
    )
    def test_an_out_of_vocabulary_value_is_rejected(self, fresh, field, bad):
        with pytest.raises(sqlite3.IntegrityError):
            self._insert(fresh, **{field: bad})

    def test_a_nonpositive_serving_size_is_rejected(self, fresh):
        with pytest.raises(sqlite3.IntegrityError):
            self._insert(fresh, serving_size_g=0.0)

    def test_a_null_serving_size_is_allowed(self, fresh):
        """USDA states none, and unknown must be representable."""
        self._insert(fresh, serving_size_g=None)


class TestTheMigration:
    def test_the_row_count_is_unchanged(self, conn):
        assert conn.execute("SELECT COUNT(*) FROM foods").fetchone()[0] == EXPECTED_ROWS

    def test_every_row_is_stamped_usda(self, conn):
        n = conn.execute(
            "SELECT COUNT(*) FROM foods WHERE source_type=? AND source_name=? "
            "AND confidence=? AND verification_status=? AND cuisine=?",
            (
                USDA_PROVENANCE["source_type"],
                USDA_PROVENANCE["source_name"],
                USDA_PROVENANCE["confidence"],
                USDA_PROVENANCE["verification_status"],
                USDA_PROVENANCE["cuisine"],
            ),
        ).fetchone()[0]
        assert n == EXPECTED_ROWS

    def test_timestamps_are_a_fixed_constant_not_wall_clock(self, conn):
        """Wall-clock here would make `make data` non-reproducible."""
        rows = conn.execute("SELECT DISTINCT created_at, updated_at FROM foods").fetchall()
        assert rows == [(MIGRATION_TIMESTAMP, MIGRATION_TIMESTAMP)]

    def test_usda_ids_are_preserved(self, conn):
        """Committed scenarios and golden traces pin these values."""
        n = conn.execute("SELECT COUNT(*) FROM foods WHERE id = source_id").fetchone()[0]
        assert n == EXPECTED_ROWS

    def test_the_sources_registry_names_usda(self, conn):
        rows = conn.execute("SELECT source_type, name FROM sources").fetchall()
        assert ("usda", "USDA FoodData Central") in rows

    def test_name_key_collisions_are_deactivated_not_dropped(self, conn):
        total, active = conn.execute("SELECT COUNT(*), SUM(is_active) FROM foods").fetchone()
        assert total == EXPECTED_ROWS
        assert active < total, "no collisions deactivated -- did the unique index vanish?"
        assert total - active < 50, "an implausible number of rows was deactivated"


class TestNamespacedId:
    def test_it_joins_source_and_id(self):
        assert namespaced_id("indian_dataset", "IFCT-A017") == "indian_dataset:IFCT-A017"

    def test_it_is_deterministic(self):
        """A UUID here would break `make data` reproducibility."""
        assert namespaced_id("government", "X") == namespaced_id("government", "X")

    @pytest.mark.parametrize(("st", "sid"), [("", "x"), ("indian_dataset", ""), ("bad:type", "x")])
    def test_malformed_input_is_rejected(self, st, sid):
        with pytest.raises(ValueError):
            namespaced_id(st, sid)
