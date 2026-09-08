"""Tests for nutrient coverage and the zero-fill bridge (rulings R8, R13).

The distinction these protect is one bit wide and load-bearing: a stored 0.0 that
means "this food contains none of it" versus one that means "nobody measured it".
The database now stores NULL for the second, and the pipeline keeps reading 0.0
until P3 -- but only through one named function, and only with the mask alongside.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from foodsense import FOOD_DB_SQLITE
from foodsense.data.coverage import (
    REQUIRED_FOR_CONDITION,
    NutrientCoverage,
    is_missing,
    unreported_among,
    zero_filled_vector,
)
from foodsense.data.fdc import get_food_db
from foodsense.data.schema import NUTRIENT_COLUMNS


@pytest.fixture(scope="module")
def db():
    return get_food_db()


class TestIsMissing:
    @pytest.mark.parametrize("value", [None, float("nan"), pd.NA])
    def test_the_three_shapes_of_null_all_count_as_missing(self, value):
        """sqlite gives None, pandas gives NA, parquet round-trips give nan."""
        assert is_missing(value) is True

    @pytest.mark.parametrize("value", [0.0, 0, 1.5, -0.0, 100.0, True, False])
    def test_a_real_number_is_never_missing(self, value):
        assert is_missing(value) is False

    def test_zero_is_the_case_that_matters(self):
        """If this ever returns True the whole distinction collapses."""
        assert is_missing(0.0) is False


class TestZeroFilledVector:
    def test_an_unreported_nutrient_becomes_zero(self):
        v = zero_filled_vector({"iron_mg": None, "energy_kcal": 120.0}, {"energy_kcal"})
        assert v.iron_mg == 0.0

    def test_a_reported_value_is_preserved(self):
        v = zero_filled_vector({"energy_kcal": 120.0}, {"energy_kcal"})
        assert v.energy_kcal == 120.0

    def test_a_reported_zero_is_preserved_as_zero(self):
        """Indistinguishable from the fill in the vector -- which is exactly why
        the mask has to travel with it."""
        v = zero_filled_vector({"fiber_g": 0.0}, {"fiber_g"})
        assert v.fiber_g == 0.0

    def test_every_nutrient_key_is_present_in_the_result(self):
        """Stage 1 needs a dense vector; a missing key would be a KeyError."""
        v = zero_filled_vector({}, set())
        for key in NUTRIENT_COLUMNS:
            assert getattr(v, key) == 0.0

    def test_a_value_present_but_not_in_the_mask_is_still_filled(self):
        """The mask is authoritative, not the presence of a number."""
        v = zero_filled_vector({"iron_mg": 5.0}, set())
        assert v.iron_mg == 0.0


class TestNutrientCoverage:
    def test_reports_distinguishes_reported_from_unreported(self):
        c = NutrientCoverage("x", {"energy_kcal", "fiber_g"})
        assert c.reports("energy_kcal")
        assert not c.reports("iron_mg")

    def test_missing_lists_only_what_is_absent(self):
        c = NutrientCoverage("x", {"energy_kcal"})
        assert c.missing(["energy_kcal", "iron_mg", "calcium_mg"]) == {"iron_mg", "calcium_mg"}

    def test_covers_is_all_or_nothing(self):
        c = NutrientCoverage("x", {"energy_kcal", "iron_mg"})
        assert c.covers(["energy_kcal", "iron_mg"])
        assert not c.covers(["energy_kcal", "iron_mg", "vitamin_d_ug"])

    def test_covers_an_empty_requirement_is_trivially_true(self):
        assert NutrientCoverage("x", set()).covers([])

    def test_membership_and_length(self):
        c = NutrientCoverage("x", {"energy_kcal", "fiber_g"})
        assert "energy_kcal" in c
        assert "iron_mg" not in c
        assert len(c) == 2

    def test_unreported_among_is_the_free_function_behind_it(self):
        assert unreported_among({"a"}, ["a", "b"]) == {"b"}


class TestRequiredForCondition:
    def test_it_is_empty_until_conditions_exist(self):
        """Ruling R3: a condition's nutrients follow from rules that cite evidence,
        and no evidence has been ingested yet. Empty, not guessed."""
        assert dict(REQUIRED_FOR_CONDITION) == {}


class TestTheDatabaseStoresRealNulls:
    """The migration's actual outcome, asserted against the built database."""

    def test_nulls_are_stored_rather_than_zeros(self):
        with sqlite3.connect(FOOD_DB_SQLITE) as conn:
            n = conn.execute("SELECT COUNT(*) FROM foods WHERE added_sugars_g IS NULL").fetchone()[
                0
            ]
        assert n > 0, "added_sugars_g has no NULLs -- the zero-fill is back"

    def test_added_sugars_is_unreported_for_every_food(self):
        """Neither USDA release consumed here reports FDC nutrient 1235.

        The rule keyed on it reads a documented proxy (goals.estimate_added_sugars),
        not this column -- so this is honest emptiness, not a broken rule.
        """
        with sqlite3.connect(FOOD_DB_SQLITE) as conn:
            total, nulls = conn.execute(
                "SELECT COUNT(*), SUM(added_sugars_g IS NULL) FROM foods"
            ).fetchone()
        assert nulls == total

    def test_energy_is_reported_for_every_food(self):
        """The build filters on it, so a NULL here means the filter broke."""
        with sqlite3.connect(FOOD_DB_SQLITE) as conn:
            n = conn.execute("SELECT COUNT(*) FROM foods WHERE energy_kcal IS NULL").fetchone()[0]
        assert n == 0

    def test_records_carry_a_coverage_mask(self, db):
        record = db.records[0]
        assert isinstance(record.reported_nutrients, frozenset)
        assert "energy_kcal" in record.reported_nutrients

    def test_no_record_claims_to_report_added_sugars(self, db):
        assert not any(r.reports("added_sugars_g") for r in db.records)

    def test_the_vector_still_carries_zero_for_unreported(self, db):
        """R13's compatibility guarantee, at the record level."""
        record = next(r for r in db.records if not r.reports("added_sugars_g"))
        assert record.nutrients_per_100g.added_sugars_g == 0.0

    def test_unreported_reports_what_a_condition_could_not_answer(self, db):
        record = db.records[0]
        assert record.unreported(["added_sugars_g"]) == {"added_sugars_g"}
