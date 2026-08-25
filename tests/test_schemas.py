"""Tests for the shared type contract in ``foodsense.schemas``.

These are the invariants every later stage relies on, so they are worth pinning
down before any of those stages exist.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from foodsense.schemas import (
    NUTRIENTS,
    AgeGroup,
    Form,
    Goal,
    HealthFlag,
    Meal,
    MealItem,
    NutrientVector,
    RuleEvaluation,
    UserProfile,
    VerificationReport,
    Violation,
)


class TestNutrientVector:
    def test_defaults_to_zeros(self):
        v = NutrientVector()
        assert v.as_tuple() == tuple([0.0] * len(NUTRIENTS))

    def test_as_tuple_follows_canonical_order(self):
        v = NutrientVector(energy_kcal=100.0, water_g=5.0)
        assert v.as_tuple()[NUTRIENTS.index("energy_kcal")] == 100.0
        assert v.as_tuple()[NUTRIENTS.index("water_g")] == 5.0

    def test_scaled_converts_per_100g_to_absolute(self):
        per_100g = NutrientVector(energy_kcal=69.0, sugars_g=15.5)
        # 40 g of grapes
        got = per_100g.scaled(40.0 / 100.0)
        assert math.isclose(got.energy_kcal, 27.6)
        assert math.isclose(got.sugars_g, 6.2)

    def test_addition_is_elementwise(self):
        a = NutrientVector(protein_g=5.0, sodium_mg=100.0)
        b = NutrientVector(protein_g=3.0, sodium_mg=250.0)
        total = a + b
        assert total.protein_g == 8.0
        assert total.sodium_mg == 350.0

    def test_sum_of_empty_list_is_zeros(self):
        assert NutrientVector.sum([]).as_tuple() == NutrientVector.zeros().as_tuple()

    def test_unknown_nutrient_is_rejected(self):
        with pytest.raises(ValidationError):
            NutrientVector(unobtainium_mg=1.0)


class TestMealItem:
    def test_form_defaults_to_whole(self):
        item = MealItem(food_id="174683", name="Grapes, raw", quantity_g=40.0)
        assert item.form is Form.WHOLE

    def test_food_id_is_coerced_to_string(self):
        """USDA fdc_ids arrive as ints from CSV; the contract stores them as strings."""
        assert MealItem(food_id=174683, name="Grapes", quantity_g=1.0).food_id == "174683"

    def test_negative_quantity_is_rejected(self):
        with pytest.raises(ValidationError):
            MealItem(food_id="1", name="x", quantity_g=-1.0)

    def test_key_pairs_food_with_form(self):
        """Identity is (food, form) -- the same grape in two forms is two different things."""
        whole = MealItem(food_id="174683", name="Grapes", quantity_g=40.0, form=Form.WHOLE)
        quartered = MealItem(food_id="174683", name="Grapes", quantity_g=40.0, form=Form.QUARTERED)
        assert whole.key() != quartered.key()


class TestMeal:
    def test_totals_and_ids(self, toddler_planned_meal):
        assert len(toddler_planned_meal) == 3
        assert toddler_planned_meal.total_quantity_g == 110.0
        assert toddler_planned_meal.food_ids() == {"174683", "172430", "169756"}

    def test_nonzero_drops_optimised_away_items(self):
        meal = Meal(
            items=[
                MealItem(food_id="1", name="kept", quantity_g=10.0),
                MealItem(food_id="2", name="dropped", quantity_g=0.0),
            ]
        )
        assert [i.name for i in meal.nonzero()] == ["kept"]


class TestUserProfile:
    def test_defaults_are_adult_balanced(self):
        p = UserProfile()
        assert p.age_group is AgeGroup.ADULT
        assert p.goal is Goal.BALANCED_NUTRITION
        assert p.health_flags == []

    def test_age_years_derived_from_months(self, toddler_profile):
        assert toddler_profile.age_years == 1.5

    def test_age_years_is_none_without_months(self):
        assert UserProfile().age_years is None

    def test_has_flag(self, toddler_profile):
        assert toddler_profile.has(HealthFlag.IRON_FOCUS)
        assert not toddler_profile.has(HealthFlag.WARFARIN)


class TestRuleEvaluation:
    def test_hard_violation_makes_meal_unsafe(self):
        ev = RuleEvaluation(
            score=0.05,
            violations=[
                Violation(
                    rule_id="toddler.choking.grape_whole",
                    severity="hard",
                    message="Whole grapes are a choking hazard under 4 years.",
                    offending_items=["174683"],
                    suggested_form=Form.QUARTERED,
                )
            ],
        )
        assert not ev.is_safe
        assert len(ev.hard_violations) == 1
        assert ev.hard_violations[0].suggested_form is Form.QUARTERED

    def test_soft_violations_leave_meal_safe(self):
        ev = RuleEvaluation(
            score=0.7,
            violations=[
                Violation(
                    rule_id="goal.balanced.fat_share",
                    severity="soft",
                    message="Fat share slightly above band.",
                )
            ],
        )
        assert ev.is_safe

    def test_score_outside_unit_interval_is_rejected(self):
        with pytest.raises(ValidationError):
            RuleEvaluation(score=1.4)


class TestVerificationReport:
    def test_clean_report_reports_no_issue(self):
        assert not VerificationReport(checked=3, matched=3, final_pass=True).had_issue

    def test_unmatched_item_counts_as_an_issue(self):
        report = VerificationReport(checked=3, matched=2, unmatched=["quinoa pilaf"])
        assert report.had_issue
        assert report.n_corrections == 0
