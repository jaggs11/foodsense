"""Tests for food validation (requirements section 23, plus provenance from 10).

One rule per test where the rule is a rule. The point of the file is that every
line of section 23 has something that fails when it is violated.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foodsense.data.units import NutrientBasis
from foodsense.data.validation import (
    NUTRIENT_KEYS,
    REQUIRED_NUTRIENTS,
    FoodCreate,
    FoodUpdate,
)


def _valid(**overrides):
    payload = {
        "canonical_name": "Ragi Idli",
        "cuisine": "south_indian",
        "serving_size_g": 100.0,
        "source_type": "indian_dataset",
        "source_reference": "IFCT-style table, page 42",
        "energy_kcal": 120.0,
        "protein_g": 3.0,
    }
    payload.update(overrides)
    return payload


class TestTheHappyPath:
    def test_a_minimal_valid_food_is_accepted(self):
        food = FoodCreate(**_valid())
        assert food.canonical_name == "Ragi Idli"
        assert food.cuisine == "south_indian"

    def test_the_nutrient_set_matches_the_stored_columns(self):
        """33 keys, generated from NutrientVector rather than written out twice."""
        assert len(NUTRIENT_KEYS) == 33
        for key in NUTRIENT_KEYS:
            assert key in FoodCreate.model_fields

    def test_name_key_is_derived_not_supplied(self):
        assert FoodCreate(**_valid(canonical_name="  MASALA-DOSA ")).name_key == "masala dosa"

    def test_whitespace_is_stripped_from_strings(self):
        assert FoodCreate(**_valid(canonical_name="  Idli  ")).canonical_name == "Idli"


class TestNumericRules:
    """Section 23: every nutrient >= 0, serving size > 0, numbers must be numbers."""

    @pytest.mark.parametrize(
        "nutrient",
        ["energy_kcal", "protein_g", "carbohydrate_g", "fat_g", "fiber_g", "sodium_mg"],
    )
    def test_a_negative_nutrient_is_rejected(self, nutrient):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            FoodCreate(**_valid(**{nutrient: -1.0}))

    def test_zero_is_a_legitimate_nutrient_value(self):
        assert FoodCreate(**_valid(fiber_g=0.0)).fiber_g == 0.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, -100.0])
    def test_serving_size_must_be_positive(self, bad):
        with pytest.raises(ValidationError, match="greater than 0"):
            FoodCreate(**_valid(serving_size_g=bad))

    def test_a_non_numeric_nutrient_is_rejected(self):
        with pytest.raises(ValidationError):
            FoodCreate(**_valid(iron_mg="quite a lot"))


class TestRequiredFields:
    @pytest.mark.parametrize(
        "field", ["canonical_name", "cuisine", "serving_size_g", "source_type"]
    )
    def test_a_missing_required_field_is_rejected(self, field):
        payload = _valid()
        del payload[field]
        with pytest.raises(ValidationError):
            FoodCreate(**payload)

    @pytest.mark.parametrize("nutrient", REQUIRED_NUTRIENTS)
    def test_energy_and_protein_are_required(self, nutrient):
        payload = _valid()
        del payload[nutrient]
        with pytest.raises(ValidationError, match="energy and protein"):
            FoodCreate(**payload)

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValidationError):
            FoodCreate(**_valid(canonical_name=""))


class TestUnreportedIsNotZero:
    """Ruling R8, at the boundary where data enters."""

    def test_an_unreported_nutrient_is_none(self):
        food = FoodCreate(**_valid())
        assert food.iron_mg is None
        assert food.calcium_mg is None

    def test_reported_nutrients_lists_only_what_the_source_gave(self):
        food = FoodCreate(**_valid(iron_mg=1.2, fiber_g=0.0))
        assert food.reported_nutrients() == {"energy_kcal", "protein_g", "iron_mg", "fiber_g"}

    def test_a_reported_zero_counts_as_reported(self):
        """The whole distinction rests on this."""
        assert "fiber_g" in FoodCreate(**_valid(fiber_g=0.0)).reported_nutrients()
        assert "fiber_g" not in FoodCreate(**_valid()).reported_nutrients()

    def test_nutrients_returns_every_key_including_unreported(self):
        n = FoodCreate(**_valid()).nutrients()
        assert set(n) == set(NUTRIENT_KEYS)
        assert n["calcium_mg"] is None


class TestEnums:
    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("cuisine", "martian"),
            ("veg_status", "sometimes"),
            ("texture", "crunchy-ish"),
            ("source_type", "a_friend_told_me"),
            ("confidence", "quite_sure"),
            ("verification_status", "probably_fine"),
        ],
    )
    def test_an_out_of_vocabulary_value_is_rejected(self, field, bad):
        with pytest.raises(ValidationError):
            FoodCreate(**_valid(**{field: bad}))

    def test_every_declared_cuisine_is_accepted(self):
        for c in ("north_indian", "south_indian", "pan_indian", "international", "other"):
            assert FoodCreate(**_valid(cuisine=c)).cuisine == c


class TestExtraFieldsAreForbidden:
    def test_an_unknown_field_is_rejected(self):
        with pytest.raises(ValidationError, match=r"[Ee]xtra"):
            FoodCreate(**_valid(vitamin_q_mg=5.0))

    def test_a_misspelled_nutrient_is_rejected_rather_than_dropped(self):
        """Silently dropping this is a food that looks complete and is not."""
        with pytest.raises(ValidationError):
            FoodCreate(**_valid(iron_mgs=1.2))


class TestProvenance:
    """Requirements section 10."""

    def test_user_added_food_is_forced_to_pending(self):
        food = FoodCreate(
            **_valid(
                source_type="user_added", source_reference=None, verification_status="verified"
            )
        )
        assert food.verification_status == "pending"

    def test_user_added_food_does_not_need_a_reference(self):
        food = FoodCreate(**_valid(source_type="user_added", source_reference=None))
        assert food.source_reference is None

    @pytest.mark.parametrize("st", ["indian_dataset", "research_dataset", "government"])
    def test_dataset_sourced_food_must_cite_something(self, st):
        with pytest.raises(ValidationError, match="provenance is mandatory"):
            FoodCreate(**_valid(source_type=st, source_reference=None, source_id=None))

    @pytest.mark.parametrize("st", ["indian_dataset", "research_dataset", "government"])
    def test_a_source_id_alone_satisfies_it(self, st):
        assert FoodCreate(**_valid(source_type=st, source_reference=None, source_id="A-017"))


class TestBasis:
    def test_the_default_basis_is_canonical_storage(self):
        assert FoodCreate(**_valid()).basis is NutrientBasis.PER_100G

    def test_a_per_serving_payload_is_accepted_with_a_serving_size(self):
        food = FoodCreate(**_valid(basis="per_serving", serving_size_g=60.0))
        assert food.basis is NutrientBasis.PER_SERVING


class TestFoodUpdate:
    def test_every_field_is_optional(self):
        assert FoodUpdate().changes() == {}

    def test_changes_reports_only_what_was_supplied(self):
        assert FoodUpdate(region="Karnataka").changes() == {"region": "Karnataka"}

    def test_an_absent_nutrient_is_not_reported_as_a_change(self):
        """Omission must never be read as "clear this value"."""
        assert "iron_mg" not in FoodUpdate(region="Kerala").changes()

    def test_a_supplied_nutrient_is_reported(self):
        assert FoodUpdate(iron_mg=2.0).changes() == {"iron_mg": 2.0}

    def test_the_same_numeric_rules_apply(self):
        with pytest.raises(ValidationError):
            FoodUpdate(iron_mg=-1.0)
        with pytest.raises(ValidationError):
            FoodUpdate(serving_size_g=0.0)

    def test_extra_fields_are_forbidden_here_too(self):
        with pytest.raises(ValidationError):
            FoodUpdate(nonsense=1)
