"""Tests for nutrient bases (requirements section 27).

The error these prevent does not look like an error. A per-serving figure compared
against a per-100 g one produces a number, a chart and a confident sentence, all of
which are wrong and none of which look it. So the guard raises rather than coercing.
"""

from __future__ import annotations

import pytest

from foodsense.data.units import (
    BasisMismatchError,
    NutrientBasis,
    convert,
    per_serving_factor,
    require_same_basis,
    to_per_100g,
    to_per_serving,
)


class TestConversion:
    def test_per_100g_is_the_identity(self):
        assert to_per_100g(120.0, NutrientBasis.PER_100G, None) == 120.0

    def test_per_serving_to_per_100g(self):
        """A 50 g serving holding 60 kcal is 120 kcal per 100 g."""
        assert to_per_100g(60.0, NutrientBasis.PER_SERVING, 50.0) == pytest.approx(120.0)

    def test_per_100g_to_per_serving(self):
        assert to_per_serving(120.0, NutrientBasis.PER_100G, 50.0) == pytest.approx(60.0)

    def test_a_round_trip_returns_the_original(self):
        there = to_per_serving(137.5, NutrientBasis.PER_100G, 43.0)
        back = to_per_100g(there, NutrientBasis.PER_SERVING, 43.0)
        assert back == pytest.approx(137.5)

    def test_convert_between_the_same_basis_is_a_noop(self):
        assert convert(9.0, frm=NutrientBasis.PER_100G, to=NutrientBasis.PER_100G) == 9.0

    def test_convert_dispatches_both_ways(self):
        assert convert(
            60.0, frm=NutrientBasis.PER_SERVING, to=NutrientBasis.PER_100G, serving_size_g=50.0
        ) == pytest.approx(120.0)
        assert convert(
            120.0, frm=NutrientBasis.PER_100G, to=NutrientBasis.PER_SERVING, serving_size_g=50.0
        ) == pytest.approx(60.0)

    def test_the_serving_factor_is_grams_over_100(self):
        assert per_serving_factor(50.0) == pytest.approx(0.5)
        assert per_serving_factor(100.0) == pytest.approx(1.0)


class TestMissingValuesStayMissing:
    """Ruling R8: a nutrient the source does not report is never a zero."""

    def test_none_survives_conversion_to_per_100g(self):
        assert to_per_100g(None, NutrientBasis.PER_SERVING, 50.0) is None

    def test_none_survives_conversion_to_per_serving(self):
        assert to_per_serving(None, NutrientBasis.PER_100G, 50.0) is None

    def test_none_survives_convert(self):
        assert (
            convert(
                None,
                frm=NutrientBasis.PER_SERVING,
                to=NutrientBasis.PER_100G,
                serving_size_g=50.0,
            )
            is None
        )

    def test_a_reported_zero_is_preserved_as_zero(self):
        """The distinction only means something if a real zero survives too."""
        assert to_per_100g(0.0, NutrientBasis.PER_SERVING, 50.0) == 0.0


class TestConversionsThatMustFail:
    def test_per_serving_without_a_serving_size_is_not_a_measurement(self):
        with pytest.raises(ValueError, match="serving_size_g"):
            to_per_100g(60.0, NutrientBasis.PER_SERVING, None)

    def test_converting_to_per_serving_needs_a_serving_size(self):
        with pytest.raises(ValueError, match="serving_size_g"):
            to_per_serving(120.0, NutrientBasis.PER_100G, None)

    @pytest.mark.parametrize("bad", [0.0, -1.0, -50.0])
    def test_a_nonpositive_serving_size_is_rejected(self, bad):
        with pytest.raises(ValueError, match="must be positive"):
            per_serving_factor(bad)

    def test_an_unknown_basis_cannot_be_converted(self):
        with pytest.raises(BasisMismatchError):
            to_per_100g(60.0, NutrientBasis.UNKNOWN, 50.0)


class TestTheComparisonGuard:
    def test_one_basis_passes_and_is_returned(self):
        got = require_same_basis(NutrientBasis.PER_100G, NutrientBasis.PER_100G)
        assert got is NutrientBasis.PER_100G

    def test_mixed_bases_raise(self):
        with pytest.raises(BasisMismatchError, match="different bases"):
            require_same_basis(NutrientBasis.PER_100G, NutrientBasis.PER_SERVING)

    def test_an_unknown_basis_poisons_an_otherwise_consistent_comparison(self):
        """Excluding unverified data from recommendation, made concrete."""
        with pytest.raises(BasisMismatchError, match="unknown basis"):
            require_same_basis(
                NutrientBasis.PER_100G, NutrientBasis.PER_100G, NutrientBasis.UNKNOWN
            )

    def test_no_bases_at_all_is_a_programming_error(self):
        with pytest.raises(ValueError, match="no bases"):
            require_same_basis()

    def test_the_message_names_both_bases(self):
        """The person reading the traceback needs to know which two."""
        with pytest.raises(BasisMismatchError) as exc:
            require_same_basis(NutrientBasis.PER_100G, NutrientBasis.PER_SERVING)
        assert "per_100g_edible_portion" in str(exc.value)
        assert "per_serving" in str(exc.value)
