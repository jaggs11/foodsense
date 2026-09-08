"""Tests for FoodRepository (P1.7), including the two equivalence guarantees.

The refactor's whole claim is that it changed nothing. Two tests carry that claim:
the BM25 retriever returns byte-identical candidates for the three demo scenarios
against a baseline captured before the move, and the golden worked-example traces
pass unmodified (in ``test_worked_examples.py``, untouched by this phase).

The rest assert the behaviour the repository adds -- visibility filtering that is a
no-op today because every row is migrated USDA, and will not be from P6 onward.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foodsense.data.fdc import get_food_db
from foodsense.data.repository import (
    RECOMMENDABLE_STATUSES,
    FoodFilters,
    FoodRepository,
    get_food_repository,
)
from foodsense.scenarios import SCENARIOS, load_scenario
from foodsense.stage3_rag.retriever import get_retriever

BASELINE = Path(__file__).parent / "data" / "bm25_baseline.json"


@pytest.fixture(scope="module")
def repo():
    return get_food_repository()


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE.read_text(encoding="utf-8"))


class TestRetrieverEquivalence:
    """Captured before FoodRepository existed; must still hold after it.

    If these fail, the repository changed what Stage 3 sees -- which would move
    every downstream explanation, silently.
    """

    def test_the_baseline_covers_all_three_demo_scenarios(self, baseline):
        assert set(baseline) == set(SCENARIOS)

    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_candidates_for_is_unchanged(self, scenario, baseline):
        meal = load_scenario(scenario).planned_meal()
        names = [i.name for i in meal.items]
        assert names == baseline[scenario]["query_names"], "the scenario itself changed"
        got = get_retriever().candidates_for(names, k=5)
        assert got == baseline[scenario]["candidates_for"]

    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_ranked_ids_and_scores_are_unchanged(self, scenario, baseline):
        """Names alone would miss a re-ranking between two identically named foods."""
        retriever = get_retriever()
        for name, expected in baseline[scenario]["search"].items():
            got = [
                [h.record.fdc_id, h.record.name, round(h.score, 6)]
                for h in retriever.search(name, k=5)
            ]
            assert got == expected, f"retrieval moved for {name!r}"


class TestVisibility:
    def test_only_verified_rows_are_recommendable(self, repo):
        assert all(r.verification_status in RECOMMENDABLE_STATUSES for r in repo.records())

    def test_include_unverified_widens_the_set(self, repo):
        assert len(repo.records(include_unverified=True)) >= len(repo.records())

    def test_the_migrated_corpus_is_entirely_verified_usda(self, repo):
        """True today, and the reason the filter is currently a no-op."""
        assert repo.source_types() == {"usda": 2590}
        assert repo.count() == 2590

    def test_get_is_not_visibility_filtered(self, repo):
        """A trace that named a food must resolve it even once it is rejected."""
        any_id = repo.records()[0].fdc_id
        assert repo.get(any_id) is not None

    def test_get_returns_none_for_an_unknown_id(self, repo):
        assert repo.get("no-such-food") is None
        assert not repo.exists("no-such-food")


class TestFilters:
    def test_an_empty_filter_matches_everything(self, repo):
        assert len(repo.records(filters=FoodFilters())) == len(repo.records())

    def test_cuisine_filters(self, repo):
        assert len(repo.records(filters=FoodFilters(cuisine="international"))) == 2590
        assert repo.records(filters=FoodFilters(cuisine="north_indian")) == []

    def test_source_type_filters(self, repo):
        assert len(repo.by_source("usda")) == 2590
        assert repo.by_source("indian_dataset") == []

    def test_a_sequence_of_values_is_an_or(self, repo):
        both = repo.records(filters=FoodFilters(cuisine=["international", "north_indian"]))
        assert len(both) == 2590

    def test_requires_nutrients_uses_coverage_not_value(self, repo):
        """The distinction R8 exists for: reported, not merely non-zero."""
        need_d = repo.records(filters=FoodFilters(requires_nutrients=["vitamin_d_ug"]))
        assert 0 < len(need_d) < 2590
        assert all(r.reports("vitamin_d_ug") for r in need_d)

    def test_a_nutrient_no_source_reports_yields_nothing(self, repo):
        """added_sugars_g is unreported for every row, so nothing can satisfy it."""
        assert repo.records(filters=FoodFilters(requires_nutrients=["added_sugars_g"])) == []

    def test_search_respects_filters(self, repo):
        assert repo.search("rice", limit=5, filters=FoodFilters(cuisine="north_indian")) == []

    def test_search_is_unfiltered_by_default(self, repo):
        assert repo.search("rice", limit=5)


class TestNutrientAccess:
    def test_the_matrix_is_row_aligned_and_33_wide(self, repo):
        ids = [r.fdc_id for r in repo.records()[:4]]
        matrix = repo.nutrient_matrix(ids)
        assert matrix.shape == (4, 33)

    def test_an_unknown_id_contributes_a_zero_row(self, repo):
        matrix = repo.nutrient_matrix(["no-such-food"])
        assert matrix.shape == (1, 33)
        assert not matrix.any()

    def test_coverage_of_an_unknown_food_reports_nothing(self, repo):
        """Which is true, and is not the same as reporting zeros."""
        assert len(repo.coverage("no-such-food")) == 0

    def test_coverage_of_a_real_food_reports_energy(self, repo):
        assert repo.coverage(repo.records()[0].fdc_id).reports("energy_kcal")


class TestConstruction:
    def test_an_injected_db_is_used(self):
        db = get_food_db()
        assert FoodRepository(db).db is db

    def test_the_shared_repository_is_cached(self):
        assert get_food_repository() is get_food_repository()
