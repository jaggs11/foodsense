"""Tests for the data layer: the curated USDA database, lookup, matching, corpora.

The invariants here are load-bearing for later stages. If ``default_form`` can sit
outside ``allowed_forms``, the optimiser can start from an infeasible point; if
popcorn gains a second allowed form, the toddler safety rule acquires an escape
hatch it was never meant to have.
"""

from __future__ import annotations

import json

import pytest

from foodsense.data.corpora import (
    FOODCOM_SAMPLE,
    KCAL_BAND,
    MAX_ENERGY_DISAGREEMENT,
    MAX_INGREDIENT_G,
    MAX_ITEMS,
    MIN_ITEMS,
    NUTRITION5K_SAMPLE,
    _parse_ingredients,
    _parse_quantity,
    _to_grams,
    load_foodcom,
    load_meals,
    load_nutrition5k,
    recipes_to_meals,
)
from foodsense.data.fdc import (
    DEFAULT_MATCH_THRESHOLD,
    FoodDatabaseMissingError,
    FoodDB,
    get_food_db,
)
from foodsense.schemas import NUTRIENTS, Form, Meal, MealItem


@pytest.fixture(scope="module")
def db():
    return get_food_db()


# ---------------------------------------------------------------------------
# Database shape
# ---------------------------------------------------------------------------


class TestFoodDatabase:
    def test_size_is_in_the_curated_range(self, db):
        """The brief calls for ~2-3k everyday foods, not all 7.8k SR Legacy rows."""
        assert 2000 <= len(db) <= 3000

    def test_ids_are_unique(self, db):
        assert len({r.fdc_id for r in db}) == len(db)

    def test_names_are_unique(self, db):
        """Duplicate names would let the optimiser 'add' a food already in the meal."""
        assert len({r.name for r in db}) == len(db)

    def test_every_food_has_forms_and_a_valid_default(self, db):
        for record in db:
            assert record.allowed_forms, f"{record.name} has no allowed forms"
            assert record.default_form in record.allowed_forms, (
                f"{record.name}: default {record.default_form} not in {record.allowed_forms}"
            )

    def test_nutrients_are_non_negative(self, db):
        for record in db:
            for nutrient in NUTRIENTS:
                assert getattr(record.nutrients_per_100g, nutrient) >= 0.0

    def test_energy_is_populated_and_plausible(self, db):
        """Every curated food carries an energy value; oils top out around 900."""
        for record in db:
            assert 0.0 <= record.nutrients_per_100g.energy_kcal <= 950.0

    def test_categories_cover_the_expected_taxonomy(self, db):
        expected = {"fruit", "vegetable", "grain", "dairy", "meat", "poultry", "fish", "legume"}
        assert expected <= set(db.categories())


# ---------------------------------------------------------------------------
# Safety-critical metadata
# ---------------------------------------------------------------------------


class TestHazardMetadata:
    @pytest.mark.parametrize("hazard", ["popcorn", "marshmallow", "hard_candy", "gum"])
    def test_no_preparation_makes_these_safe(self, db, hazard):
        """These have no safe toddler form, so the search space must offer none.

        If any of them gained a second form, the optimiser could 'fix' a hazard by
        re-forming it instead of removing it -- which is precisely the failure the
        toddler scenario is meant to demonstrate is impossible.
        """
        records = db.by_hazard(hazard)
        assert records, f"no foods carry hazard_class={hazard}"
        for record in records:
            assert record.allowed_forms == (Form.WHOLE,), record.name

    def test_grapes_can_be_quartered(self, db):
        """The worked example turns on grapes having a safe form to move to."""
        grapes = db.by_hazard("grape")
        assert grapes
        for record in grapes:
            assert Form.QUARTERED in record.allowed_forms

    def test_nuts_can_be_ground(self, db):
        nuts = db.by_hazard("nut")
        assert nuts
        for record in nuts:
            assert Form.GROUND in record.allowed_forms

    def test_peanuts_are_classified_as_a_nut_hazard(self, db):
        """FDC files peanuts under legumes; the hazard rule must still catch them."""
        record, score = db.match("peanuts")
        assert record is not None and score >= DEFAULT_MATCH_THRESHOLD
        assert record.hazard_class == "nut"
        assert Form.WHOLE in record.allowed_forms

    def test_nut_butter_can_be_thinly_spread(self, db):
        butters = db.by_hazard("nut_butter")
        assert butters
        for record in butters:
            assert Form.THIN_SPREAD in record.allowed_forms
            assert Form.SPOONFUL in record.allowed_forms

    def test_hot_dogs_have_both_the_unsafe_and_safe_cuts(self, db):
        dogs = db.by_hazard("hot_dog")
        assert dogs
        for record in dogs:
            assert Form.SLICED_ROUNDS in record.allowed_forms  # the banned cut
            assert Form.MINCED in record.allowed_forms  # the safe one

    def test_hard_raw_vegetables_can_be_soft_cooked(self, db):
        vegetables = db.by_hazard("hard_raw_vegetable")
        assert vegetables
        for record in vegetables:
            assert Form.SOFT_COOKED in record.allowed_forms
            assert Form.MASHED in record.allowed_forms

    @pytest.mark.parametrize(
        "tag",
        [
            "grapefruit",
            "alcohol",
            "aged_cheese",
            "cured_meat",
            "high_tyramine",
            "leafy_green_vitk",
            "high_potassium",
            "high_sodium",
            "whole_nut",
            "raw_hard_veg",
        ],
    )
    def test_medication_interaction_tags_are_populated(self, db, tag):
        """Every tag the age/medication rules key on must actually exist in the data."""
        assert db.by_tag(tag), f"no foods carry tag={tag}"

    def test_leafy_green_tag_tracks_the_measured_vitamin_k(self, db):
        for record in db.by_tag("leafy_green_vitk"):
            assert record.nutrients_per_100g.vitamin_k_ug >= 50.0

    def test_grapefruit_tag_is_on_grapefruit(self, db):
        names = {r.name.lower() for r in db.by_tag("grapefruit")}
        assert any("grapefruit" in n for n in names)


class TestScenarioFoodsExist:
    """Curation must never silently delete a food the demo scenarios depend on."""

    @pytest.mark.parametrize(
        "query",
        [
            "grapes",
            "peanuts",
            "white rice",
            "plain yogurt",
            "carrots",
            "ground chicken",
            "low sodium chicken broth",
            "saltine crackers",
            "whole wheat crackers",
            "canned tomato soup",
            "chicken breast",
            "ground beef",
        ],
    )
    def test_scenario_food_is_present(self, db, query):
        record, score = db.search(query, limit=1)[0]
        assert record is not None
        assert score >= 80.0, f"{query!r} only reached {score}"


# ---------------------------------------------------------------------------
# Lookup and matching
# ---------------------------------------------------------------------------


class TestLookup:
    def test_get_and_find(self, db):
        record = next(iter(db))
        assert db.get(record.fdc_id) is record
        assert db.find(record.fdc_id) is record
        assert record.fdc_id in db

    def test_missing_id_raises_but_find_returns_none(self, db):
        with pytest.raises(KeyError):
            db.get("does-not-exist")
        assert db.find("does-not-exist") is None

    def test_missing_database_file_raises_a_helpful_error(self, tmp_path):
        with pytest.raises(FoodDatabaseMissingError, match="build_food_db"):
            FoodDB.load(tmp_path / "nope.parquet")


class TestMatching:
    def test_exact_usda_name_scores_100(self, db):
        for record in list(db)[:50]:
            matched, score = db.match(record.name)
            assert score == pytest.approx(100.0), record.name
            assert matched is not None

    def test_empty_query_returns_nothing(self, db):
        assert db.search("") == []
        assert db.match("")[0] is None

    def test_nonexistent_food_falls_below_threshold(self, db):
        """Stage 4 relies on this: an invented food must not match confidently."""
        for query in ["unicorn steak", "quinoa pilaf with feta and pomegranate"]:
            record, score = db.match(query)
            assert record is None, f"{query!r} wrongly matched {score}"

    @pytest.mark.parametrize(
        ("query", "expected_prefix"),
        [
            ("carrots", "Carrots"),
            ("carrot", "Carrots"),  # singular must not land on "Carrot, dehydrated"
            ("grape", "Grapes"),  # must not land on "Grape leaves"
            ("grapes", "Grapes"),
            ("orange", "Oranges"),  # must not land on "Orange peel"
            ("onions", "Onions"),
            ("broccoli", "Broccoli"),
            ("spinach", "Spinach"),
            ("garlic", "Garlic"),
            ("bananas", "Bananas"),
            ("cheddar cheese", "Cheese, cheddar"),
        ],
    )
    def test_common_names_resolve_to_the_plain_food(self, db, query, expected_prefix):
        record, _ = db.search(query, limit=1)[0]
        assert record.name.startswith(expected_prefix), f"{query!r} -> {record.name}"

    def test_singular_and_plural_agree(self, db):
        assert db.search("carrot", limit=1)[0][0] is db.search("carrots", limit=1)[0][0]

    def test_search_is_ordered_by_descending_score(self, db):
        scores = [s for _, s in db.search("chicken breast", limit=10)]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_is_respected(self, db):
        record, score = db.match("grapes", threshold=99.9)
        assert record is None and score > 80.0


# ---------------------------------------------------------------------------
# Nutrient recomputation -- the Stage-4 ground truth
# ---------------------------------------------------------------------------


class TestNutrientRecomputation:
    def test_quantity_scales_linearly(self, db):
        record = db.search("grapes", limit=1)[0][0]
        at_100 = record.nutrients_for(100.0).energy_kcal
        at_40 = record.nutrients_for(40.0).energy_kcal
        assert at_40 == pytest.approx(at_100 * 0.4)

    def test_meal_totals_are_the_sum_of_items(self, db):
        record = db.search("grapes", limit=1)[0][0]
        meal = Meal(items=[record.as_item(40.0), record.as_item(60.0)])
        assert db.nutrients_for(meal).energy_kcal == pytest.approx(
            record.nutrients_for(100.0).energy_kcal
        )

    def test_form_does_not_change_nutrients(self, db):
        """Quartering a grape does not alter its composition -- form is safety, not nutrition."""
        record = db.search("grapes", limit=1)[0][0]
        whole = MealItem(food_id=record.fdc_id, name=record.name, quantity_g=40, form=Form.WHOLE)
        quartered = whole.model_copy(update={"form": Form.QUARTERED})
        assert (
            db.nutrients_for_item(whole).as_tuple() == db.nutrients_for_item(quartered).as_tuple()
        )

    def test_unknown_food_contributes_zero_and_is_reported(self, db):
        ghost = MealItem(food_id="000000", name="ghost", quantity_g=100)
        assert db.nutrients_for_item(ghost).energy_kcal == 0.0
        assert db.unknown_ids([ghost]) == ["000000"]

    def test_empty_meal_is_all_zeros(self, db):
        assert db.nutrients_for(Meal()).as_tuple() == tuple([0.0] * len(NUTRIENTS))


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------


class TestQuantityParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2", 2.0),
            ("1/4", 0.25),
            ("2 1/2", 2.5),
            ("", 1.0),
            ("some", 1.0),
            ("0", 1.0),
            (None, 1.0),
            (3, 3.0),
        ],
    )
    def test_parse_quantity(self, raw, expected):
        assert _parse_quantity(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (
                '[{"quantity": "4", "unit": "cups", "name": "blueberries"}]',
                [{"quantity": "4", "unit": "cups", "name": "blueberries"}],
            ),
            ("[]", []),
            ("not json", []),
            ("", []),
            (None, []),
        ],
    )
    def test_parse_ingredients(self, raw, expected):
        assert _parse_ingredients(raw) == expected


class TestUnitConversion:
    """Food.com states real units; getting these wrong silently corrupts every meal."""

    def test_mass_units_are_exact(self, db):
        record = db.search("chicken breast", limit=1)[0][0]
        assert _to_grams(1.0, "lb", record) == pytest.approx(453.6)
        assert _to_grams(4.0, "oz", record) == pytest.approx(113.4)
        assert _to_grams(250.0, "g", record) == pytest.approx(250.0)

    def test_volume_units_use_the_food_density(self, db):
        """A cup of oil and a cup of flour do not weigh the same."""
        oil = db.search("olive oil", limit=1)[0][0]
        flour = db.search("whole wheat crackers", limit=1)[0][0]
        assert oil.category == "fat_oil"
        assert _to_grams(1.0, "cup", oil) > _to_grams(1.0, "cup", flour)
        assert _to_grams(1.0, "cup", oil) == pytest.approx(236.6 * 0.92, rel=1e-3)

    def test_teaspoon_is_a_third_of_a_tablespoon(self, db):
        record = db.search("salt", limit=1)[0][0]
        assert _to_grams(3.0, "teaspoon", record) == pytest.approx(
            _to_grams(1.0, "tablespoon", record), rel=0.02
        )

    def test_bare_count_uses_a_per_piece_mass(self, db):
        """ "2 eggs" has no unit, so it has to fall back to the mass of one egg."""
        egg = db.search("egg whole cooked", limit=1)[0][0]
        assert _to_grams(2.0, None, egg) == pytest.approx(100.0)

    def test_size_adjectives_scale_the_piece(self, db):
        onion = db.search("onions", limit=1)[0][0]
        assert _to_grams(1.0, "large", onion) > _to_grams(1.0, "medium", onion)
        assert _to_grams(1.0, "small", onion) < _to_grams(1.0, "medium", onion)

    def test_unknown_unit_degrades_to_a_piece(self, db):
        record = db.search("carrots", limit=1)[0][0]
        assert _to_grams(1.0, "wibble", record) == pytest.approx(_to_grams(1.0, None, record))


class TestCorpora:
    def test_samples_are_committed(self):
        """A clone with no network must still be able to run both corpora."""
        assert FOODCOM_SAMPLE.exists()
        assert NUTRITION5K_SAMPLE.exists()

    def test_foodcom_sample_loads_with_structured_ingredients(self):
        frame = load_foodcom(limit=50)
        assert len(frame) > 0
        assert frame["ingredients"].apply(lambda x: isinstance(x, list)).all()
        first = frame["ingredients"].iloc[0]
        assert first and "name" in first[0]

    def test_nutrition5k_sample_loads_with_gram_amounts(self):
        frame = load_nutrition5k(limit=50)
        assert len(frame) > 0
        first = frame["ingredients"].iloc[0]
        parsed = json.loads(first) if isinstance(first, str) else first
        assert parsed and {"name", "grams"} <= set(parsed[0])

    @pytest.mark.parametrize("source", ["foodcom", "nutrition5k"])
    def test_meals_are_valid_and_resolvable(self, db, source):
        frame = load_foodcom(limit=300) if source == "foodcom" else load_nutrition5k(limit=300)
        meals = recipes_to_meals(frame, source, db=db, limit=25)
        assert meals
        for meal in meals:
            assert db.unknown_ids(meal.items) == []
            assert MIN_ITEMS <= len(meal.items) <= MAX_ITEMS
            assert all(0 < item.quantity_g <= MAX_INGREDIENT_G for item in meal.items)
            assert KCAL_BAND[0] <= meal.nutrients.energy_kcal <= KCAL_BAND[1]
            assert meal.match_confidence >= 60.0

    @pytest.mark.parametrize("source", ["foodcom", "nutrition5k"])
    def test_junk_filter_bounds_energy_disagreement(self, db, source):
        """Meals whose reconstruction contradicts the corpus are dropped, not trained on."""
        frame = load_foodcom(limit=300) if source == "foodcom" else load_nutrition5k(limit=300)
        for meal in recipes_to_meals(frame, source, db=db, limit=40):
            error = meal.reconstruction_error().get("energy_kcal")
            assert error is None or abs(error) <= MAX_ENERGY_DISAGREEMENT

    def test_nutrients_come_from_usda_not_from_the_corpus(self, db):
        """The whole two-corpus comparison rests on this: one nutrient definition."""
        meals = recipes_to_meals(load_nutrition5k(limit=200), "nutrition5k", db=db, limit=5)
        assert meals
        for meal in meals:
            assert meal.nutrients.as_tuple() == db.nutrients_for(meal.items).as_tuple()

    def test_unknown_corpus_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown corpus"):
            load_meals("tastyrecipes")
