"""Tests for the constraint layer -- the safety-critical part of the project.

The choking and medication tests are **parametrised over the YAML itself** rather
than over a hand-written list. Adding a ban to ``configs/age_groups/toddler.yaml``
or a rule to ``configs/health_flags.yaml`` therefore automatically adds a test
case, and a rule that no food in the database can trigger fails loudly instead of
sitting there looking enforced.
"""

from __future__ import annotations

import pytest

from foodsense.constraints.age_rules import (
    check_age_gated_foods,
    check_choking,
    check_excluded_tags,
    check_texture,
    implicit_flags,
    load_age_config,
    load_flag_rules,
    nearest_safe_form,
    permitted_forms,
)
from foodsense.constraints.engine import RuleEngine, RuleEngineConfig
from foodsense.constraints.goals import (
    GI_BY_CATEGORY,
    Threshold,
    energy_shares,
    estimate_added_sugars,
    estimate_glycemic_load,
    load_goal_config,
    meal_metrics,
    satisfaction,
)
from foodsense.data.fdc import get_food_db
from foodsense.schemas import AgeGroup, Form, Goal, HealthFlag, Meal, NutrientVector, UserProfile


@pytest.fixture(scope="module")
def db():
    return get_food_db()


@pytest.fixture(scope="module")
def engine(db):
    return RuleEngine(db=db)


@pytest.fixture(scope="module")
def toddler_bans():
    return load_age_config(AgeGroup.TODDLER).active_bans()


def _toddler(age_months: int = 18, **kwargs) -> UserProfile:
    return UserProfile(age_group=AgeGroup.TODDLER, age_months=age_months, weight_kg=11.0, **kwargs)


def _older_adult(**kwargs) -> UserProfile:
    return UserProfile(age_group=AgeGroup.OLDER_ADULT, age_months=78 * 12, weight_kg=70.0, **kwargs)


def _sugary(db):
    """A food the added-sugar proxy applies to *and* that has a sugars value.

    USDA reports sugars for only 57% of the curated database, so picking the
    first tagged row is not safe -- it may be one of the gaps.
    """
    return max(
        (r for r in db.by_tag("added_sugar_source")),
        key=lambda r: r.nutrients_per_100g.sugars_g,
    )


def _staple(db):
    """A safe, unremarkable filler item so test meals are not single-item."""
    return db.search("rice white long grain cooked", limit=1)[0][0].as_item(50.0)


# ---------------------------------------------------------------------------
# Threshold mathematics
# ---------------------------------------------------------------------------


class TestSatisfaction:
    def test_exactly_half_at_a_ceiling(self):
        """Softening must not move the boundary, only blur the fall-off."""
        assert satisfaction(500.0, Threshold(maximum=500.0), 0.15) == pytest.approx(0.5)

    def test_exactly_half_at_a_floor(self):
        assert satisfaction(25.0, Threshold(minimum=25.0), 0.15) == pytest.approx(0.5)

    def test_ceiling_is_monotonically_decreasing(self):
        threshold = Threshold(maximum=500.0)
        scores = [satisfaction(v, threshold, 0.15) for v in (200, 400, 500, 600, 900)]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > 0.95 and scores[-1] < 0.01

    def test_floor_is_monotonically_increasing(self):
        threshold = Threshold(minimum=25.0)
        scores = [satisfaction(v, threshold, 0.15) for v in (5, 15, 25, 35, 60)]
        assert scores == sorted(scores)

    def test_band_peaks_inside_and_falls_off_both_sides(self):
        """A value in the middle of a band must score near 1, not near a half.

        Regression test: softening each bound by its own magnitude made the
        acceptable fat share (25-35% of energy) unsatisfiable, scoring 0.57 at a
        perfect 30%.
        """
        threshold = Threshold(minimum=0.25, maximum=0.35)
        assert satisfaction(0.30, threshold, 0.15) > 0.9
        assert satisfaction(0.10, threshold, 0.15) < 0.1
        assert satisfaction(0.55, threshold, 0.15) < 0.1

    def test_band_edges_are_still_exactly_a_half(self):
        threshold = Threshold(minimum=0.25, maximum=0.35)
        assert satisfaction(0.25, threshold, 0.15) == pytest.approx(0.5, abs=0.01)
        assert satisfaction(0.35, threshold, 0.15) == pytest.approx(0.5, abs=0.01)

    def test_unbounded_threshold_is_always_satisfied(self):
        assert satisfaction(1e9, Threshold(), 0.15) == 1.0

    def test_zero_threshold_does_not_divide_by_zero(self):
        assert 0.0 <= satisfaction(5.0, Threshold(maximum=0.0), 0.15) <= 1.0

    def test_extreme_values_saturate_without_overflow(self):
        assert satisfaction(1e12, Threshold(maximum=1.0), 0.15) == 0.0
        assert satisfaction(-1e12, Threshold(minimum=1.0), 0.15) == 0.0

    def test_is_satisfied_uses_the_hard_boundary(self):
        """Validity is decided by the threshold, never by the softened score."""
        threshold = Threshold(maximum=500.0)
        assert threshold.is_satisfied(500.0)
        assert not threshold.is_satisfied(500.001)

    def test_floor_attainment_scales_only_the_minimum(self):
        scaled = Threshold(minimum=1000.0, maximum=2300.0).scaled(1 / 3, floor_attainment=0.6)
        assert scaled.minimum == pytest.approx(1000.0 / 3 * 0.6)
        assert scaled.maximum == pytest.approx(2300.0 / 3)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfigs:
    @pytest.mark.parametrize("age_group", list(AgeGroup))
    def test_every_age_group_has_a_config(self, age_group):
        config = load_age_config(age_group)
        assert config.age_group is age_group
        assert config.meals_per_day > 0
        assert 0 < config.per_meal_floor_attainment <= 1.0
        assert config.daily

    @pytest.mark.parametrize("goal", list(Goal))
    def test_every_goal_has_a_config_with_rules(self, goal):
        config = load_goal_config(goal)
        assert config.goal is goal
        assert config.rules()

    @pytest.mark.parametrize("flag", list(HealthFlag))
    def test_every_health_flag_has_a_rule(self, flag):
        """A flag the schema allows but no config implements would silently do nothing."""
        assert flag in load_flag_rules()

    def test_only_toddlers_carry_choking_bans(self):
        assert load_age_config(AgeGroup.TODDLER).active_bans()
        assert load_age_config(AgeGroup.ADULT).active_bans() == ()
        assert load_age_config(AgeGroup.OLDER_ADULT).active_bans() == ()

    def test_every_gi_category_is_a_real_database_category(self, db):
        assert set(GI_BY_CATEGORY) <= set(db.categories())


# ---------------------------------------------------------------------------
# Choking hazards -- one case per (hazard_class, banned form) pair in the YAML
# ---------------------------------------------------------------------------


def _choking_cases():
    """Every (hazard_class, banned form) pair the toddler config actually forbids.

    A wildcard ban covers all twelve forms, but popcorn only ever exists as
    ``whole``. Intersecting with the forms a real database food can take keeps
    each generated case a genuine test rather than a skip.
    """
    database = get_food_db()
    cases = []
    for ban in load_age_config(AgeGroup.TODDLER).active_bans():
        records = database.by_hazard(ban.hazard_class)
        available = {form for record in records for form in record.allowed_forms}
        banned = set(ban.banned_forms) if ban.banned_forms is not None else set(Form)
        for form in sorted(banned & available, key=lambda f: f.value):
            cases.append(pytest.param(ban.hazard_class, form, id=f"{ban.hazard_class}-{form}"))
    assert cases, "no choking ban is reachable with the curated database"
    return cases


class TestChokingHazards:
    @pytest.mark.parametrize(("hazard_class", "form"), _choking_cases())
    def test_every_banned_pair_is_caught(self, db, hazard_class, form):
        """The core safety guarantee: each (hazard_class, form) ban actually fires."""
        records = [r for r in db.by_hazard(hazard_class) if r.permits(form)]
        if not records:
            pytest.skip(f"no curated food takes {hazard_class} in form {form}")
        meal = Meal(items=[records[0].as_item(30.0, form), _staple(db)])

        violations = check_choking(meal, _toddler(), db)
        assert violations, f"{hazard_class} in form {form} was not flagged"
        assert violations[0].rule_id == f"toddler.choking.{hazard_class}"

    @pytest.mark.parametrize(("hazard_class", "form"), _choking_cases())
    def test_every_banned_pair_drives_the_score_toward_zero(self, engine, db, hazard_class, form):
        records = [r for r in db.by_hazard(hazard_class) if r.permits(form)]
        if not records:
            pytest.skip(f"no curated food takes {hazard_class} in form {form}")
        meal = Meal(items=[records[0].as_item(30.0, form), _staple(db)])

        evaluation = engine.evaluate(meal, _toddler())
        assert not evaluation.is_safe
        assert evaluation.score < 0.1 * evaluation.soft_score + 1e-9

    def test_quartering_grapes_removes_the_hazard(self, db):
        """The worked example: a form change, not a removal, is the repair."""
        grapes = db.by_hazard("grape")[0]
        profile = _toddler()
        assert check_choking(Meal(items=[grapes.as_item(40, Form.WHOLE)]), profile, db)
        assert not check_choking(Meal(items=[grapes.as_item(40, Form.QUARTERED)]), profile, db)

    def test_grape_violation_suggests_quartered(self, db):
        grapes = db.by_hazard("grape")[0]
        violation = check_choking(Meal(items=[grapes.as_item(40, Form.WHOLE)]), _toddler(), db)[0]
        assert violation.suggested_form is Form.QUARTERED
        assert not violation.removable_only

    @pytest.mark.parametrize("hazard_class", ["popcorn", "marshmallow", "hard_candy", "gum"])
    def test_hazards_with_no_safe_form_can_only_be_removed(self, db, hazard_class):
        record = db.by_hazard(hazard_class)[0]
        meal = Meal(items=[record.as_item(20.0, record.default_form)])
        violation = check_choking(meal, _toddler(), db)[0]
        assert violation.suggested_form is None
        assert violation.removable_only
        assert nearest_safe_form(meal.items[0], _toddler(), db) is None

    def test_whole_nuts_have_no_safe_form_at_eighteen_months(self, db):
        """The toddler scenario turns on this: the peanuts must be substituted, not re-formed."""
        nut = db.by_hazard("nut")[0]
        violation = check_choking(Meal(items=[nut.as_item(20, Form.WHOLE)]), _toddler(18), db)[0]
        assert violation.suggested_form is None
        assert violation.removable_only

    def test_whole_nuts_can_be_ground_for_an_older_toddler(self, db):
        nut = db.by_hazard("nut")[0]
        violation = check_choking(Meal(items=[nut.as_item(20, Form.WHOLE)]), _toddler(30), db)[0]
        assert violation.suggested_form is Form.GROUND
        assert not violation.removable_only

    def test_unknown_age_is_treated_as_the_youngest(self, db):
        """Guessing wrong here costs a choking hazard, so the rule refuses to guess up."""
        nut = db.by_hazard("nut")[0]
        profile = UserProfile(age_group=AgeGroup.TODDLER, age_months=None)
        violation = check_choking(Meal(items=[nut.as_item(20, Form.WHOLE)]), profile, db)[0]
        assert violation.suggested_form is None

    def test_a_suggested_form_is_always_one_the_food_can_take(self, db):
        profile = _toddler(30)
        for ban in load_age_config(AgeGroup.TODDLER).active_bans():
            for record in db.by_hazard(ban.hazard_class):
                form = next(
                    (f for f in (ban.banned_forms or tuple(Form)) if record.permits(f)), None
                )
                if form is None:
                    continue
                violations = check_choking(Meal(items=[record.as_item(30, form)]), profile, db)
                for violation in violations:
                    if violation.suggested_form is not None:
                        assert record.permits(violation.suggested_form), record.name

    @pytest.mark.parametrize("age_group", [AgeGroup.ADULT, AgeGroup.OLDER_ADULT])
    def test_adults_have_no_choking_bans(self, db, age_group):
        grapes = db.by_hazard("grape")[0]
        profile = UserProfile(age_group=age_group, weight_kg=70.0)
        assert check_choking(Meal(items=[grapes.as_item(40, Form.WHOLE)]), profile, db) == []


# ---------------------------------------------------------------------------
# Medication and condition rules -- one case per rule in health_flags.yaml
# ---------------------------------------------------------------------------


class TestMedicationInteractions:
    @pytest.mark.parametrize(
        "flag",
        [f for f, r in load_flag_rules().items() if r.exclude_tags],
        ids=lambda f: f.value,
    )
    def test_every_exclusion_rule_fires_on_a_tagged_food(self, db, flag):
        rule = load_flag_rules()[flag]
        tag = next(t for t in sorted(rule.exclude_tags) if db.by_tag(t))
        offending = db.by_tag(tag)[0]
        meal = Meal(items=[offending.as_item(60.0), _staple(db)])

        violations = check_excluded_tags(meal, _older_adult(health_flags=[flag]), db)
        assert violations, f"{flag.value} did not exclude a {tag} food"
        assert violations[0].rule_id == f"flag.{flag.value}.excluded_food"
        assert violations[0].removable_only

    @pytest.mark.parametrize(
        "flag",
        [f for f, r in load_flag_rules().items() if r.exclude_tags],
        ids=lambda f: f.value,
    )
    def test_exclusions_do_not_fire_without_the_flag(self, db, flag):
        rule = load_flag_rules()[flag]
        tag = next(t for t in sorted(rule.exclude_tags) if db.by_tag(t))
        meal = Meal(items=[db.by_tag(tag)[0].as_item(60.0), _staple(db)])
        assert check_excluded_tags(meal, _older_adult(), db) == []

    @pytest.mark.parametrize(
        ("flag", "quantity"),
        [(f, q) for f, r in load_flag_rules().items() for q in r.per_meal_max],
        ids=lambda v: v.value if isinstance(v, HealthFlag) else str(v),
    )
    def test_every_numeric_ceiling_fires_when_exceeded(self, engine, db, flag, quantity):
        """Constructed to breach the ceiling, then checked that the rule notices."""
        limit = load_flag_rules()[flag].per_meal_max[quantity]
        profile = _older_adult(health_flags=[flag])

        # Find a food rich in the quantity and scale it past the limit.
        best = max(
            db.records,
            key=lambda r: meal_metrics([r.as_item(100.0)], db).get(quantity, 0.0),
        )
        per_100g = meal_metrics([best.as_item(100.0)], db).get(quantity, 0.0)
        if per_100g <= 0:
            pytest.skip(f"no curated food supplies {quantity}")
        grams = min(max(limit / per_100g * 100.0 * 1.6, 10.0), 2000.0)

        evaluation = engine.evaluate(Meal(items=[best.as_item(grams)]), profile)
        assert any(v.rule_id == f"flag.{flag.value}.{quantity}" for v in evaluation.violations), (
            f"{flag.value}/{quantity} not flagged at {grams:.0f} g of {best.name}"
        )

    def test_warfarin_caps_vitamin_k(self, engine, db):
        spinach = db.search("spinach raw", limit=1)[0][0]
        meal = Meal(items=[spinach.as_item(200.0), _staple(db)])
        evaluation = engine.evaluate(meal, _older_adult(health_flags=[HealthFlag.WARFARIN]))
        violation = next(v for v in evaluation.violations if v.rule_id.startswith("flag.warfarin"))
        assert violation.severity == "hard"
        assert violation.observed > violation.threshold
        assert not evaluation.is_safe

    def test_statin_excludes_grapefruit(self, engine, db):
        grapefruit = db.by_tag("grapefruit")[0]
        meal = Meal(items=[grapefruit.as_item(150.0), _staple(db)])
        assert not engine.evaluate(meal, _older_adult(health_flags=[HealthFlag.STATIN])).is_safe
        assert engine.evaluate(meal, _older_adult()).is_safe

    def test_hypertension_is_soft_not_a_safety_hazard(self, engine, db):
        """High sodium is a goal failure, not an immediate danger; the severity says so."""
        salty = max(db.records, key=lambda r: r.nutrients_per_100g.sodium_mg)
        meal = Meal(items=[salty.as_item(80.0), _staple(db)])
        evaluation = engine.evaluate(meal, _older_adult(health_flags=[HealthFlag.HYPERTENSION]))
        violation = next(
            v for v in evaluation.violations if v.rule_id == "flag.hypertension.sodium_mg"
        )
        assert violation.severity == "soft"
        assert evaluation.is_safe


class TestTexture:
    def test_dysphagia_restricts_forms(self, db):
        profile = _older_adult(health_flags=[HealthFlag.DYSPHAGIA])
        allowed = permitted_forms(profile)
        assert allowed == frozenset({Form.MINCED, Form.MASHED, Form.PUREED, Form.SOFT_COOKED})

    def test_unsafe_texture_is_flagged_and_repaired(self, db):
        profile = _older_adult(health_flags=[HealthFlag.DYSPHAGIA])
        chicken = db.search("chicken breast meat only roasted", limit=1)[0][0]
        item = chicken.as_item(120.0, Form.WHOLE)

        violations = check_texture(Meal(items=[item]), profile, db)
        assert violations and violations[0].rule_id == "flag.dysphagia.texture"
        assert violations[0].suggested_form in permitted_forms(profile)
        assert nearest_safe_form(item, profile, db) in permitted_forms(profile)

    def test_permitted_forms_are_unrestricted_without_the_flag(self):
        assert permitted_forms(_older_adult()) is None

    def test_a_compliant_texture_passes(self, db):
        profile = _older_adult(health_flags=[HealthFlag.DYSPHAGIA])
        carrots = db.search("carrots cooked", limit=1)[0][0]
        assert check_texture(Meal(items=[carrots.as_item(80, Form.MASHED)]), profile, db) == []


class TestAgeGatedFoods:
    def test_honey_is_barred_below_twelve_months(self, db):
        honey = db.by_tag("honey")[0]
        meal = Meal(items=[honey.as_item(10.0), _staple(db)])
        assert check_age_gated_foods(meal, _toddler(9), db)
        assert not check_age_gated_foods(meal, _toddler(18), db)

    def test_added_sugar_rule_is_implied_below_two_years(self):
        assert HealthFlag.STRICT_NO_ADDED_SUGAR in implicit_flags(_toddler(18))
        assert HealthFlag.STRICT_NO_ADDED_SUGAR not in implicit_flags(_toddler(30))

    def test_implied_flag_actually_constrains(self, engine, db):
        meal = Meal(items=[_sugary(db).as_item(60.0), _staple(db)])
        assert not engine.evaluate(meal, _toddler(18)).is_safe


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


class TestDerivedMetrics:
    def test_glycemic_load_uses_available_carbohydrate(self, db):
        rice = db.search("rice white long grain cooked", limit=1)[0][0]
        per_100g = rice.nutrients_per_100g
        expected = (
            GI_BY_CATEGORY["grain"]
            * max(per_100g.carbohydrate_g - per_100g.fiber_g, 0.0)
            * 1.5
            / 100.0
        )
        assert estimate_glycemic_load([rice.as_item(150.0)], db) == pytest.approx(expected)

    def test_glycemic_load_scales_with_quantity(self, db):
        rice = db.search("rice white long grain cooked", limit=1)[0][0]
        assert estimate_glycemic_load([rice.as_item(200.0)], db) == pytest.approx(
            2 * estimate_glycemic_load([rice.as_item(100.0)], db)
        )

    def test_meat_contributes_no_glycemic_load(self, db):
        chicken = db.search("chicken breast meat only roasted", limit=1)[0][0]
        assert estimate_glycemic_load([chicken.as_item(150.0)], db) == pytest.approx(0.0)

    def test_added_sugar_proxy_applies_to_confectionery_not_fruit(self, db):
        candy = _sugary(db)
        grapes = db.by_hazard("grape")[0]
        assert estimate_added_sugars([candy.as_item(50.0)], db) > 0
        assert estimate_added_sugars([grapes.as_item(100.0)], db) == pytest.approx(0.0)

    def test_energy_shares_sum_to_about_one(self, db):
        meal = [
            db.search("chicken breast meat only roasted", limit=1)[0][0].as_item(120.0),
            db.search("rice brown long grain cooked", limit=1)[0][0].as_item(150.0),
        ]
        shares = energy_shares(db.nutrients_for(meal))
        assert 0.85 <= sum(shares.values()) <= 1.15

    def test_energy_shares_of_an_empty_meal_are_zero(self):
        assert set(energy_shares(NutrientVector()).values()) == {0.0}

    def test_meal_metrics_exposes_everything_rules_reference(self, engine, db):
        profile = _older_adult(health_flags=[HealthFlag.WARFARIN, HealthFlag.HYPERTENSION])
        metrics = meal_metrics([db.records[0].as_item(100.0)], db)
        for rule in engine.rules_for(profile):
            assert rule.quantity in metrics, f"no metric for rule {rule.rule_id}"


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------


class TestRuleEngine:
    def test_score_is_soft_score_when_nothing_hard_is_broken(self, engine, db):
        meal = Meal(items=[db.search("carrots cooked", limit=1)[0][0].as_item(100.0)])
        evaluation = engine.evaluate(meal, _older_adult())
        assert evaluation.is_safe
        assert evaluation.score == pytest.approx(evaluation.soft_score)

    def test_each_hard_violation_multiplies_the_score_down(self, db):
        engine = RuleEngine(db=db, config=RuleEngineConfig(hard_violation_factor=0.1))
        grapes = db.by_hazard("grape")[0]
        popcorn = db.by_hazard("popcorn")[0]
        profile = _toddler()

        one = engine.evaluate(Meal(items=[grapes.as_item(40, Form.WHOLE), _staple(db)]), profile)
        two = engine.evaluate(
            Meal(items=[grapes.as_item(40, Form.WHOLE), popcorn.as_item(15), _staple(db)]),
            profile,
        )
        assert one.score > two.score
        assert len(two.hard_violations) == 2

    def test_repairing_a_hazard_raises_the_score(self, engine, db):
        grapes = db.by_hazard("grape")[0]
        profile = _toddler()
        unsafe = engine.evaluate(Meal(items=[grapes.as_item(40, Form.WHOLE), _staple(db)]), profile)
        safe = engine.evaluate(
            Meal(items=[grapes.as_item(40, Form.QUARTERED), _staple(db)]), profile
        )
        assert safe.score > unsafe.score
        assert safe.soft_score == pytest.approx(unsafe.soft_score), (
            "form must not change nutrients, so the soft score must be identical"
        )

    def test_score_stays_in_the_unit_interval(self, engine, db):
        for record in db.records[:150]:
            evaluation = engine.evaluate(Meal(items=[record.as_item(150.0)]), _older_adult())
            assert 0.0 <= evaluation.score <= 1.0
            assert 0.0 <= evaluation.soft_score <= 1.0

    def test_empty_meal_is_scored_without_crashing(self, engine):
        evaluation = engine.evaluate(Meal(), _older_adult())
        assert 0.0 <= evaluation.score <= 1.0

    def test_unknown_food_is_ignored_rather_than_crashing(self, engine, db):
        from foodsense.schemas import MealItem

        ghost = MealItem(food_id="000000", name="ghost", quantity_g=100.0)
        assert 0.0 <= engine.evaluate(Meal(items=[ghost]), _older_adult()).score <= 1.0

    def test_flags_tighten_rather_than_duplicate_the_age_rule(self, engine, db):
        """The hypertension ceiling replaces the general sodium one; both cannot apply."""
        plain = engine.rules_for(_older_adult())
        flagged = engine.rules_for(_older_adult(health_flags=[HealthFlag.HYPERTENSION]))
        assert sum(r.quantity == "sodium_mg" for r in plain) == 1
        assert sum(r.quantity == "sodium_mg" for r in flagged) == 1
        plain_max = next(r for r in plain if r.quantity == "sodium_mg").threshold.maximum
        flagged_max = next(r for r in flagged if r.quantity == "sodium_mg").threshold.maximum
        assert flagged_max < plain_max

    def test_is_safe_agrees_with_evaluate(self, engine, db):
        grapes = db.by_hazard("grape")[0]
        profile = _toddler()
        for form in (Form.WHOLE, Form.QUARTERED):
            meal = Meal(items=[grapes.as_item(40, form), _staple(db)])
            assert engine.is_safe(meal, profile) == engine.evaluate(meal, profile).is_safe

    def test_is_valid_requires_both_safety_and_the_target(self, engine, db):
        grapes = db.by_hazard("grape")[0]
        meal = Meal(items=[grapes.as_item(40, Form.QUARTERED), _staple(db)])
        assert not engine.is_valid(meal, _toddler(), target_score=0.99)

    def test_the_same_meal_scores_differently_for_different_profiles(self, engine, db):
        """Personalisation has to be visible in the score or it is not personalisation."""
        meal = Meal(
            items=[
                db.search("chicken breast meat only roasted", limit=1)[0][0].as_item(120.0),
                db.search("rice brown long grain cooked", limit=1)[0][0].as_item(150.0),
            ]
        )
        scores = {
            goal: engine.evaluate(
                meal, UserProfile(age_group=AgeGroup.ADULT, weight_kg=70.0, goal=goal)
            ).soft_score
            for goal in Goal
        }
        assert len({round(s, 4) for s in scores.values()}) > 1
