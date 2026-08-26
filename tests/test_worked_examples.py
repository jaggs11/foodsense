"""THE acceptance tests: the proposal's worked examples, asserted end to end.

Every other test file checks a component. This one checks the claims the project
was approved on, through the whole four-stage pipeline, on the exact scenarios
the proposal describes. If these pass, the system does what it said it would.

The assertions come from the design brief's acceptance criteria (section 7):

  Scenario 1  the final meal contains grapes with ``form == quartered``;
              contains no item violating a toddler choking rule; every item's
              nutrients are within +/-10% of the database recomputation; and
              suitability(final) > suitability(planned).
  Scenario 2  final sodium <= 500 mg; at most four items changed (minimality);
              verification passes.
"""

from __future__ import annotations

import json

import pytest

from foodsense.constraints.age_rules import check_choking, check_texture
from foodsense.constraints.engine import RuleEngine
from foodsense.constraints.goals import meal_metrics
from foodsense.data.fdc import get_food_db
from foodsense.pipeline import run_scenario
from foodsense.scenarios import SCENARIOS, load_scenario
from foodsense.schemas import Form, PipelineTrace
from foodsense.stage1_prediction.predict import LIGHTGBM_PATH

pytestmark = pytest.mark.skipif(
    not LIGHTGBM_PATH.exists(), reason="Stage-1 model not trained; run `make train`"
)


@pytest.fixture(scope="module")
def db():
    return get_food_db()


@pytest.fixture(scope="module")
def engine(db):
    return RuleEngine(db=db)


@pytest.fixture(scope="module")
def traces(db) -> dict[str, PipelineTrace]:
    """Every scenario, run once. The pipeline is deterministic, so this is safe."""
    return {key: run_scenario(key, db=db) for key in SCENARIOS}


# ---------------------------------------------------------------------------
# Properties every scenario must satisfy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(SCENARIOS))
class TestEveryScenario:
    def test_the_pipeline_completes_all_four_stages(self, traces, key):
        trace = traces[key]
        assert trace.stage1 is not None
        assert trace.stage2 is not None
        assert trace.stage3 is not None
        assert trace.stage4 is not None

    def test_verification_passes(self, traces, key):
        assert traces[key].stage4.final_pass, traces[key].stage4.flagged

    def test_the_final_meal_is_safe(self, traces, key, engine):
        trace = traces[key]
        assert engine.evaluate(trace.final_meal, trace.profile).is_safe

    def test_it_never_recommends_a_food_the_user_does_not_have(self, traces, key, db):
        """Extension #1, end to end: availability holds through all four stages."""
        trace = traces[key]
        scenario = load_scenario(key)
        available = scenario.planned_meal(db).food_ids() | scenario.pantry_meal(db).food_ids()
        assert trace.final_meal.food_ids() <= available

    def test_every_final_item_exists_in_the_database(self, traces, key, db):
        assert db.unknown_ids(traces[key].final_meal) == []

    def test_every_form_is_one_the_food_can_take(self, traces, key, db):
        for item in traces[key].final_meal.items:
            assert db.get(item.food_id).permits(item.form)

    def test_nutrients_match_the_database_recomputation(self, traces, key, db):
        """The Stage-4 guarantee: what is reported is what the database says.

        The design brief asks for +/-10%; after verification this should be exact,
        because the reported figures *are* the recomputation.
        """
        trace = traces[key]
        reported = trace.stage4.verified_nutrients.as_dict()
        recomputed = db.nutrients_for(trace.final_meal).as_dict()
        for nutrient, value in recomputed.items():
            if value <= 0:
                continue
            assert abs(reported[nutrient] - value) / value <= 0.10, nutrient

    def test_the_meal_improves_or_was_already_fine(self, traces, key, engine):
        trace = traces[key]
        before = engine.evaluate(trace.planned_meal, trace.profile).score
        after = engine.evaluate(trace.final_meal, trace.profile).score
        assert after >= before - 1e-9

    def test_the_recommendation_says_something(self, traces, key):
        trace = traces[key]
        assert trace.stage3.text.strip()
        assert trace.stage3.rationale

    def test_the_offline_path_needs_no_provider(self, traces, key):
        assert traces[key].stage3.provider == "template"
        assert not traces[key].stage3.fallback_used

    def test_the_trace_serialises_with_what_the_ui_needs(self, traces, key):
        """The API returns this verbatim, so it has to serialise -- computed fields included.

        The trace is an output document, not an input one: its computed fields
        (`is_safe`, `n_corrections`, ...) are exactly what the UI renders badges
        from, and models that accept them back would also accept typos in a
        request body. So this asserts the serialisation the frontend depends on,
        and the round-trip is asserted on the request models below.
        """
        payload = json.loads(traces[key].model_dump_json())
        assert payload["final_rule_evaluation"]["is_safe"] is not None
        assert payload["stage4"]["n_corrections"] >= 0
        assert payload["succeeded"] in (True, False)
        assert payload["final_meal"]["total_quantity_g"] > 0

    def test_the_request_models_round_trip(self, traces, key):
        """What the API *accepts* must survive a round trip exactly."""
        trace = traces[key]
        from foodsense.schemas import Meal as MealModel
        from foodsense.schemas import UserProfile as ProfileModel

        assert (
            ProfileModel.model_validate_json(trace.profile.model_dump_json(exclude={"age_years"}))
            == trace.profile
        )
        restored = MealModel.model_validate_json(
            trace.planned_meal.model_dump_json(exclude={"total_quantity_g"})
        )
        assert restored.items == trace.planned_meal.items


# ---------------------------------------------------------------------------
# Scenario 1 -- the toddler choking case
# ---------------------------------------------------------------------------


class TestToddlerChoking:
    @pytest.fixture(scope="class")
    def trace(self, traces):
        return traces["toddler_choking"]

    def test_the_planned_meal_really_is_unsafe(self, trace, engine, db):
        """Guard: if the starting meal were safe the scenario would prove nothing."""
        evaluation = engine.evaluate(trace.planned_meal, trace.profile)
        assert not evaluation.is_safe
        rules = {v.rule_id for v in evaluation.hard_violations}
        assert "toddler.choking.grape" in rules
        assert "toddler.choking.nut" in rules

    def test_grapes_survive_and_are_quartered(self, trace, db):
        """The headline claim: a hazard repaired by re-forming, not by deleting."""
        grape_ids = {r.fdc_id for r in db.by_hazard("grape")}
        grapes = [i for i in trace.final_meal.items if i.food_id in grape_ids]
        assert grapes, "the grapes were removed rather than re-formed"
        assert grapes[0].form is Form.QUARTERED

    def test_the_whole_peanuts_are_gone(self, trace, db):
        """At 18 months whole nuts have no safe form, so substitution is the only fix."""
        nut_ids = {r.fdc_id for r in db.by_hazard("nut")}
        assert not (trace.final_meal.food_ids() & nut_ids)

    def test_no_choking_rule_is_violated(self, trace, db):
        assert check_choking(trace.final_meal, trace.profile, db) == []

    def test_suitability_improves(self, trace, engine):
        before = engine.evaluate(trace.planned_meal, trace.profile)
        after = engine.evaluate(trace.final_meal, trace.profile)
        assert after.score > before.score
        assert after.soft_score > before.soft_score

    def test_the_substitution_came_from_the_pantry(self, trace, db):
        scenario = load_scenario("toddler_choking")
        planned = scenario.planned_meal(db).food_ids()
        pantry = scenario.pantry_meal(db).food_ids()
        added = trace.final_meal.food_ids() - planned
        assert added, "nothing was substituted in"
        assert added <= pantry

    def test_the_explanation_names_the_hazard(self, trace):
        text = (trace.stage3.text + " " + " ".join(trace.stage3.rationale)).lower()
        assert "grape" in text
        assert "quarter" in text or "choking" in text


# ---------------------------------------------------------------------------
# Scenario 2 -- the older adult sodium case
# ---------------------------------------------------------------------------


class TestElderlySodium:
    @pytest.fixture(scope="class")
    def trace(self, traces):
        return traces["elderly_sodium"]

    def test_the_planned_meal_really_is_over_the_ceiling(self, trace, db):
        sodium = meal_metrics(trace.planned_meal, db)["sodium_mg"]
        assert sodium > 500, f"planned sodium {sodium:.0f} mg is not a breach"

    def test_final_sodium_is_within_the_per_meal_ceiling(self, trace, db):
        """The acceptance criterion from the brief: <= 500 mg."""
        sodium = meal_metrics(trace.final_meal, db)["sodium_mg"]
        assert sodium <= 500.0, f"final sodium {sodium:.0f} mg exceeds the 500 mg ceiling"

    def test_the_edit_is_minimal(self, trace):
        """At most four items changed, per the brief."""
        assert trace.stage2.diff.n_items_changed <= 4

    def test_verification_passes(self, trace):
        assert trace.stage4.final_pass

    def test_the_sodium_rule_no_longer_fires(self, trace, engine):
        evaluation = engine.evaluate(trace.final_meal, trace.profile)
        assert not any(v.rule_id == "flag.hypertension.sodium_mg" for v in evaluation.violations)


# ---------------------------------------------------------------------------
# Scenario 3 -- the adult weight-management case
# ---------------------------------------------------------------------------


class TestAdultWeight:
    @pytest.fixture(scope="class")
    def trace(self, traces):
        return traces["adult_weight"]

    def test_no_safety_rule_was_involved(self, trace, engine):
        """The point of this scenario: the goal layer works without any hazard."""
        assert engine.evaluate(trace.planned_meal, trace.profile).is_safe

    def test_the_goal_thresholds_are_met(self, trace, db):
        metrics = meal_metrics(trace.final_meal, db)
        assert metrics["energy_kcal"] <= 550.0, "weight-management kcal ceiling"
        assert metrics["protein_g"] >= 25.0 * 0.9, "protein floor (within rounding)"

    def test_the_meal_was_actually_edited(self, trace):
        assert trace.stage2.diff.n_items_changed > 0

    def test_the_optimiser_believes_it_reached_the_target(self, trace, engine):
        """It does, by the surrogate. The rule engine, which is the judge, disagrees.

        The optimiser minimises a validity term built from the *surrogate's*
        estimate, and that term reaches zero here. Validity is then decided by the
        rule engine, deliberately -- an optimiser graded by its own model can win by
        finding the model's blind spots. This scenario sits in the gap between the
        two, and the gap is the honest reason it falls short: the surrogate scores
        the final meal just above 0.70 while the rules score it just below.

        Asserted rather than tuned away. Narrowing it means a better Stage-1 model,
        not a different weight in `pipeline.yaml`.
        """
        surrogate = trace.stage2.suitability_after
        actual = engine.evaluate(trace.final_meal, trace.profile).score
        assert surrogate >= 0.70, "the optimiser did not think it had finished"
        assert actual < surrogate, "no surrogate gap left to explain the shortfall"
        assert surrogate - actual < 0.10, f"surrogate is off by {surrogate - actual:.3f}"

    def test_the_shortfall_is_micronutrient_floors_a_single_meal_cannot_meet(self, trace, engine):
        """What the composite score is losing on, named rather than hand-waved.

        The weight-management goal this scenario is *about* is met (see above). The
        composite also carries per-meal micronutrient floors -- vitamin D in
        particular, which almost no unfortified single meal supplies -- and those
        are what hold the score under the target.
        """
        per_rule = engine.evaluate(trace.final_meal, trace.profile).per_rule
        worst = min(per_rule, key=per_rule.get)
        assert worst.startswith("age."), f"the weakest rule is {worst}, not a micronutrient floor"
        goal_rules = {k: v for k, v in per_rule.items() if k.startswith("goal.")}
        assert min(goal_rules.values()) > per_rule[worst]


# ---------------------------------------------------------------------------
# Cross-cutting guarantees
# ---------------------------------------------------------------------------


class TestPipelineGuarantees:
    def test_the_pipeline_is_deterministic(self, db):
        """A demo that gives a different answer each run is not a demo."""
        first = run_scenario("toddler_choking", db=db)
        second = run_scenario("toddler_choking", db=db)
        assert first.final_meal.model_dump() == second.final_meal.model_dump()
        assert first.stage3.text == second.stage3.text

    def test_an_empty_pantry_still_produces_a_safe_answer(self, db, engine):
        """With nothing to substitute in, the only repairs left are form and removal."""
        from foodsense.pipeline import run_pipeline

        scenario = load_scenario("toddler_choking")
        trace = run_pipeline(
            scenario.profile, scenario.planned_meal(db), None, db=db, engine=engine
        )
        assert engine.evaluate(trace.final_meal, trace.profile).is_safe
        assert check_choking(trace.final_meal, trace.profile, db) == []

    def test_dysphagia_restricts_the_final_textures(self, db, engine):
        """A texture-restricted profile must not be handed something unswallowable."""
        from foodsense.pipeline import run_pipeline
        from foodsense.schemas import HealthFlag

        scenario = load_scenario("elderly_sodium")
        profile = scenario.profile.model_copy(
            update={"health_flags": [*scenario.profile.health_flags, HealthFlag.DYSPHAGIA]}
        )
        trace = run_pipeline(
            profile, scenario.planned_meal(db), scenario.pantry_meal(db), db=db, engine=engine
        )
        assert check_texture(trace.final_meal, profile, db) == []

    def test_every_scenario_loads_its_pinned_foods(self, db):
        for scenario in SCENARIOS.values():
            assert scenario.planned_meal(db).items
            assert db.unknown_ids(scenario.planned_meal(db)) == []
            assert db.unknown_ids(scenario.pantry_meal(db)) == []

    def test_unknown_scenario_is_rejected(self):
        with pytest.raises(KeyError, match="Unknown scenario"):
            load_scenario("brunch")
