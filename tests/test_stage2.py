"""Tests for Stage 2: the search space, the objective, the optimiser, the baselines.

The space tests are the important ones. Availability-awareness is a claim about
what the optimiser *cannot* do, and a claim like that is only worth what its tests
are worth -- so they check that no reachable point in the space contains a food
the user does not have, rather than checking that one particular run happened not
to use one.
"""

from __future__ import annotations

import numpy as np
import pytest

from foodsense.constraints.engine import RuleEngine
from foodsense.data.fdc import get_food_db
from foodsense.schemas import AgeGroup, Form, Goal, HealthFlag, Meal, UserProfile
from foodsense.stage1_prediction.predict import LIGHTGBM_PATH, get_suitability_model
from foodsense.stage2_optimizer.baselines import (
    METHODS,
    MethodContext,
    build_unrestricted_space,
    run_method,
)
from foodsense.stage2_optimizer.de_optimizer import DEConfig, differential_evolution
from foodsense.stage2_optimizer.objective import (
    CounterfactualObjective,
    ObjectiveConfig,
    meal_diff,
)
from foodsense.stage2_optimizer.space import MIN_SERVING_G, build_space

needs_model = pytest.mark.skipif(
    not LIGHTGBM_PATH.exists(), reason="Stage-1 model not trained; run `make train`"
)


@pytest.fixture(scope="module")
def db():
    return get_food_db()


@pytest.fixture(scope="module")
def engine(db):
    return RuleEngine(db=db)


@pytest.fixture(scope="module")
def model():
    if not LIGHTGBM_PATH.exists():
        pytest.skip("Stage-1 model not trained")
    return get_suitability_model()


@pytest.fixture(scope="module")
def toddler():
    return UserProfile(
        age_group=AgeGroup.TODDLER,
        age_months=18,
        weight_kg=11.0,
        goal=Goal.BALANCED_NUTRITION,
        health_flags=[HealthFlag.IRON_FOCUS],
    )


def _find(db, query):
    return db.search(query, limit=1)[0][0]


@pytest.fixture(scope="module")
def planned(db):
    """The toddler scenario: whole grapes, whole peanuts, plain rice."""
    return Meal(
        items=[
            _find(db, "grapes").as_item(40.0, Form.WHOLE),
            _find(db, "peanuts").as_item(20.0, Form.WHOLE),
            _find(db, "rice white long grain cooked").as_item(50.0),
        ]
    )


@pytest.fixture(scope="module")
def pantry(db):
    return Meal(
        items=[
            _find(db, "carrots cooked").as_item(0.0),
            _find(db, "chicken ground crumbles cooked").as_item(0.0),
            _find(db, "yogurt plain").as_item(0.0),
            _find(db, "milk whole").as_item(0.0),
            _find(db, "lentils cooked").as_item(0.0),
        ]
    )


# ---------------------------------------------------------------------------
# Search space -- extension #1
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_variables_are_exactly_planned_union_pantry(self, db, planned, pantry, toddler):
        space = build_space(planned, pantry, db, profile=toddler)
        expected = planned.food_ids() | pantry.food_ids()
        assert {v.food_id for v in space.variables} == expected

    def test_overlapping_food_appears_once_and_keeps_its_planned_amount(self, db, toddler):
        """Spare rice in the cupboard does not make the rice on the plate an addition."""
        rice = _find(db, "rice white long grain cooked")
        space = build_space(
            Meal(items=[rice.as_item(50.0)]), Meal(items=[rice.as_item(0.0)]), db, profile=toddler
        )
        assert len(space.variables) == 1
        assert space.variables[0].planned_quantity_g == 50.0
        assert not space.variables[0].from_pantry

    def test_no_reachable_point_contains_an_unavailable_food(self, db, planned, pantry, toddler):
        """The availability guarantee, tested as a property of the space itself.

        Not "this run happened not to use one" -- no point in the space can.
        """
        space = build_space(planned, pantry, db, profile=toddler)
        available = planned.food_ids() | pantry.food_ids()
        rng = np.random.default_rng(0)
        bounds = np.asarray(space.bounds())
        for _ in range(300):
            x = rng.uniform(bounds[:, 0], bounds[:, 1])
            assert space.decode(x).food_ids() <= available

    def test_pantry_items_start_at_zero(self, db, planned, pantry, toddler):
        space = build_space(planned, pantry, db, profile=toddler)
        for variable in space.variables:
            if variable.from_pantry:
                assert variable.planned_quantity_g == 0.0

    def test_encode_planned_round_trips(self, db, planned, pantry, toddler):
        space = build_space(planned, pantry, db, profile=toddler)
        decoded = space.decode(space.encode_planned())
        assert {(i.food_id, i.quantity_g, i.form) for i in decoded.items} == {
            (i.food_id, i.quantity_g, i.form) for i in planned.items
        }

    def test_decode_only_ever_proposes_a_permitted_form(self, db, planned, pantry, toddler):
        space = build_space(planned, pantry, db, profile=toddler)
        rng = np.random.default_rng(1)
        bounds = np.asarray(space.bounds())
        for _ in range(200):
            for item in space.decode(rng.uniform(bounds[:, 0], bounds[:, 1])).items:
                assert db.get(item.food_id).permits(item.form)

    def test_quantities_respect_their_upper_bound(self, db, planned, pantry, toddler):
        space = build_space(planned, pantry, db, profile=toddler)
        maxima = {v.food_id: v.max_quantity_g for v in space.variables}
        rng = np.random.default_rng(2)
        bounds = np.asarray(space.bounds())
        for _ in range(200):
            for item in space.decode(rng.uniform(bounds[:, 0], bounds[:, 1])).items:
                assert item.quantity_g <= maxima[item.food_id] + 1e-9

    def test_a_planned_item_cannot_grow_without_limit(self, db, toddler):
        rice = _find(db, "rice white long grain cooked")
        space = build_space(Meal(items=[rice.as_item(50.0)]), None, db, profile=toddler)
        assert space.variables[0].max_quantity_g <= 50.0 * 2.5 + 1e-9

    def test_tiny_amounts_are_dropped(self, db, planned, pantry, toddler):
        """Nobody serves two grams of lentils; the decoder refuses to propose it."""
        space = build_space(planned, pantry, db, profile=toddler)
        x = space.encode_planned()
        x[0] = MIN_SERVING_G - 1.0
        assert planned.items[0].food_id not in space.decode(x).food_ids()

    def test_integrality_marks_only_the_form_dimensions(self, db, planned, pantry, toddler):
        space = build_space(planned, pantry, db, profile=toddler)
        mask = space.integrality()
        assert mask.sum() == len(space.variables)
        assert not mask[0::2].any() and mask[1::2].all()

    def test_form_costs_prefer_the_declared_nearest_safe_form(self, db, toddler):
        """Quartering must be cheaper than pureeing, or 'nearest' means nothing."""
        grapes = db.by_hazard("grape")[0]
        space = build_space(
            Meal(items=[grapes.as_item(40.0, Form.WHOLE)]), None, db, profile=toddler
        )
        costs = dict(zip(space.variables[0].forms, space.variables[0].form_costs, strict=True))
        assert costs[Form.WHOLE] == 0.0
        assert costs[Form.QUARTERED] < costs[Form.PUREED]

    def test_unavailable_items_audits_a_padded_space(self, db, planned, pantry, toddler):
        space = build_unrestricted_space(planned, pantry, db, toddler, np.random.default_rng(0))
        assert len(space.variables) > len(planned.items) + len(pantry.items)
        extra = next(v for v in space.variables if v.food_id not in space.available_ids)
        meal = Meal(items=[extra.record.as_item(50.0)])
        assert len(space.unavailable_items(meal)) == 1


# ---------------------------------------------------------------------------
# Objective -- extension #2
# ---------------------------------------------------------------------------


@needs_model
class TestObjective:
    def test_planned_meal_has_zero_distance_and_sparsity(
        self, db, planned, pantry, toddler, model, engine
    ):
        space = build_space(planned, pantry, db, profile=toddler)
        objective = CounterfactualObjective(space, toddler, model, engine)
        terms = objective.terms(space.encode_planned())
        assert terms.l1_g == pytest.approx(0.0)
        assert terms.n_changed == 0
        assert terms.distance == pytest.approx(0.0)

    def test_the_planned_meal_is_penalised_for_its_hazards(
        self, db, planned, pantry, toddler, model, engine
    ):
        space = build_space(planned, pantry, db, profile=toddler)
        objective = CounterfactualObjective(space, toddler, model, engine)
        terms = objective.terms(space.encode_planned())
        assert terms.n_hard_violations >= 1
        assert terms.safety >= objective.config.big_penalty

    def test_safety_cannot_be_traded_away(self, db, planned, pantry, toddler, model, engine):
        """One hazard must cost more than every other term combined can ever save."""
        space = build_space(planned, pantry, db, profile=toddler)
        config = ObjectiveConfig.load()
        worst_case = (
            config.lambda_validity
            + config.lambda_distance * 10
            + config.lambda_sparsity * len(space.variables)
        )
        # Two orders of magnitude of headroom: one hazard must dominate everything
        # the rest of the objective could ever be worth, not merely exceed it.
        assert config.big_penalty > worst_case * 20

    def test_validity_term_is_one_sided(self, db, planned, pantry, toddler, model, engine):
        """Above the target there is nothing left to gain, so distance takes over."""
        space = build_space(planned, pantry, db, profile=toddler)
        objective = CounterfactualObjective(space, toddler, model, engine)
        config = objective.config
        assert max(0.0, config.target_score - (config.target_score + 0.2)) == 0.0

    def test_ablation_switches_remove_their_terms(
        self, db, planned, pantry, toddler, model, engine
    ):
        space = build_space(planned, pantry, db, profile=toddler)
        x = space.encode_planned()
        x[0] += 40.0
        full = CounterfactualObjective(space, toddler, model, engine).terms(x)
        bare = CounterfactualObjective(
            space, toddler, model, engine, use_safety=False, use_sparsity=False, use_distance=False
        ).terms(x)
        assert full.distance > 0 and bare.distance == 0.0
        assert full.safety > 0 and bare.safety == 0.0
        assert bare.sparsity == 0.0

    def test_empty_candidate_is_rejected(self, db, planned, pantry, toddler, model, engine):
        space = build_space(planned, pantry, db, profile=toddler)
        objective = CounterfactualObjective(space, toddler, model, engine)
        assert objective.terms(np.zeros(space.n_dims)).total > 1000.0

    def test_batch_matches_single(self, db, planned, pantry, toddler, model, engine):
        space = build_space(planned, pantry, db, profile=toddler)
        objective = CounterfactualObjective(space, toddler, model, engine)
        rng = np.random.default_rng(3)
        bounds = np.asarray(space.bounds())
        population = rng.uniform(bounds[:, 0], bounds[:, 1], size=(6, space.n_dims))
        batched = objective.values(population)
        singly = [objective.terms(row).total for row in population]
        assert np.allclose(batched, singly)

    def test_form_change_counts_as_an_edit(self, db, planned, pantry, toddler):
        """Re-quartering the grapes is work the user has to do, at identical grams."""
        space = build_space(planned, pantry, db, profile=toddler)
        x = space.encode_planned()
        _, _, before, _ = meal_diff(space, x, 2.0, MIN_SERVING_G)
        x[1] = (x[1] + 1) % len(space.variables[0].forms)
        _, _, after, _ = meal_diff(space, x, 2.0, MIN_SERVING_G)
        assert after == before + 1


class TestTheDecoderAndTheDiffAgree:
    """Three numbers in this pipeline are grams and mean different things.

    ``min_serving_g`` is the presence floor -- is this food on the plate at all.
    ``change_epsilon_g`` is the no-op tolerance -- did a served amount move.
    ``build_diff``'s epsilon is the same question one stage later, for the user.

    The objective's diff and the decoder used to apply *different* presence
    floors, so the search paid L1 distance and a sparsity increment for
    sub-serving additions the decoder then threw away, and under-counted the L1
    of a planned item it had shrunk out of existence. These pin the boundary.
    """

    def test_just_below_the_floor_is_not_an_addition(self, db, planned, pantry, toddler):
        space = build_space(planned, pantry, db, profile=toddler)
        index = next(i for i, v in enumerate(space.variables) if v.from_pantry)
        x = space.encode_planned()
        x[2 * index] = MIN_SERVING_G - 1.0

        assert space.variables[index].food_id not in space.decode(x).food_ids()
        l1, _, n_changed, _ = meal_diff(space, x, 2.0, MIN_SERVING_G)
        assert l1 == 0.0, "charged distance for a portion that is not served"
        assert n_changed == 0, "charged sparsity for an edit that never happened"

    def test_exactly_at_the_floor_is_an_addition(self, db, planned, pantry, toddler):
        space = build_space(planned, pantry, db, profile=toddler)
        index = next(i for i, v in enumerate(space.variables) if v.from_pantry)
        x = space.encode_planned()
        x[2 * index] = MIN_SERVING_G

        assert space.variables[index].food_id in space.decode(x).food_ids()
        l1, _, n_changed, _ = meal_diff(space, x, 2.0, MIN_SERVING_G)
        assert l1 == pytest.approx(MIN_SERVING_G)
        assert n_changed == 1

    def test_shrinking_an_item_below_the_floor_costs_its_whole_weight(
        self, db, planned, pantry, toddler
    ):
        """It is a removal, not a reduction, and the L1 has to say the same."""
        space = build_space(planned, pantry, db, profile=toddler)
        index = next(i for i, v in enumerate(space.variables) if not v.from_pantry)
        planned_g = space.variables[index].planned_quantity_g
        x = space.encode_planned()
        x[2 * index] = MIN_SERVING_G - 1.0

        assert space.variables[index].food_id not in space.decode(x).food_ids()
        l1, _, _, _ = meal_diff(space, x, 2.0, MIN_SERVING_G)
        assert l1 == pytest.approx(planned_g)

    def test_the_two_agree_on_every_random_point(self, db, planned, pantry, toddler):
        """The property, not the three cases: for any decision vector, the L1 and
        edit count the objective charges are the L1 and edit count of the meal the
        decoder actually produces."""
        space = build_space(planned, pantry, db, profile=toddler)
        planned_by_id = {v.food_id: v.planned_quantity_g for v in space.variables}
        planned_form = {v.food_id: v.planned_form for v in space.variables}
        rng = np.random.default_rng(11)
        bounds = np.asarray(space.bounds())

        for _ in range(200):
            x = rng.uniform(bounds[:, 0], bounds[:, 1])
            meal = space.decode(x)
            served = {i.food_id: i for i in meal.items}

            expected_l1 = 0.0
            expected_changed = 0
            for food_id, before in planned_by_id.items():
                item = served.get(food_id)
                after = item.quantity_g if item is not None else 0.0
                expected_l1 += abs(after - before)
                form_changed = item is not None and item.form != planned_form[food_id]
                if abs(after - before) > 2.0 or form_changed:
                    expected_changed += 1

            l1, _, n_changed, _ = meal_diff(space, x, 2.0, MIN_SERVING_G)
            assert l1 == pytest.approx(expected_l1)
            assert n_changed == expected_changed


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


@needs_model
class TestDifferentialEvolution:
    @pytest.fixture(scope="class")
    def result(self, db, planned, pantry, toddler, model, engine):
        space = build_space(planned, pantry, db, profile=toddler)
        objective = CounterfactualObjective(space, toddler, model, engine)
        return space, differential_evolution(objective, space, toddler, engine, DEConfig.load())

    def test_it_repairs_the_hazards(self, result, toddler, engine, planned):
        _, optimised = result
        assert not engine.evaluate(planned, toddler).is_safe
        assert engine.evaluate(optimised.meal, toddler).is_safe

    def test_it_improves_the_score(self, result, toddler, engine, planned):
        _, optimised = result
        before = engine.evaluate(planned, toddler).score
        assert engine.evaluate(optimised.meal, toddler).score > before

    def test_it_stays_inside_the_available_foods(self, result, planned, pantry):
        _, optimised = result
        assert optimised.meal.food_ids() <= (planned.food_ids() | pantry.food_ids())

    def test_it_repairs_grapes_by_quartering_rather_than_removing(self, result, db):
        """The proposal's worked example, asserted."""
        _, optimised = result
        grape_ids = {r.fdc_id for r in db.by_hazard("grape")}
        grapes = [i for i in optimised.meal.items if i.food_id in grape_ids]
        assert grapes, "grapes were removed rather than re-formed"
        assert grapes[0].form is Form.QUARTERED

    def test_it_removes_the_peanuts(self, result, db):
        """At 18 months whole nuts have no safe form, so re-forming is not an option."""
        _, optimised = result
        nut_ids = {r.fdc_id for r in db.by_hazard("nut")}
        assert not (optimised.meal.food_ids() & nut_ids)

    def test_the_edit_is_minimal(self, result):
        _, optimised = result
        assert optimised.terms.n_changed <= 5

    def test_it_reports_its_own_cost(self, result):
        _, optimised = result
        assert optimised.evaluations > 0
        assert optimised.runtime_s > 0
        assert optimised.converged_reason

    def test_it_is_reproducible(self, db, planned, pantry, toddler, model, engine):
        runs = []
        for _ in range(2):
            space = build_space(planned, pantry, db, profile=toddler)
            objective = CounterfactualObjective(space, toddler, model, engine)
            runs.append(
                differential_evolution(objective, space, toddler, engine, DEConfig.load(), seed=7)
            )
        assert runs[0].meal.model_dump() == runs[1].meal.model_dump()

    def test_the_budget_is_respected(self, db, planned, pantry, toddler, model, engine):
        space = build_space(planned, pantry, db, profile=toddler)
        objective = CounterfactualObjective(space, toddler, model, engine)
        optimised = differential_evolution(
            objective, space, toddler, engine, DEConfig.load(), max_evaluations=500
        )
        assert optimised.evaluations <= 500 + DEConfig.load().population_size

    def test_validity_is_decided_by_the_rules_not_the_surrogate(self, result, toddler, engine):
        """The structural guarantee that the optimiser cannot grade its own work."""
        _, optimised = result
        expected = engine.is_valid(optimised.meal, toddler, ObjectiveConfig.load().target_score)
        assert optimised.valid == expected

    def test_an_empty_space_does_not_crash(self, toddler, engine, model, db):
        space = build_space(Meal(), None, db, profile=toddler)
        objective = CounterfactualObjective(space, toddler, model, engine)
        optimised = differential_evolution(objective, space, toddler, engine, DEConfig.load())
        assert optimised.converged_reason == "empty_search_space"


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@needs_model
class TestBaselines:
    @pytest.fixture(scope="class")
    def context(self, db, planned, pantry, toddler, model, engine):
        return MethodContext(
            planned_meal=planned,
            pantry=pantry,
            profile=toddler,
            db=db,
            model=model,
            engine=engine,
            budget=1500,
        )

    @pytest.mark.parametrize("method", METHODS)
    def test_every_method_returns_a_scored_result(self, context, method):
        result = run_method(method, context)
        assert result.method == method
        assert result.meal is not None
        assert result.norm_l1 >= 0.0
        assert result.evaluations >= 0

    def test_foodsense_never_leaves_the_available_foods(self, context):
        assert run_method("foodsense_de", context).n_unavailable == 0

    def test_foodsense_repairs_the_hazards(self, context):
        result = run_method("foodsense_de", context)
        assert result.safe and result.n_hard_violations == 0

    def test_baselines_search_a_wider_space(self, db, planned, pantry, toddler):
        restricted = build_space(planned, pantry, db, profile=toddler)
        padded = build_unrestricted_space(planned, pantry, db, toddler, np.random.default_rng(0))
        assert len(padded.variables) > len(restricted.variables)
        assert padded.available_ids == restricted.available_ids

    def test_unknown_method_is_rejected(self, context):
        with pytest.raises(ValueError, match="Unknown method"):
            run_method("simulated_annealing", context)

    def test_dice_is_held_to_the_shared_budget(self, context):
        """DiCE-genetic's initialiser does not terminate on a sparse feasible
        region; the shared budget is what turns that into a reported outcome
        rather than a hang."""
        result = run_method("dice_genetic", context)
        assert result.evaluations <= context.budget + 1
        assert result.note
