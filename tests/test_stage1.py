"""Tests for Stage 1: features, weak-supervision labels, and the trained surrogate.

The feature-contract tests matter more than they look. Stage 2 calls the model
thousands of times per recommendation; if a column silently moves, the optimiser
climbs a scrambled surface and still returns a confident-looking answer.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from foodsense.constraints.engine import RuleEngine
from foodsense.constraints.goals import estimate_glycemic_load
from foodsense.data.fdc import get_food_db
from foodsense.schemas import AgeGroup, Goal, HealthFlag, Meal, UserProfile
from foodsense.stage1_prediction.features import (
    FEATURE_NUTRIENTS,
    feature_names,
    meal_features,
    n_features,
)
from foodsense.stage1_prediction.labels import (
    FLAG_PRIORS,
    build_dataset,
    perturb_meal,
    sample_profile,
)
from foodsense.stage1_prediction.predict import (
    LIGHTGBM_PATH,
    ModelMissingError,
    SuitabilityModel,
)
from foodsense.stage1_prediction.train import evaluate_predictions

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def db():
    return get_food_db()


@pytest.fixture(scope="module")
def engine(db):
    return RuleEngine(db=db)


@pytest.fixture(scope="module")
def meal(db):
    return Meal(
        items=[
            db.search("chicken breast meat only roasted", limit=1)[0][0].as_item(120.0),
            db.search("rice brown long grain cooked", limit=1)[0][0].as_item(150.0),
            db.search("broccoli cooked boiled drained", limit=1)[0][0].as_item(100.0),
        ]
    )


@pytest.fixture
def profile():
    return UserProfile(
        age_group=AgeGroup.ADULT,
        age_months=35 * 12,
        weight_kg=70.0,
        goal=Goal.BALANCED_NUTRITION,
    )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_row_length_matches_the_declared_contract(self, meal, profile, db):
        assert len(meal_features(meal, profile, db)) == n_features() == len(feature_names())

    def test_feature_names_are_unique(self):
        names = feature_names()
        assert len(names) == len(set(names))

    def test_nutrient_columns_match_the_database_totals(self, meal, profile, db):
        row = meal_features(meal, profile, db)
        names = feature_names()
        totals = db.nutrients_for(meal).as_dict()
        for nutrient in FEATURE_NUTRIENTS:
            assert row[names.index(nutrient)] == pytest.approx(totals[nutrient])

    def test_glycemic_load_column_matches_the_estimator(self, meal, profile, db):
        row = meal_features(meal, profile, db)
        assert row[feature_names().index("glycemic_load")] == pytest.approx(
            estimate_glycemic_load(meal, db)
        )

    def test_structural_columns(self, meal, profile, db):
        row = meal_features(meal, profile, db)
        names = feature_names()
        assert row[names.index("item_count")] == 3.0
        assert row[names.index("total_mass_g")] == pytest.approx(370.0)
        assert row[names.index("max_item_mass_share")] == pytest.approx(150.0 / 370.0)

    def test_energy_shares_are_fractions_that_roughly_sum_to_one(self, meal, profile, db):
        row = meal_features(meal, profile, db)
        names = feature_names()
        shares = [row[names.index(f"share_{m}")] for m in ("protein", "carbohydrate", "fat")]
        assert all(0.0 <= s <= 1.0 for s in shares)
        assert 0.85 <= sum(shares) <= 1.15

    def test_profile_is_one_hot_encoded(self, meal, db):
        names = feature_names()
        for age_group in AgeGroup:
            for goal in Goal:
                row = meal_features(
                    meal, UserProfile(age_group=age_group, goal=goal, weight_kg=70.0), db
                )
                age_bits = [row[names.index(f"age_{a.value}")] for a in AgeGroup]
                goal_bits = [row[names.index(f"goal_{g.value}")] for g in Goal]
                assert sum(age_bits) == 1.0 and row[names.index(f"age_{age_group.value}")] == 1.0
                assert sum(goal_bits) == 1.0 and row[names.index(f"goal_{goal.value}")] == 1.0

    def test_health_flags_are_bits(self, meal, db):
        names = feature_names()
        row = meal_features(
            meal,
            UserProfile(
                age_group=AgeGroup.OLDER_ADULT,
                weight_kg=70.0,
                health_flags=[HealthFlag.WARFARIN, HealthFlag.HYPERTENSION],
            ),
            db,
        )
        assert row[names.index("flag_warfarin")] == 1.0
        assert row[names.index("flag_hypertension")] == 1.0
        assert row[names.index("flag_statin")] == 0.0

    def test_form_does_not_change_features(self, db, profile):
        """Form is safety and texture, never nutrition -- the features must agree."""
        from foodsense.schemas import Form

        record = db.by_hazard("grape")[0]
        whole = Meal(items=[record.as_item(40.0, Form.WHOLE)])
        quartered = Meal(items=[record.as_item(40.0, Form.QUARTERED)])
        assert np.array_equal(
            meal_features(whole, profile, db), meal_features(quartered, profile, db)
        )

    def test_empty_meal_produces_a_finite_zero_row(self, profile, db):
        row = meal_features(Meal(), profile, db)
        assert np.isfinite(row).all()
        assert row[feature_names().index("item_count")] == 0.0

    def test_unknown_food_does_not_crash_or_produce_nan(self, profile, db):
        from foodsense.schemas import MealItem

        row = meal_features(
            [MealItem(food_id="000000", name="ghost", quantity_g=50.0)], profile, db
        )
        assert np.isfinite(row).all()

    def test_features_are_deterministic(self, meal, profile, db):
        assert np.array_equal(meal_features(meal, profile, db), meal_features(meal, profile, db))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestProfileSampling:
    def test_sampled_profiles_are_internally_consistent(self):
        rng = random.Random(0)
        for _ in range(300):
            sampled = sample_profile(rng)
            assert sampled.age_months is not None and sampled.age_months > 0
            assert sampled.weight_kg and sampled.weight_kg > 0
            assert set(sampled.health_flags) <= {
                flag for flag, _ in FLAG_PRIORS.get(sampled.age_group, ())
            }

    def test_toddlers_never_get_medication_flags(self):
        """Sampling a toddler on warfarin would spend model capacity on nonsense."""
        rng = random.Random(1)
        medication = {
            HealthFlag.WARFARIN,
            HealthFlag.MAOI,
            HealthFlag.STATIN,
            HealthFlag.METFORMIN,
            HealthFlag.ACE_INHIBITOR_OR_K_SPARING_DIURETIC,
        }
        for _ in range(400):
            sampled = sample_profile(rng, AgeGroup.TODDLER)
            assert not (set(sampled.health_flags) & medication)

    def test_sampling_is_reproducible(self):
        a = [sample_profile(random.Random(7)) for _ in range(5)]
        b = [sample_profile(random.Random(7)) for _ in range(5)]
        assert [p.model_dump() for p in a] == [p.model_dump() for p in b]

    def test_all_age_groups_and_goals_are_reachable(self):
        rng = random.Random(3)
        sampled = [sample_profile(rng) for _ in range(600)]
        assert {p.age_group for p in sampled} == set(AgeGroup)
        assert {p.goal for p in sampled} == set(Goal)


class TestPerturbation:
    def test_perturbed_meals_stay_valid(self, meal, db):
        rng = random.Random(0)
        for _ in range(200):
            perturbed = perturb_meal(meal, rng, db)
            assert all(i.quantity_g > 0 for i in perturbed.items)
            assert db.unknown_ids(perturbed.items) == []

    def test_perturbation_actually_moves_the_meal(self, meal, db):
        rng = random.Random(0)
        changed = sum(
            perturb_meal(meal, rng, db).model_dump() != meal.model_dump() for _ in range(50)
        )
        assert changed > 40

    def test_perturbation_widens_the_energy_distribution(self, meal, db):
        """The optimiser will visit portions no recipe contains; training must too."""
        rng = random.Random(0)
        energies = [db.nutrients_for(perturb_meal(meal, rng, db)).energy_kcal for _ in range(200)]
        base = db.nutrients_for(meal).energy_kcal
        assert min(energies) < base * 0.8
        assert max(energies) > base * 1.3


class TestDataset:
    @pytest.fixture(scope="class")
    def dataset(self, db, engine):
        meals = [
            Meal(items=[db.records[i].as_item(100.0), db.records[i + 1].as_item(80.0)])
            for i in range(0, 120, 2)
        ]
        return build_dataset(meals, engine, n_profiles_per_meal=3, seed=42)

    def test_shapes_line_up(self, dataset):
        assert dataset.X.shape == (len(dataset), n_features())
        assert dataset.y.shape == dataset.y_clean.shape == dataset.groups.shape

    def test_labels_are_in_the_unit_interval(self, dataset):
        assert ((dataset.y >= 0.0) & (dataset.y <= 1.0)).all()
        assert ((dataset.y_clean >= 0.0) & (dataset.y_clean <= 1.0)).all()

    def test_noise_matches_the_configured_sigma(self, dataset):
        """Clipping at 0 and 1 biases the residual slightly; the band allows for it."""
        residual = dataset.y - dataset.y_clean
        assert 0.02 < residual.std() < 0.08

    def test_clean_labels_are_the_rule_engine_soft_score(self, db, engine):
        """The label is the guideline score, not the safety-penalised one."""
        meals = [
            Meal(items=[db.records[i].as_item(120.0), db.records[i + 1].as_item(90.0)])
            for i in range(0, 40, 2)
        ]
        dataset = build_dataset(meals, engine, n_profiles_per_meal=2, seed=42, keep_examples=True)
        assert dataset.examples
        for label, (candidate, used_profile) in zip(dataset.y_clean, dataset.examples, strict=True):
            assert label == pytest.approx(engine.evaluate(candidate, used_profile).soft_score)

    def test_labels_ignore_hard_safety(self, db, engine):
        """A choking hazard must not move the label: no nutrient feature encodes it."""
        from foodsense.schemas import Form

        grapes = db.by_hazard("grape")[0]
        staple = db.search("rice white long grain cooked", limit=1)[0][0]
        whole = Meal(items=[grapes.as_item(40.0, Form.WHOLE), staple.as_item(60.0)])
        safe = Meal(items=[grapes.as_item(40.0, Form.QUARTERED), staple.as_item(60.0)])
        a = build_dataset([whole], engine, n_profiles_per_meal=2, perturbation_rate=0.0, seed=5)
        b = build_dataset([safe], engine, n_profiles_per_meal=2, perturbation_rate=0.0, seed=5)
        assert np.allclose(a.y_clean, b.y_clean)

    def test_groups_track_the_source_meal(self, dataset):
        assert dataset.n_groups == 60
        assert set(np.bincount(dataset.groups)) == {3}

    def test_dataset_is_reproducible(self, db, engine):
        meals = [Meal(items=[db.records[i].as_item(100.0)]) for i in range(20)]
        a = build_dataset(meals, engine, n_profiles_per_meal=2, seed=42)
        b = build_dataset(meals, engine, n_profiles_per_meal=2, seed=42)
        assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)

    def test_a_different_seed_gives_a_different_dataset(self, db, engine):
        meals = [Meal(items=[db.records[i].as_item(100.0)]) for i in range(20)]
        a = build_dataset(meals, engine, n_profiles_per_meal=2, seed=42)
        b = build_dataset(meals, engine, n_profiles_per_meal=2, seed=43)
        assert not np.array_equal(a.y, b.y)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([0.1, 0.4, 0.8, 0.95])
        metrics = evaluate_predictions(y, y.copy(), 0.7, (0.45, 0.55))
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["auc"] == pytest.approx(1.0)

    def test_auc_is_none_rather_than_faked_when_single_class(self):
        """A fabricated 0.5 would look like a real, if poor, measurement."""
        y = np.array([0.1, 0.2, 0.3])
        assert evaluate_predictions(y, y.copy(), 0.7)["auc"] is None

    def test_auc_reported_at_every_requested_threshold(self):
        y = np.linspace(0.0, 1.0, 40)
        metrics = evaluate_predictions(y, y.copy(), 0.7, (0.45, 0.55, 0.70))
        assert set(metrics["auc_at"]) == {"0.45", "0.55", "0.70"}
        assert all(v == pytest.approx(1.0) for v in metrics["auc_at"].values())

    def test_positive_rate_is_reported(self):
        y = np.array([0.1, 0.8, 0.9, 0.95])
        assert evaluate_predictions(y, y.copy(), 0.7)["positive_rate"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# The trained model
# ---------------------------------------------------------------------------


needs_model = pytest.mark.skipif(
    not LIGHTGBM_PATH.exists(), reason="Stage-1 model not trained; run `make train`"
)


@needs_model
class TestSuitabilityModel:
    @pytest.fixture(scope="class")
    def model(self):
        return SuitabilityModel.load()

    def test_prediction_is_a_probability_like_scalar(self, model, meal, profile):
        value = model.predict(meal, profile)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_feature_contract_matches_the_current_code(self, model):
        """A model trained against an older feature set must not be fed new columns."""
        assert model.columns == feature_names()

    def test_batch_prediction_matches_single_prediction(self, model, meal, profile, db):
        meals = [meal, Meal(items=[db.records[0].as_item(100.0)])]
        batch = model.predict_many(meals, profile)
        singles = [model.predict(m, profile) for m in meals]
        assert np.allclose(batch, singles)

    def test_empty_batch_returns_an_empty_array(self, model, profile):
        assert model.predict_many([], profile).shape == (0,)

    def test_prediction_is_deterministic(self, model, meal, profile):
        assert model.predict(meal, profile) == model.predict(meal, profile)

    def test_the_surrogate_tracks_the_rule_engine(self, model, engine, db):
        """The whole premise: f approximates the guideline score well enough to climb."""
        rng = random.Random(11)
        meals = [
            Meal(items=[db.records[i].as_item(rng.uniform(40, 220))]) for i in range(0, 400, 4)
        ]
        profiles = [sample_profile(rng) for _ in meals]
        predicted = np.array([model.predict(m, p) for m, p in zip(meals, profiles, strict=True)])
        actual = np.array(
            [engine.evaluate(m, p).soft_score for m, p in zip(meals, profiles, strict=True)]
        )
        rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        correlation = float(np.corrcoef(predicted, actual)[0, 1])
        assert rmse < 0.15, f"surrogate RMSE {rmse:.3f} against the rule engine"
        assert correlation > 0.85, f"surrogate correlation {correlation:.3f}"

    def test_the_surrogate_ranks_a_better_meal_higher(self, model, db):
        """Directionally correct is what the optimiser actually needs."""
        adult = UserProfile(age_group=AgeGroup.ADULT, weight_kg=70.0, goal=Goal.WEIGHT_MANAGEMENT)
        lean = Meal(
            items=[
                db.search("chicken breast meat only roasted", limit=1)[0][0].as_item(130.0),
                db.search("lentils cooked", limit=1)[0][0].as_item(120.0),
                db.search("spinach cooked boiled drained", limit=1)[0][0].as_item(120.0),
            ]
        )
        heavy = Meal(
            items=[
                db.search("potatoes french fried", limit=1)[0][0].as_item(250.0),
                db.search("beef ground 80% lean cooked", limit=1)[0][0].as_item(150.0),
            ]
        )
        assert model.predict(lean, adult) > model.predict(heavy, adult)

    def test_xgboost_backend_also_loads(self, meal, profile):
        from foodsense.stage1_prediction.predict import XGBOOST_PATH

        if not XGBOOST_PATH.exists():
            pytest.skip("XGBoost model not trained")
        assert 0.0 <= SuitabilityModel.load("xgboost").predict(meal, profile) <= 1.0

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            SuitabilityModel.load("randomforest")


def test_missing_model_raises_a_helpful_error(monkeypatch, tmp_path):
    import foodsense.stage1_prediction.predict as predict_module

    monkeypatch.setattr(predict_module, "LIGHTGBM_PATH", tmp_path / "absent.txt")
    with pytest.raises(ModelMissingError, match=r"stage1_prediction\.train"):
        SuitabilityModel.load()
