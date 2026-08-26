# Proposal traceability

Every claim made in the Review-1 proposal, mapped to the code that implements it and
the test that proves it. Filled in as phases land; the point is that no objective can
be "claimed" without a file and a test behind it.

| # | Proposal claim | Module / file | Test |
|---|----------------|---------------|------|
| 1 | Availability-aware CF search space | `stage2_optimizer/space.py` | `test_stage2.py::TestSearchSpace::test_no_reachable_point_contains_an_unavailable_food` (property over 300 random points), `test_variables_are_exactly_planned_union_pantry`; measured 0% violation in `results/cf_comparison.md` |
| 2 | Modification-based minimal editing | `stage2_optimizer/objective.py` | `test_stage2.py::TestObjective` (planned meal has zero distance/sparsity; safety cannot be traded away), `TestDifferentialEvolution::test_the_edit_is_minimal` |
| 3 | Post-generation verification layer | `stage4_verification/verifier.py` | `test_stage4.py::TestVerificationCatchesHallucinations`, `::TestVerificationCatchesSafety`; measured 100% detection on 4 of 5 injected fault classes in `results/verification_eval.md` |
| 4 | Generalised health goals | `constraints/goals.py`, `configs/goals/` | `test_constraints.py::TestConfigs::test_every_goal_has_a_config_with_rules`, `TestRuleEngine::test_the_same_meal_scores_differently_for_different_profiles` |
| 5 | Age / life-stage personalisation | `constraints/age_rules.py`, `configs/age_groups/` | `test_constraints.py::TestChokingHazards` (one case per banned pair), `TestMedicationInteractions`, `TestTexture` |
| -- | Four-stage MetaPlate pipeline preserved | `pipeline.py` | `test_worked_examples.py::TestEveryScenario::test_the_pipeline_completes_all_four_stages` |
| -- | Offline-first: no key, no network | `stage3_rag/providers.py` | `test_worked_examples.py::...::test_the_offline_path_needs_no_provider`, `test_stage4.py::TestProviderContract::test_an_unavailable_provider_falls_back_rather_than_failing` |
| -- | Scenario 1 worked example | `scenarios.py` | `test_worked_examples.py::TestToddlerChoking` (grapes quartered, peanuts removed, suitability improves) |
| -- | Scenario 2 worked example | `scenarios.py` | `test_worked_examples.py::TestElderlySodium` (final sodium <= 500 mg, <= 4 edits, verified) |
| -- | Scenario 3 worked example | `scenarios.py` | `test_worked_examples.py::TestAdultWeight` (goal thresholds met with no safety rule involved) |
| -- | Nearest-safe-form repair, not removal | `stage2_optimizer/space.py` form costs | `test_stage2.py::TestDifferentialEvolution::test_it_repairs_grapes_by_quartering_rather_than_removing`, `test_it_removes_the_peanuts` |
| -- | Two-corpus comparative analysis | `data/corpora.py`, `experiments/run_dataset_comparison.py` | `test_data.py::TestCorpora`, Stage-1 metrics on both corpora in `models/stage1_metrics.json` |

## Safety rules, individually

Choking bans and medication interactions are tested by parametrising over the YAML
itself, so a rule added to `configs/` automatically gains a test case and a rule no
food in the database can trigger fails rather than looking enforced.

| Rule family | Config | Test |
|-------------|--------|------|
| Choking `(hazard_class, form)` bans | `configs/age_groups/toddler.yaml` | `TestChokingHazards::test_every_banned_pair_is_caught` / `..._drives_the_score_toward_zero` |
| Nearest-safe-form repair | same | `test_quartering_grapes_removes_the_hazard`, `test_a_suggested_form_is_always_one_the_food_can_take` |
| Hazards with no safe form | same | `test_hazards_with_no_safe_form_can_only_be_removed` |
| Age-gated safe forms (whole nuts) | same | `test_whole_nuts_have_no_safe_form_at_eighteen_months`, `..._can_be_ground_for_an_older_toddler` |
| Medication exclusions (MAOI, statin, metformin) | `configs/health_flags.yaml` | `TestMedicationInteractions::test_every_exclusion_rule_fires_on_a_tagged_food` |
| Numeric medication ceilings (warfarin, ACE/K-sparing) | same | `test_every_numeric_ceiling_fires_when_exceeded` |
| Dysphagia texture (IDDSI) | same | `TestTexture` |
| Honey below 12 months | `configs/age_groups/toddler.yaml` | `TestAgeGatedFoods::test_honey_is_barred_below_twelve_months` |
| No added sugar below 24 months | same | `TestAgeGatedFoods::test_added_sugar_rule_is_implied_below_two_years` |

## Stage-1 design claims

| Claim | Where | Evidence |
|-------|-------|----------|
| The surrogate learns the guidelines, not the corpus | `stage1_prediction/train.py` | Nutrition5k (a different corpus) RMSE 0.0573 vs Food.com held-out 0.0571 |
| Validity is judged by rules, never the surrogate | `constraints/engine.py::RuleEngine.is_valid` | `test_constraints.py::TestRuleEngine::test_is_valid_requires_both_safety_and_the_target` |
| Hard safety is excluded from the label by design | `stage1_prediction/labels.py` | `test_stage1.py::TestDataset::test_labels_ignore_hard_safety` |
| The surrogate tracks the rule engine closely enough to climb | `stage1_prediction/predict.py` | `test_stage1.py::TestSuitabilityModel::test_the_surrogate_tracks_the_rule_engine` |
