# Evaluation

> **Phase 0 skeleton.** Every metric below is defined here and computed by a script in
> `experiments/`. Nothing is written into `results/` until that script has actually run,
> and no number appears in this repository that was not produced by code.

## How to regenerate everything

```bash
make eval      # or ./make.ps1 eval on Windows
```

## Planned artefacts

| Script | Output | Metrics |
|--------|--------|---------|
| `experiments/run_cf_eval.py` | `results/cf_comparison.{csv,md}` | validity, usable validity, normalised L1/L2, sparsity, availability-violation %, safety-violation %, runtime -- by age group |
| `experiments/run_validity_decomposition.py` | `results/validity_decomposition.{csv,md}` | why each invalid FoodSense case was invalid; validity by age group for every method. Reads `cf_comparison_raw.csv`, re-runs nothing |
| `experiments/run_lambda_sweep.py` | `results/lambda_sweep.{csv,md}` | sensitivity of validity, edits and distance to `lambda_validity`, with safety and availability held up as controls |
| `experiments/run_verification_eval.py` | `results/verification_eval.md` | share of Stage-3 outputs with >=1 corrected quantity or unsafe item, before vs after Stage 4; plus injected-fault detection split into caught-by-construction and caught-by-re-derivation |
| `experiments/run_dataset_comparison.py` | `results/dataset_comparison.md` | Stage-1 RMSE / R^2 / thresholded AUC on Food.com vs Nutrition5k |
| `experiments/run_llm_benchmark.py` | `results/llm_benchmark.md` | macro RMSE, goal consistency, diversity by provider |

## A verification gap that was found the hard way

Worth recording, because the shape of the mistake generalises.

`foodsense serve` shipped broken. The `api` package sat at the repository root
and was never installed — `pyproject.toml` packages `src/` only — so
`from foodsense.api.main import ...` resolved solely when the process happened to
start in the repo root. Run from anywhere else, the command raised
`ModuleNotFoundError`.

**538 tests passed against it**, including tests that imported the API and
exercised every endpoint, because `pyproject.toml` carried `pythonpath = ["."]`
for pytest. The suite was testing an import path the shipped CLI did not have.
The Phase-6 clean-clone verification did not catch it either: it ran `pytest` and
`foodsense demo` from inside the clone, and never ran `serve` from a different
working directory — so the one command with the defect was the one command not
exercised.

Two things changed. The package moved to `src/foodsense/api/` and is installed
like everything else, and the `pythonpath` hack is gone, so the tests now import
it the way the CLI does. `tests/test_api.py` adds five tests that assert the
property rather than the behaviour: that the module resolves from *inside the
installed package*, and that importing it — and running `serve --help` — works in
a subprocess started in a temporary directory. A subprocess is the only honest
check here; an in-process import proves nothing when pytest has already arranged
`sys.path` to its liking.

The general lesson: **a test that passes because of the harness is not evidence
about the product.** Anything that only works from one working directory should
be tested from a different one.

## Reading the results honestly

Three habits this project holds itself to, because each of them is a way a results
directory can be technically true and still misleading.

**Numbers in prose are computed, not typed.** The narrative paragraphs in
`cf_comparison.md` interpolate their figures from the same aggregate the tables are
rendered from. A hand-written number goes stale the first time an experiment is
re-run, and a stale number in a results file is indistinguishable from an invented
one.

**A rate that cannot fail is reported separately from one that can.** The Stage-4
fault study splits its faults into those caught *by construction* -- a food id
absent from the database fails a dictionary lookup; a claim 90% over a 10%
tolerance is outside it by arithmetic -- and those that require the verifier to
independently recompute or re-derive something. Pooling them yields a flattering
"100% detected" that measures nothing. The second block is the number that counts.

**A zero earned by doing nothing is not the same zero.** DiCE-genetic shows a 0%
availability-violation rate because it never edits the meal. FoodSense shows 0%
while actively editing, and shows it because an unavailable food has no decision
variable to begin with. Both caveats are printed next to the table rather than in
a footnote, because a reader skimming the table is exactly the reader who will
otherwise conflate them.

## Metric definitions

_(Phase 3-6: each metric gets its formula and rationale here as it is implemented.)_

---

## Archived note: what the surrogate's error was measured against

*Recorded during P0 of the repositioning. Archived rather than acted on: the
studies this belongs to measured a population the system no longer serves, and
are parked under ruling R10 until the P9 re-run on the toddler and condition
harness.*

The surrogate's reported error was computed against the **noised training label**
(0.0572). That is the wrong denominator for the claim it was being used to
support. The label carries deliberate Gaussian noise (sigma = 0.05) as weak
supervision; measuring the model against it charges the model for noise that was
added on purpose.

Measured instead against the **decision-relevant quantity** -- the rule engine's
own `soft_score`, which is what the optimiser is actually climbing -- the error
is approximately 0.0306.

Both numbers are about the same model. The first answers "how closely did it fit
the labels it was given", the second "how closely does it track the thing the
optimiser cares about". Only the second bears on whether Stage 2's search is
guided by a faithful objective, which is the claim the figure was cited for.

Neither number is re-derived here, and neither is a new result: they are recorded
so the correction is not lost between the pre-repositioning evaluation and the
P9 re-run, which will recompute both on the toddler and condition harness.

---

## Archived note: what the Stage-4 repair hypothesis found

*Recorded during P0 of the repositioning.*

The Phase 4.5 plan carried a suspected **second-pass control-flow bug** in
`stage4_verification/verifier.py:_repair()`. Investigation confirmed a real defect
and located it somewhere else.

The second pass was sound. Four existing tests already covered it, including the
double-bind case it was written for (a frankfurter served to someone with dysphagia
*and* an MAOI prescription: mincing answers the texture rule and does nothing about
the tyramine), and the log-superseding behaviour that keeps the safety-fix log
describing the plate actually returned.

The defect was in **pass 1**, and it was a modelling error rather than a
control-flow one. `_repair()` built its set of offending items from
`Violation.offending_items`, which carries food_ids. Safety in this project is a
property of `(food_id, form)` — that is the premise the entire constraint layer
rests on, and the reason a hazard can be repaired by quartering a grape instead of
deleting it. Keying repair on food_id alone therefore reached items that were never
at fault.

Reproduced with beef served two ways in one meal: 40 g in chunks, which trips
`toddler.choking.meat_chunk`, and 30 g already minced, which is safe. Pass 1
re-formed the chunks correctly, then found the minced portion under the same
food_id, called `nearest_safe_form()`, got back the form it was already in, and fell
through to the removal branch. Thirty grams left the plate; the log recorded that
the food "has no safe preparation for this profile", which is false, since minced is
precisely its safe preparation; and `final_pass` stayed `True`, because the meal
that remained was genuinely safe.

Both passes now key on `(food_id, form)`, and `RuleEngine.structural_violations()`
returns the checks with the item that caused each one. `stage3_rag/translate.py`
already paired food_id with form when deciding which violations a candidate had
fixed; Stage 4 was the only place that did not.

Two things are worth keeping about this. The hypothesis named the right function for
the wrong reason, and following it still found a real bug that silently removed food
and then filed a false report about it — which is the same claim the project makes
for Stage 4 itself, that a check worth having is one that catches things its author
did not predict. And the bug was invisible to every existing test because each of
them put a food in a meal once.

---

## Archived note: the zero-fill regime the archived results were computed under

*Recorded during P1 of the repositioning (ruling R13).*

Every result in `results/archive/v1_multiage_counterfactual/` was computed against a
database in which `build_food_db.py` ended its nutrient handling with `fillna(0.0)`.
A nutrient USDA does not report for a food was stored as a measured zero, and no
consumer could tell the two apart.

The scale, measured during P1 against the rebuilt database:

| | |
|---|---|
| Nutrient cells that were fabricated zeros | 11,281 of 85,470 (13.2%) |
| Foods with at least one fabricated zero | 2,590 of 2,590 (100%) |
| `added_sugars_g` | unreported for **100%** of rows |
| `trans_fat_g` | unreported for 54.2% |
| `vitamin_k_ug` / `vitamin_d_ug` / `vitamin_e_mg` | 35.4% / 32.0% / 31.1% |
| Fully reported for every food | `energy_kcal`, `water_g` only |

`added_sugars_g` being entirely absent is *not* a latent rule bug. The project had
already established that neither USDA release it consumes reports FDC nutrient 1235,
and the rule keyed on added sugars reads a documented proxy
(`constraints/goals.estimate_added_sugars`, which substitutes total sugars in the
sweets, snack, baked and cereal categories and in sweetened beverages) rather than
the column. See `configs/age_groups/toddler.yaml` and the comment block above that
function. The column was fabricated; the rule that appears to depend on it was not
reading the fabrication.

**The archived results are not restated or recomputed.** They were produced under
this regime, they remain exactly as produced, and this note exists so a reader knows
which regime that was. Under ruling R13 the database is honest from P1 onward while
the pipeline's answers are held constant through one named function
(`data/coverage.zero_filled_vector`), so that the commit which moves the numbers is a
separate, dated, deliberate one in P3.

For scale, the soft-score effect measured on the three demo scenarios — treating a
rule as `undetermined` when any nutrient its quantity depends on is unreported by any
item in the meal, rather than reading the zero:

| scenario | soft (zero-fill) | soft (honest) | delta | rules undetermined |
|---|---|---|---|---|
| `toddler_choking` | 0.4863 | 0.4770 | −0.0092 | `vitamin_d_ug`, `saturated_fat_g` |
| `elderly_sodium` | 0.2641 | 0.2641 | 0.0000 | none |
| `adult_weight` | 0.2795 | 0.2795 | 0.0000 | none |

Measurement only; nothing in the pipeline was changed by it, and these numbers are
not a result — they are a bound on how much P3's switch can move. They also close a
question the Phase 4.5 plan left open: this is **not** the ~0.021 surrogate /
rule-engine boundary discrepancy. Two of the three scenarios move by exactly zero,
and the mechanism does not work in any case — surrogate and rule engine consumed the
same zero-filled features, so a shared bias cancels rather than separating them.
