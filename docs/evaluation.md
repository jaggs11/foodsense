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
