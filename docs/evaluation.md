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
| `experiments/run_cf_eval.py` | `results/cf_comparison.{csv,md}` | validity, normalised L1/L2, sparsity, availability-violation %, safety-violation %, runtime -- by age group |
| `experiments/run_verification_eval.py` | `results/verification_eval.md` | share of Stage-3 outputs with >=1 corrected quantity or unsafe item, before vs after Stage 4 |
| `experiments/run_dataset_comparison.py` | `results/dataset_comparison.md` | Stage-1 RMSE / R^2 / thresholded AUC on Food.com vs Nutrition5k |
| `experiments/run_llm_benchmark.py` | `results/llm_benchmark.md` | macro RMSE, goal consistency, diversity by provider |

## Metric definitions

_(Phase 3-6: each metric gets its formula and rationale here as it is implemented.)_
