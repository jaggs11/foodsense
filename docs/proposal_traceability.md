# Proposal traceability

Every claim made in the Review-1 proposal, mapped to the code that implements it and the
test that proves it. Filled in as phases land; the point is that no objective can be
"claimed" without a file and a test behind it.

| # | Proposal claim | Module / file | Test |
|---|----------------|---------------|------|
| 1 | Availability-aware CF search space | `stage2_optimizer/space.py` | _(Phase 3)_ |
| 2 | Modification-based minimal editing | `stage2_optimizer/objective.py` | _(Phase 3)_ |
| 3 | Post-generation verification layer | `stage4_verification/verifier.py` | _(Phase 4)_ |
| 4 | Generalised health goals | `constraints/goals.py`, `configs/goals/` | _(Phase 2)_ |
| 5 | Age / life-stage personalisation | `constraints/age_rules.py`, `configs/age_groups/` | _(Phase 2)_ |
| -- | Four-stage MetaPlate pipeline preserved | `pipeline.py` | _(Phase 4)_ |
| -- | Two-corpus comparative analysis | `data/corpora.py`, `experiments/run_dataset_comparison.py` | _(Phase 6)_ |
