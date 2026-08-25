"""Counterfactual baselines for the ablation table.

- DiCE (random and genetic) via ``dice-ml`` -- model-agnostic, availability-blind.
- Wachter-style ablation: our own DE with validity + L1 only (no availability,
  safety or sparsity terms) -- isolates the contribution of each constraint.
- Greedy substitution heuristic.

All run under identical evaluation budgets.

TODO(Phase 3): implement.
"""
