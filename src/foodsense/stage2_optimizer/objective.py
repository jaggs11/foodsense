"""The counterfactual objective.

``lambda1 * max(0, target_score - f(x)) + lambda2 * L1(x, x0)/scale
  + lambda3 * n_items_changed + BIG * hard_safety_violations``

with weights from ``configs/pipeline.yaml``. Pantry items start at 0 g, so adding
one costs both distance and sparsity -- which is what makes an edit minimal rather
than a from-scratch meal.

TODO(Phase 3): implement.
"""
