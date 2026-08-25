"""Goal threshold logic and the per-meal glycemic-load estimator.

Loads ``configs/goals/*.yaml`` and evaluates a meal's nutrient totals against the
selected goal. The GL estimator assigns a glycemic-index class per food category
and computes ``GI * available_carb / 100`` summed over items -- keeping the
``glycemic_control`` goal faithful to MetaPlate's glucose origin.

TODO(Phase 2): implement.
"""
