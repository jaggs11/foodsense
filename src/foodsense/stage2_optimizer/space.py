"""Decision-variable construction -- where availability-awareness actually lives.

Variables are built from ``planned_meal union pantry`` and nothing else, so an
unavailable food is not penalised: it has no variable to take a value. Each item
contributes a continuous ``quantity_g`` bounded by a per-food maximum, plus a
categorical ``form`` index decoded through that food's ``allowed_forms``.

TODO(Phase 3): implement.
"""
