"""Age/life-stage rules: choking hazards, medication interactions, texture limits.

Choking bans are ``(food_category, form)`` pairs with a nearest-safe-form map, so
the optimiser can repair a hazard by re-forming a food rather than removing it.
Medication rules are activated by ``UserProfile.health_flags`` and act on the
food-database ``tags`` column.

TODO(Phase 2): implement.
"""
