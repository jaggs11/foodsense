"""Differential evolution over the mixed continuous/categorical meal-edit space.

Population 40, up to 200 generations, early stop once the RuleEngine reports
validity and the objective plateaus; always returns the best feasible candidate.
Validity is judged by the RuleEngine, never by the surrogate, so the optimiser
cannot win by exploiting its own model.

TODO(Phase 3): implement.
"""
