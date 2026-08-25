"""Weak-supervision label generation from the RuleEngine.

Label = ``RuleEngine.evaluate(...).score`` perturbed by ``N(0, 0.05)`` and clipped
to [0,1]. The noise keeps the surrogate from memorising the rule boundaries exactly
and forces it to learn a smooth approximation -- which is the whole point of having
a model to optimise against.

TODO(Phase 2): implement.
"""
