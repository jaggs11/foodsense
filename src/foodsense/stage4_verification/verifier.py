"""Stage 4 -- post-generation verification against USDA ground truth.

For every generated item: fuzzy-match the name to the food database (accept at
score >= 85, else flag ``unmatched`` and substitute the retriever's top candidate);
recompute nutrients from the database times the quantity; compare against the
Stage-3 claims at +/-10% tolerance and correct to database truth; re-run the
RuleEngine safety scan and repair any hard violation by nearest-safe-form or
removal. Emits a :class:`foodsense.schemas.VerificationReport`.

TODO(Phase 4): implement.
"""
