"""``RuleEngine`` -- the single source of truth for guideline compliance.

``evaluate(meal, profile)`` returns a :class:`foodsense.schemas.RuleEvaluation`:
a continuous score in [0,1] built from soft margins over each threshold, driven
toward zero by any hard-safety violation, plus the structured violation list.

The same engine supplies Stage-1 weak-supervision labels, Stage-2 validity checks
and the Stage-4 safety scan, so the three can never disagree.

TODO(Phase 2): implement.
"""
