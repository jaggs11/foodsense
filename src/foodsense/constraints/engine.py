"""``RuleEngine`` -- the single source of truth for guideline compliance.

One object answers "is this meal appropriate for this person?", and three
different parts of the pipeline ask it:

* **Stage 1** turns its score into weak-supervision labels, so the learned
  surrogate is an approximation of *these* rules and no others.
* **Stage 2** asks it whether a candidate is valid. Deliberately not the
  surrogate: an optimiser judged by its own model can win by finding the model's
  blind spots, so validity is always decided by the rules themselves.
* **Stage 4** re-runs its safety scan on whatever Stage 3 actually produced.

Because all three call the same engine, they cannot disagree about what "safe"
or "compliant" means.

Two scores come out of an evaluation, and the difference matters:

``soft_score``
    The numeric guidelines alone -- energy, protein, sodium, glycemic load,
    macronutrient shares. Smooth, differentiable enough to optimise, and the
    thing the Stage-1 surrogate is trained to predict.
``score``
    ``soft_score`` driven toward zero by any hard-safety violation. This is the
    honest verdict on a meal and what verification reports.

The surrogate learns ``soft_score`` rather than ``score`` on purpose. Whether a
meal contains whole grapes is a discrete fact about ``(hazard_class, form)``
pairs; no nutrient vector encodes it, so a model given nutrient features could
only ever fit that part of the label as noise. Hard safety is therefore enforced
where it can actually be enforced -- as an explicit penalty term in the Stage-2
objective and as a scan in Stage 4. See docs/architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from foodsense import CONFIG_DIR
from foodsense.constraints.age_rules import (
    StructuralViolation,
    check_age_gated_foods,
    check_choking,
    check_excluded_tags,
    check_texture,
    implicit_flags,
    load_age_config,
    load_flag_rules,
)
from foodsense.constraints.goals import (
    Rule,
    load_goal_config,
    meal_metrics,
    satisfaction,
)
from foodsense.data.fdc import FoodDB, get_food_db
from foodsense.schemas import Meal, MealItem, RuleEvaluation, UserProfile, Violation

__all__ = ["RuleEngine", "RuleEngineConfig", "get_rule_engine"]

PIPELINE_CONFIG = CONFIG_DIR / "pipeline.yaml"


@dataclass(frozen=True, slots=True)
class RuleEngineConfig:
    """Scoring parameters from ``configs/pipeline.yaml``."""

    softness: float = 0.15
    hard_violation_factor: float = 0.10
    soft_violation_threshold: float = 0.5

    @classmethod
    def load(cls) -> RuleEngineConfig:
        if not PIPELINE_CONFIG.exists():
            return cls()
        raw = yaml.safe_load(PIPELINE_CONFIG.read_text(encoding="utf-8")) or {}
        section = raw.get("rule_engine") or {}
        return cls(
            softness=float(section.get("softness", 0.15)),
            hard_violation_factor=float(section.get("hard_violation_factor", 0.10)),
            soft_violation_threshold=float(section.get("soft_violation_threshold", 0.5)),
        )


class RuleEngine:
    """Evaluates a meal against every rule that applies to a profile."""

    def __init__(self, db: FoodDB | None = None, config: RuleEngineConfig | None = None) -> None:
        self.db = db or get_food_db()
        self.config = config or RuleEngineConfig.load()

    # -- rule assembly ------------------------------------------------------

    def rules_for(self, profile: UserProfile) -> list[Rule]:
        """Every numeric rule in force for this profile: age, then goal, then flags.

        Later rules override earlier ones on the same quantity, so a flag can
        tighten what the age group allows -- the hypertension sodium ceiling
        replaces the general one rather than fighting it.
        """
        by_quantity: dict[str, Rule] = {}
        for rule in load_age_config(profile.age_group).rules(profile):
            by_quantity[rule.quantity] = rule
        for rule in load_goal_config(profile.goal).rules():
            by_quantity[rule.quantity] = rule

        flag_rules = load_flag_rules()
        for flag in self._effective_flags(profile):
            flag_rule = flag_rules.get(flag)
            if flag_rule is None:
                continue
            for rule in flag_rule.rules():
                by_quantity[rule.quantity] = rule
        return list(by_quantity.values())

    def _effective_flags(self, profile: UserProfile):
        return [*profile.health_flags, *implicit_flags(profile)]

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, meal: Meal | list[MealItem], profile: UserProfile) -> RuleEvaluation:
        """Score a meal for a profile and list everything it breaks."""
        items = meal.items if isinstance(meal, Meal) else list(meal)
        metrics = meal_metrics(items, self.db)

        per_rule: dict[str, float] = {}
        violations: list[Violation] = []
        weighted_total = 0.0
        weight_total = 0.0
        n_hard = 0

        for rule in self.rules_for(profile):
            value = metrics.get(rule.quantity)
            if value is None:
                # A rule written against a quantity we cannot measure is skipped
                # rather than silently scored as satisfied.
                continue
            score = satisfaction(value, rule.threshold, self.config.softness)
            per_rule[rule.rule_id] = score

            if rule.severity == "soft":
                weighted_total += score * rule.weight
                weight_total += rule.weight

            if not rule.threshold.is_satisfied(value):
                if rule.severity == "hard":
                    n_hard += 1
                violations.append(self._numeric_violation(rule, value))

        for structural in self._structural_checks(items, profile):
            n_hard += 1
            violations.append(
                Violation(
                    rule_id=structural.rule_id,
                    severity="hard",
                    message=structural.message,
                    offending_items=[structural.item.food_id],
                    suggested_form=structural.suggested_form,
                )
            )
            per_rule[structural.rule_id] = 0.0

        soft_score = weighted_total / weight_total if weight_total else 1.0
        score = soft_score * (self.config.hard_violation_factor**n_hard)

        return RuleEvaluation(
            score=_clip01(score),
            soft_score=_clip01(soft_score),
            violations=violations,
            per_rule=per_rule,
        )

    def _numeric_violation(self, rule: Rule, value: float) -> Violation:
        crossed = (
            rule.threshold.maximum
            if rule.threshold.maximum is not None and value > rule.threshold.maximum
            else rule.threshold.minimum
        )
        return Violation(
            rule_id=rule.rule_id,
            severity=rule.severity,
            message=rule.message,
            observed=round(value, 3),
            threshold=crossed,
        )

    def _structural_checks(
        self, items: list[MealItem], profile: UserProfile
    ) -> list[StructuralViolation]:
        """Hard-safety checks about *what* is in the meal rather than how much."""
        return [
            *check_choking(items, profile, self.db),
            *check_excluded_tags(items, profile, self.db),
            *check_texture(items, profile, self.db),
            *check_age_gated_foods(items, profile, self.db),
        ]

    # -- convenience --------------------------------------------------------

    def is_safe(self, meal: Meal | list[MealItem], profile: UserProfile) -> bool:
        """True when no hard-safety rule is broken. Cheaper than a full evaluation."""
        items = meal.items if isinstance(meal, Meal) else list(meal)
        if self._structural_checks(items, profile):
            return False
        metrics = meal_metrics(items, self.db)
        return all(
            rule.threshold.is_satisfied(metrics[rule.quantity])
            for rule in self.rules_for(profile)
            if rule.severity == "hard" and rule.quantity in metrics
        )

    def hard_violations(self, meal: Meal | list[MealItem], profile: UserProfile) -> list[Violation]:
        return self.evaluate(meal, profile).hard_violations

    def is_valid(
        self, meal: Meal | list[MealItem], profile: UserProfile, target_score: float
    ) -> bool:
        """Stage-2 validity: safe, and scoring at least ``target_score``.

        Called with the rule engine's own score, never the surrogate's, so the
        optimiser cannot declare victory by exploiting its model.
        """
        evaluation = self.evaluate(meal, profile)
        return evaluation.is_safe and evaluation.score >= target_score


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


@lru_cache(maxsize=1)
def get_rule_engine() -> RuleEngine:
    """Process-wide cached engine, sharing the cached food database."""
    return RuleEngine()
