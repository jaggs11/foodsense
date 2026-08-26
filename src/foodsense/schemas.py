"""Shared type contract for the whole FoodSense pipeline.

Every stage speaks in the types defined here, so the four stages can be developed,
tested and swapped independently:

    Stage 1  Meal + UserProfile          -> Stage1Result
    Stage 2  Meal + Pantry + UserProfile -> Stage2Result
    Stage 3  Stage2Result                -> Stage3Result
    Stage 4  Stage3Result                -> VerificationReport
    all four                             -> PipelineTrace

Design note (see docs/architecture.md): a meal item is a ``(food_id, quantity_g, form)``
triple, never just a food. Choking hazards are a property of the *pair* -- whole grapes
are unsafe for a toddler, quartered grapes are not -- so ``form`` has to be a
first-class decision variable, not metadata.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

__all__ = [
    "NUTRIENTS",
    "AgeGroup",
    "Form",
    "Goal",
    "HealthFlag",
    "ItemChange",
    "ItemCorrection",
    "Meal",
    "MealDiff",
    "MealItem",
    "NutrientVector",
    "PipelineTrace",
    "RuleEvaluation",
    "SafetyFix",
    "Stage1Result",
    "Stage2Result",
    "Stage3Result",
    "UserProfile",
    "VerificationReport",
    "Violation",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Form(StrEnum):
    """Preparation form of a meal item.

    The first ten are the core set from the design brief. ``SLICED_ROUNDS`` and
    ``SPOONFUL`` exist because the AAP/CDC choking rules are written against them
    specifically -- hot dogs cut into rounds, and nut butter eaten by the spoonful,
    are named hazards that need a form to attach to.
    """

    WHOLE = "whole"
    CHOPPED = "chopped"
    QUARTERED = "quartered"
    SLICED = "sliced"
    MASHED = "mashed"
    PUREED = "pureed"
    MINCED = "minced"
    GROUND = "ground"
    SOFT_COOKED = "soft_cooked"
    THIN_SPREAD = "thin_spread"
    SLICED_ROUNDS = "sliced_rounds"
    SPOONFUL = "spoonful"


class AgeGroup(StrEnum):
    """Life-stage band. ``ADULT`` is the default; the other two are the deep-dive cases."""

    TODDLER = "toddler"  # 1-3 years
    ADULT = "adult"  # 4-64 years
    OLDER_ADULT = "older_adult"  # 65+ years


class Goal(StrEnum):
    """Configurable health goal (extension #4 over MetaPlate)."""

    GLYCEMIC_CONTROL = "glycemic_control"
    WEIGHT_MANAGEMENT = "weight_management"
    BALANCED_NUTRITION = "balanced_nutrition"


class HealthFlag(StrEnum):
    """Conditions and medications that activate extra constraint rules."""

    HYPERTENSION = "hypertension"
    DIABETES = "diabetes"
    DYSPHAGIA = "dysphagia"
    IRON_FOCUS = "iron_focus"
    STRICT_NO_ADDED_SUGAR = "strict_no_added_sugar"
    # medication -- food interactions
    WARFARIN = "warfarin"
    MAOI = "maoi"
    ACE_INHIBITOR_OR_K_SPARING_DIURETIC = "ace_inhibitor_or_k_sparing_diuretic"
    STATIN = "statin"
    METFORMIN = "metformin"


Severity = Literal["hard", "soft"]
ChangeType = Literal["unchanged", "modified", "added", "removed"]


# ---------------------------------------------------------------------------
# Nutrients
# ---------------------------------------------------------------------------

#: Canonical nutrient field order. The food database stores these per 100 g;
#: meal totals are the same fields in absolute units. Keeping one ordered tuple
#: means feature vectors, database columns and diffs never drift apart.
NUTRIENTS: tuple[str, ...] = (
    "energy_kcal",
    "protein_g",
    "carbohydrate_g",
    "sugars_g",
    "added_sugars_g",
    "fiber_g",
    "fat_g",
    "saturated_fat_g",
    "trans_fat_g",
    "monounsaturated_fat_g",
    "polyunsaturated_fat_g",
    "cholesterol_mg",
    "sodium_mg",
    "potassium_mg",
    "calcium_mg",
    "iron_mg",
    "magnesium_mg",
    "zinc_mg",
    "phosphorus_mg",
    "copper_mg",
    "selenium_ug",
    "vitamin_a_rae_ug",
    "vitamin_c_mg",
    "vitamin_d_ug",
    "vitamin_e_mg",
    "vitamin_k_ug",
    "thiamin_mg",
    "riboflavin_mg",
    "niacin_mg",
    "vitamin_b6_mg",
    "folate_dfe_ug",
    "vitamin_b12_ug",
    "water_g",
)


class NutrientVector(BaseModel):
    """A bag of nutrient amounts.

    Interpreted either as *per 100 g* (a food-database row) or as *absolute totals*
    (a meal). The class does not track which -- callers do, because the arithmetic
    is identical either way.
    """

    model_config = ConfigDict(extra="forbid")

    energy_kcal: float = 0.0
    protein_g: float = 0.0
    carbohydrate_g: float = 0.0
    sugars_g: float = 0.0
    added_sugars_g: float = 0.0
    fiber_g: float = 0.0
    fat_g: float = 0.0
    saturated_fat_g: float = 0.0
    trans_fat_g: float = 0.0
    monounsaturated_fat_g: float = 0.0
    polyunsaturated_fat_g: float = 0.0
    cholesterol_mg: float = 0.0
    sodium_mg: float = 0.0
    potassium_mg: float = 0.0
    calcium_mg: float = 0.0
    iron_mg: float = 0.0
    magnesium_mg: float = 0.0
    zinc_mg: float = 0.0
    phosphorus_mg: float = 0.0
    copper_mg: float = 0.0
    selenium_ug: float = 0.0
    vitamin_a_rae_ug: float = 0.0
    vitamin_c_mg: float = 0.0
    vitamin_d_ug: float = 0.0
    vitamin_e_mg: float = 0.0
    vitamin_k_ug: float = 0.0
    thiamin_mg: float = 0.0
    riboflavin_mg: float = 0.0
    niacin_mg: float = 0.0
    vitamin_b6_mg: float = 0.0
    folate_dfe_ug: float = 0.0
    vitamin_b12_ug: float = 0.0
    water_g: float = 0.0

    # -- arithmetic ---------------------------------------------------------

    def scaled(self, factor: float) -> NutrientVector:
        """Return a copy with every nutrient multiplied by ``factor``.

        Used to turn a per-100 g row into the contribution of ``quantity_g`` grams
        (``factor = quantity_g / 100``).
        """
        return NutrientVector(**{n: getattr(self, n) * factor for n in NUTRIENTS})

    def __add__(self, other: NutrientVector) -> NutrientVector:
        return NutrientVector(**{n: getattr(self, n) + getattr(other, n) for n in NUTRIENTS})

    @classmethod
    def zeros(cls) -> NutrientVector:
        return cls()

    @classmethod
    def sum(cls, vectors: list[NutrientVector]) -> NutrientVector:
        total = cls.zeros()
        for v in vectors:
            total = total + v
        return total

    def as_tuple(self) -> tuple[float, ...]:
        """Nutrients in canonical :data:`NUTRIENTS` order -- the feature-vector view."""
        return tuple(getattr(self, n) for n in NUTRIENTS)

    def as_dict(self) -> dict[str, float]:
        return {n: getattr(self, n) for n in NUTRIENTS}


# ---------------------------------------------------------------------------
# Meals and profiles
# ---------------------------------------------------------------------------


class MealItem(BaseModel):
    """One component of a meal: a food, an amount, and how it is prepared."""

    model_config = ConfigDict(extra="forbid")

    food_id: str = Field(..., description="USDA FoodData Central id (fdc_id) as a string")
    name: str = Field(..., description="Human-readable food name, ideally the USDA description")
    quantity_g: float = Field(..., ge=0.0, description="Amount in grams")
    form: Form = Field(default=Form.WHOLE, description="Preparation form")

    @field_validator("food_id", mode="before")
    @classmethod
    def _coerce_food_id(cls, v: Any) -> str:
        return str(v)

    def key(self) -> tuple[str, str]:
        """Identity used when diffing two meals."""
        return (self.food_id, self.form.value)


class Meal(BaseModel):
    """An ordered list of meal items, plus convenience accessors."""

    model_config = ConfigDict(extra="forbid")

    items: list[MealItem] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):  # type: ignore[override]
        return iter(self.items)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_quantity_g(self) -> float:
        return sum(i.quantity_g for i in self.items)

    def food_ids(self) -> set[str]:
        return {i.food_id for i in self.items}

    def nonzero(self) -> Meal:
        """Drop items whose quantity has been optimised down to (effectively) zero."""
        return Meal(items=[i for i in self.items if i.quantity_g > 1e-6])


class UserProfile(BaseModel):
    """Who we are recommending for."""

    model_config = ConfigDict(extra="forbid")

    age_group: AgeGroup = AgeGroup.ADULT
    age_months: int | None = Field(
        default=None, ge=0, description="Exact age in months; used for toddler sub-rules (<24 mo)"
    )
    weight_kg: float | None = Field(
        default=None, gt=0, description="Body weight; drives per-kg protein targets"
    )
    goal: Goal = Goal.BALANCED_NUTRITION
    health_flags: list[HealthFlag] = Field(default_factory=list)
    notes: str | None = None

    def has(self, flag: HealthFlag) -> bool:
        return flag in self.health_flags

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age_years(self) -> float | None:
        return None if self.age_months is None else round(self.age_months / 12.0, 2)


# ---------------------------------------------------------------------------
# Rule engine output
# ---------------------------------------------------------------------------


class Violation(BaseModel):
    """One broken rule.

    ``severity == "hard"`` means a safety rule (choking hazard, medication
    interaction, dysphagia texture). Hard violations drive the RuleEngine score
    toward zero and are never traded off against convenience.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    severity: Severity
    message: str
    offending_items: list[str] = Field(
        default_factory=list, description="food_ids (or names) responsible"
    )
    observed: float | None = None
    threshold: float | None = None
    suggested_form: Form | None = Field(
        default=None, description="Nearest safe form, when the rule has one"
    )


class RuleEvaluation(BaseModel):
    """Result of ``RuleEngine.evaluate(meal, profile)``."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0.0, le=1.0, description="Continuous guideline-compliance score")
    soft_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Guideline score from the numeric rules alone, before hard-safety violations "
            "drive it toward zero. This is what the Stage-1 surrogate learns: hard safety "
            "is a discrete property of (hazard_class, form) that no nutrient vector can "
            "express, so Stage 2 enforces it with an explicit penalty term instead."
        ),
    )
    violations: list[Violation] = Field(default_factory=list)
    per_rule: dict[str, float] = Field(
        default_factory=dict, description="rule_id -> soft satisfaction in [0,1]"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hard_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "hard"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_safe(self) -> bool:
        """True when no hard-safety rule is broken."""
        return not any(v.severity == "hard" for v in self.violations)


# ---------------------------------------------------------------------------
# Diffs
# ---------------------------------------------------------------------------


class ItemChange(BaseModel):
    """A single edit the optimiser made, rendered for the UI's before/after panel."""

    model_config = ConfigDict(extra="forbid")

    change_type: ChangeType
    food_id: str
    name: str
    old_quantity_g: float | None = None
    new_quantity_g: float | None = None
    old_form: Form | None = None
    new_form: Form | None = None
    reason: str | None = Field(default=None, description="One-line human explanation")


class MealDiff(BaseModel):
    """The full edit set, plus the minimality metrics that justify calling it minimal."""

    model_config = ConfigDict(extra="forbid")

    changes: list[ItemChange] = Field(default_factory=list)
    l1_distance_g: float = Field(default=0.0, description="Sum of |new - old| grams")
    l2_distance_g: float = 0.0
    n_items_changed: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def edits(self) -> list[ItemChange]:
        return [c for c in self.changes if c.change_type != "unchanged"]


# ---------------------------------------------------------------------------
# Stage results
# ---------------------------------------------------------------------------


class Stage1Result(BaseModel):
    """Suitability of a meal for a profile, per the learned surrogate."""

    model_config = ConfigDict(extra="forbid")

    suitability: float = Field(..., ge=0.0, le=1.0)
    nutrients: NutrientVector = Field(default_factory=NutrientVector)
    glycemic_load: float | None = None
    rule_evaluation: RuleEvaluation | None = Field(
        default=None, description="Ground-truth rule score alongside the surrogate, for comparison"
    )
    model_name: str = "lightgbm"
    runtime_s: float = 0.0


class Stage2Result(BaseModel):
    """Output of the availability-aware counterfactual optimiser."""

    model_config = ConfigDict(extra="forbid")

    optimized_meal: Meal
    diff: MealDiff
    suitability_before: float
    suitability_after: float
    objective_value: float
    valid: bool = Field(..., description="Judged by the RuleEngine, never by the surrogate")
    rule_evaluation_after: RuleEvaluation | None = None
    n_generations: int = 0
    n_evaluations: int = 0
    runtime_s: float = 0.0
    method: str = "foodsense_de"
    search_space_size: int = Field(default=0, description="|planned union pantry|")


class Stage3Result(BaseModel):
    """Natural-language rendering of the optimised meal, grounded in USDA names."""

    model_config = ConfigDict(extra="forbid")

    items: list[MealItem] = Field(default_factory=list)
    text: str = ""
    rationale: list[str] = Field(default_factory=list)
    provider: str = "template"
    retrieved_candidates: dict[str, list[str]] = Field(
        default_factory=dict, description="query -> retrieved USDA names (BM25), for traceability"
    )
    claimed_nutrients: NutrientVector | None = Field(
        default=None, description="What the provider asserted, before Stage-4 recomputation"
    )
    fallback_used: bool = Field(
        default=False, description="True when an LLM provider failed and the template took over"
    )
    fallback_reason: str = Field(
        default="",
        description=(
            "Why the provider was not used, when it was not. Carried on the trace "
            "because the UI has to show it: 'the LLM was skipped' is alarming and "
            "'ANTHROPIC_API_KEY not set' is not, and the difference between those "
            "two readings is the whole offline-first claim. Never contains key "
            "material -- providers report the absence of a key, not its value."
        ),
    )
    runtime_s: float = 0.0


class ItemCorrection(BaseModel):
    """A Stage-3 claim that disagreed with the USDA database and was overwritten."""

    model_config = ConfigDict(extra="forbid")

    food_id: str
    name: str
    field: str = Field(..., description="e.g. 'quantity_g' or a nutrient name")
    claimed: float
    corrected: float
    relative_error: float
    note: str | None = None


class SafetyFix(BaseModel):
    """A hard-safety violation found after generation and repaired before output."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    food_id: str
    name: str
    action: Literal["reform", "remove", "substitute"]
    old_form: Form | None = None
    new_form: Form | None = None
    replacement_food_id: str | None = None
    replacement_name: str | None = None
    message: str = ""


class VerificationReport(BaseModel):
    """Stage-4 output. Its counts are the project's headline metric."""

    model_config = ConfigDict(extra="forbid")

    checked: int = 0
    matched: int = 0
    unmatched: list[str] = Field(default_factory=list)
    corrected: list[ItemCorrection] = Field(default_factory=list)
    flagged: list[Violation] = Field(default_factory=list)
    safety_fixes: list[SafetyFix] = Field(default_factory=list)
    verified_nutrients: NutrientVector = Field(default_factory=NutrientVector)
    final_pass: bool = False
    runtime_s: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_corrections(self) -> int:
        return len(self.corrected)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def had_issue(self) -> bool:
        """True when Stage 3 produced anything that needed correcting or fixing."""
        return bool(self.corrected or self.safety_fixes or self.unmatched)


# ---------------------------------------------------------------------------
# End-to-end trace
# ---------------------------------------------------------------------------


class PipelineTrace(BaseModel):
    """Everything the four stages did, in one serialisable object.

    This *is* the API response body: the frontend renders its stepper directly
    from these fields, and ``tests/test_worked_examples.py`` asserts against them.
    """

    model_config = ConfigDict(extra="forbid")

    profile: UserProfile
    planned_meal: Meal
    pantry: Meal
    scenario: str | None = None

    stage1: Stage1Result | None = None
    stage2: Stage2Result | None = None
    stage3: Stage3Result | None = None
    stage4: VerificationReport | None = None

    final_meal: Meal = Field(default_factory=Meal)
    final_rule_evaluation: RuleEvaluation | None = None

    seed: int = 42
    total_runtime_s: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def succeeded(self) -> bool:
        return self.stage4 is not None and self.stage4.final_pass
