"""Life-stage rules: choking hazards, medication interactions, texture limits.

The central idea, and the reason ``Form`` is a first-class field on every meal
item: **a choking hazard is a property of the (food, preparation form) pair, not
of the food.** Whole grapes are unsafe for a toddler and quartered grapes are
not. Encoding the ban that way gives the optimiser a cheap repair -- change the
form -- instead of forcing it to delete the food, which is what makes the toddler
worked example come out the way the proposal describes.

Some hazards have no safe form at all (popcorn, marshmallow, hard candy, gum),
and some have one only above a certain age (whole nuts). Both cases are data in
``configs/age_groups/*.yaml``, not branches in code.

Medication and condition rules live in ``configs/health_flags.yaml`` because they
are age-independent: a 40-year-old on warfarin needs the same vitamin-K
consistency as an 80-year-old.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Any

import yaml

from foodsense import CONFIG_DIR
from foodsense.constraints.goals import Rule, Severity, Threshold
from foodsense.data.fdc import FoodDB
from foodsense.schemas import AgeGroup, Form, HealthFlag, Meal, MealItem, UserProfile

__all__ = [
    "AgeConfig",
    "ChokingBan",
    "FlagRule",
    "StructuralViolation",
    "check_choking",
    "check_excluded_tags",
    "check_texture",
    "load_age_config",
    "load_flag_rules",
    "nearest_safe_form",
    "permitted_forms",
]

AGE_DIR = CONFIG_DIR / "age_groups"
FLAGS_PATH = CONFIG_DIR / "health_flags.yaml"

#: Wildcard in a config's ``banned_forms`` meaning "every form of this food".
ANY_FORM = "*"


# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChokingBan:
    """One ``(hazard_class, banned forms)`` prohibition and its repair."""

    hazard_class: str
    banned_forms: frozenset[Form] | None  # None == every form
    #: Repairs in preference order. Empty means no preparation makes this food
    #: safe, so the only fix is removal or substitution. An ordered list rather
    #: than a single value because the AAP's map genuinely offers alternatives --
    #: meat is "minced or ground", nuts are "ground or thinly spread" -- and which
    #: one applies depends on what the specific food can physically take.
    safe_forms: tuple[Form, ...]
    message: str
    safe_form_min_age_months: int | None = None
    enabled: bool = True

    @property
    def rule_id(self) -> str:
        return f"toddler.choking.{self.hazard_class}"

    def bans(self, form: Form) -> bool:
        return self.banned_forms is None or form in self.banned_forms

    def safe_forms_for(self, age_months: int | None) -> tuple[Form, ...]:
        """The forms this hazard can be repaired to, in preference order, at this age.

        A ban may carry ``safe_form_min_age_months``: below that age there is no
        safe preparation and the only repair is removal or substitution. Whole
        nuts are the case that matters -- grinding is an acceptable answer for a
        three-year-old and not for an eighteen-month-old.
        """
        if not self.safe_forms or self.safe_form_min_age_months is None:
            return self.safe_forms if self.safe_form_min_age_months is None else ()
        if age_months is None:
            # Unknown age inside a life stage with an age-gated rule: assume the
            # youngest, because the failure mode of guessing wrong is a choking
            # hazard rather than an inconvenience.
            return ()
        return self.safe_forms if age_months >= self.safe_form_min_age_months else ()

    def repair_for(self, allowed_forms: tuple[Form, ...], age_months: int | None) -> Form | None:
        """First safe form this particular food can actually take, or ``None``."""
        return next((f for f in self.safe_forms_for(age_months) if f in allowed_forms), None)


@dataclass(frozen=True, slots=True)
class AgeConfig:
    """One file from ``configs/age_groups/``."""

    age_group: AgeGroup
    label: str
    description: str
    meals_per_day: float
    per_meal_floor_attainment: float
    default_weight_kg: float
    daily: dict[str, Threshold]
    per_meal: dict[str, Threshold]
    protein_g_per_kg: Threshold | None
    choking_bans: tuple[ChokingBan, ...]
    texture_notes: dict[str, str] = field(default_factory=dict)
    strict_no_added_sugar_below_months: int | None = None
    honey_min_age_months: int | None = None

    def rules(self, profile: UserProfile) -> list[Rule]:
        """Numeric per-meal rules for this life stage and profile.

        Daily reference intakes are divided by ``meals_per_day``; per-kg targets
        use the profile's body weight when it supplies one.
        """
        source = f"configs/age_groups/{self.age_group.value}.yaml"
        share = 1.0 / self.meals_per_day
        rules = [
            Rule(
                rule_id=f"age.{self.age_group.value}.{quantity}",
                quantity=quantity,
                threshold=threshold.scaled(share, self.per_meal_floor_attainment),
                severity="soft",
                message=(
                    f"{quantity.replace('_', ' ')} outside the per-meal target for "
                    f"{self.label.lower()}"
                ),
                source=source,
            )
            for quantity, threshold in self.daily.items()
        ]
        rules += [
            Rule(
                rule_id=f"age.{self.age_group.value}.per_meal.{quantity}",
                quantity=quantity,
                threshold=threshold,
                severity="soft",
                message=f"{quantity.replace('_', ' ')} outside the per-meal target",
                source=source,
            )
            for quantity, threshold in self.per_meal.items()
        ]

        if self.protein_g_per_kg is not None:
            weight_kg = profile.weight_kg or self.default_weight_kg
            rules.append(
                Rule(
                    rule_id=f"age.{self.age_group.value}.protein_per_kg",
                    quantity="protein_g",
                    threshold=self.protein_g_per_kg.scaled(
                        weight_kg * share, self.per_meal_floor_attainment
                    ),
                    severity="soft",
                    message=(
                        f"protein outside the per-meal target for {weight_kg:g} kg at this "
                        f"life stage"
                    ),
                    source=source,
                )
            )
        return rules

    def active_bans(self) -> tuple[ChokingBan, ...]:
        return tuple(b for b in self.choking_bans if b.enabled)


@dataclass(frozen=True, slots=True)
class FlagRule:
    """One entry from ``configs/health_flags.yaml``."""

    flag: HealthFlag
    label: str
    severity: Severity
    per_meal_max: dict[str, float] = field(default_factory=dict)
    per_meal_min: dict[str, float] = field(default_factory=dict)
    exclude_tags: frozenset[str] = frozenset()
    allowed_forms: frozenset[Form] | None = None
    watch_tags: frozenset[str] = frozenset()
    weight: float = 1.0
    message: str = ""

    def rules(self) -> list[Rule]:
        """The numeric half of this flag's constraints."""
        source = "configs/health_flags.yaml"
        out = [
            Rule(
                rule_id=f"flag.{self.flag.value}.{quantity}",
                quantity=quantity,
                threshold=Threshold(maximum=limit, weight=self.weight),
                severity=self.severity,
                message=self.message
                or f"{quantity.replace('_', ' ')} above the limit for {self.label}",
                source=source,
            )
            for quantity, limit in self.per_meal_max.items()
        ]
        out += [
            Rule(
                rule_id=f"flag.{self.flag.value}.{quantity}",
                quantity=quantity,
                threshold=Threshold(minimum=limit, weight=self.weight),
                severity=self.severity,
                message=self.message
                or f"{quantity.replace('_', ' ')} below the target for {self.label}",
                source=source,
            )
            for quantity, limit in self.per_meal_min.items()
        ]
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _threshold_map(raw: dict[str, Any] | None) -> dict[str, Threshold]:
    return {k: Threshold.from_config(v) for k, v in (raw or {}).items()}


def _parse_forms(raw: Any) -> frozenset[Form] | None:
    if raw is None:
        return None
    values = list(raw)
    if ANY_FORM in values:
        return None
    return frozenset(Form(v) for v in values)


@cache
def load_age_config(age_group: AgeGroup) -> AgeConfig:
    """Load and cache ``configs/age_groups/<age_group>.yaml``."""
    path = AGE_DIR / f"{AgeGroup(age_group).value}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No age-group configuration at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    bans = []
    for entry in raw.get("choking_bans") or []:
        # Accepts `safe_forms: [minced, ground]`, or a scalar `safe_form: minced`,
        # or an explicit null meaning no preparation makes this food safe.
        raw_safe = entry.get("safe_forms", entry.get("safe_form"))
        if raw_safe is None:
            safe_forms: tuple[Form, ...] = ()
        elif isinstance(raw_safe, str):
            safe_forms = (Form(raw_safe),)
        else:
            safe_forms = tuple(Form(f) for f in raw_safe)
        bans.append(
            ChokingBan(
                hazard_class=entry["hazard_class"],
                banned_forms=_parse_forms(entry.get("banned_forms")),
                safe_forms=safe_forms,
                message=(entry.get("message") or "").strip(),
                safe_form_min_age_months=entry.get("safe_form_min_age_months"),
                enabled=bool(entry.get("enabled", True)),
            )
        )

    protein_per_kg = raw.get("protein_g_per_kg")
    return AgeConfig(
        age_group=AgeGroup(raw["age_group"]),
        label=raw.get("label", raw["age_group"]),
        description=(raw.get("description") or "").strip(),
        meals_per_day=float(raw.get("meals_per_day", 3.0)),
        per_meal_floor_attainment=float(raw.get("per_meal_floor_attainment", 1.0)),
        default_weight_kg=float(raw.get("default_weight_kg", 70.0)),
        daily=_threshold_map(raw.get("daily")),
        per_meal=_threshold_map(raw.get("per_meal")),
        protein_g_per_kg=Threshold.from_config(protein_per_kg) if protein_per_kg else None,
        choking_bans=tuple(bans),
        texture_notes=dict(raw.get("texture_notes") or {}),
        strict_no_added_sugar_below_months=raw.get("strict_no_added_sugar_below_months"),
        honey_min_age_months=raw.get("honey_min_age_months"),
    )


@lru_cache(maxsize=1)
def load_flag_rules() -> dict[HealthFlag, FlagRule]:
    """Load and cache ``configs/health_flags.yaml``."""
    if not FLAGS_PATH.exists():
        raise FileNotFoundError(f"No health-flag configuration at {FLAGS_PATH}")
    raw = yaml.safe_load(FLAGS_PATH.read_text(encoding="utf-8"))

    out: dict[HealthFlag, FlagRule] = {}
    for name, entry in (raw.get("rules") or {}).items():
        flag = HealthFlag(name)
        out[flag] = FlagRule(
            flag=flag,
            label=entry.get("label", name),
            severity=entry.get("severity", "soft"),
            per_meal_max=dict(entry.get("per_meal_max") or {}),
            per_meal_min=dict(entry.get("per_meal_min") or {}),
            exclude_tags=frozenset(entry.get("exclude_tags") or ()),
            allowed_forms=(
                frozenset(Form(f) for f in entry["allowed_forms"])
                if entry.get("allowed_forms")
                else None
            ),
            watch_tags=frozenset(entry.get("watch_tags") or ()),
            weight=float(entry.get("weight", 1.0)),
            message=(entry.get("message") or "").strip(),
        )
    return out


# ---------------------------------------------------------------------------
# Structural (non-numeric) checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructuralViolation:
    """A hard-safety breach that is about *what* is in the meal, not how much."""

    rule_id: str
    item: MealItem
    message: str
    suggested_form: Form | None = None
    removable_only: bool = False
    source: str = ""


def check_choking(
    meal: Meal | list[MealItem], profile: UserProfile, db: FoodDB
) -> list[StructuralViolation]:
    """Every ``(hazard_class, form)`` ban this meal breaks for this profile."""
    config = load_age_config(profile.age_group)
    bans = config.active_bans()
    if not bans:
        return []
    by_hazard = {ban.hazard_class: ban for ban in bans}

    items = meal.items if isinstance(meal, Meal) else meal
    violations: list[StructuralViolation] = []
    for item in items:
        record = db.find(item.food_id)
        if record is None or not record.hazard_class:
            continue
        ban = by_hazard.get(record.hazard_class)
        if ban is None or not ban.bans(item.form):
            continue
        # A repair is only real if the food can physically take that form.
        safe_form = ban.repair_for(record.allowed_forms, profile.age_months)
        violations.append(
            StructuralViolation(
                rule_id=ban.rule_id,
                item=item,
                message=ban.message,
                suggested_form=safe_form,
                removable_only=safe_form is None,
                source=f"configs/age_groups/{config.age_group.value}.yaml",
            )
        )
    return violations


def check_excluded_tags(
    meal: Meal | list[MealItem], profile: UserProfile, db: FoodDB
) -> list[StructuralViolation]:
    """Foods a medication rule forbids outright (grapefruit on a statin, and so on)."""
    flag_rules = load_flag_rules()
    active = [flag_rules[f] for f in profile.health_flags if f in flag_rules]
    excluding = [r for r in active if r.exclude_tags]
    if not excluding:
        return []

    items = meal.items if isinstance(meal, Meal) else meal
    violations: list[StructuralViolation] = []
    for item in items:
        record = db.find(item.food_id)
        if record is None:
            continue
        for rule in excluding:
            hit = rule.exclude_tags & record.tags
            if hit:
                violations.append(
                    StructuralViolation(
                        rule_id=f"flag.{rule.flag.value}.excluded_food",
                        item=item,
                        message=f"{record.name}: {rule.message}",
                        suggested_form=None,
                        removable_only=True,
                        source="configs/health_flags.yaml",
                    )
                )
                break
    return violations


def permitted_forms(profile: UserProfile) -> frozenset[Form] | None:
    """Forms this profile may eat at all, or ``None`` when unrestricted.

    Currently driven by the dysphagia flag (IDDSI-style texture modification).
    """
    flag_rules = load_flag_rules()
    allowed: frozenset[Form] | None = None
    for flag in profile.health_flags:
        rule = flag_rules.get(flag)
        if rule is not None and rule.allowed_forms is not None:
            allowed = rule.allowed_forms if allowed is None else (allowed & rule.allowed_forms)
    return allowed


def check_texture(
    meal: Meal | list[MealItem], profile: UserProfile, db: FoodDB
) -> list[StructuralViolation]:
    """Items whose form is outside what this profile can safely swallow."""
    allowed = permitted_forms(profile)
    if allowed is None:
        return []
    flag_rules = load_flag_rules()
    message = next(
        (
            flag_rules[f].message
            for f in profile.health_flags
            if f in flag_rules and flag_rules[f].allowed_forms is not None
        ),
        "Texture is unsafe for this profile.",
    )

    items = meal.items if isinstance(meal, Meal) else meal
    violations: list[StructuralViolation] = []
    for item in items:
        if item.form in allowed:
            continue
        record = db.find(item.food_id)
        candidates = [f for f in (record.allowed_forms if record else ()) if f in allowed]
        violations.append(
            StructuralViolation(
                rule_id="flag.dysphagia.texture",
                item=item,
                message=message,
                suggested_form=candidates[0] if candidates else None,
                removable_only=not candidates,
                source="configs/health_flags.yaml",
            )
        )
    return violations


def nearest_safe_form(item: MealItem, profile: UserProfile, db: FoodDB) -> Form | None:
    """The closest form that makes ``item`` safe for ``profile``, or ``None``.

    Used by Stage 4 to repair a violation it finds after generation. Returns
    ``None`` when the food cannot be made safe at all -- the caller must then
    remove or substitute it.
    """
    record = db.find(item.food_id)
    if record is None:
        return None

    allowed = permitted_forms(profile)
    candidates = [f for f in record.allowed_forms if allowed is None or f in allowed]
    if not candidates:
        return None

    config = load_age_config(profile.age_group)
    bans = {b.hazard_class: b for b in config.active_bans()}
    ban = bans.get(record.hazard_class)

    if ban is not None:
        preferred = ban.repair_for(tuple(candidates), profile.age_months)
        if preferred is not None:
            return preferred
        candidates = [f for f in candidates if not ban.bans(f)]

    if not candidates:
        return None
    if item.form in candidates:
        return item.form
    return candidates[0]


def check_age_gated_foods(
    meal: Meal | list[MealItem], profile: UserProfile, db: FoodDB
) -> list[StructuralViolation]:
    """Foods barred by age alone rather than by preparation.

    Currently honey, which carries infant-botulism risk below 12 months (AAP/CDC).
    Unlike a choking hazard there is no safe form -- the food simply must not be
    served yet.
    """
    config = load_age_config(profile.age_group)
    minimum = config.honey_min_age_months
    if minimum is None or profile.age_months is None or profile.age_months >= minimum:
        return []

    items = meal.items if isinstance(meal, Meal) else meal
    violations: list[StructuralViolation] = []
    for item in items:
        record = db.find(item.food_id)
        if record is not None and "honey" in record.tags:
            violations.append(
                StructuralViolation(
                    rule_id=f"age.{config.age_group.value}.honey",
                    item=item,
                    message=(
                        f"Honey must not be given below {minimum} months (infant botulism risk)."
                    ),
                    suggested_form=None,
                    removable_only=True,
                    source=f"configs/age_groups/{config.age_group.value}.yaml",
                )
            )
    return violations


def implicit_flags(profile: UserProfile) -> list[HealthFlag]:
    """Flags a profile carries by virtue of its age rather than by being told.

    AAP advises no added sugars below 24 months, so a toddler under that age gets
    the strict rule whether or not the caller thought to set it.
    """
    config = load_age_config(profile.age_group)
    out: list[HealthFlag] = []
    cutoff = config.strict_no_added_sugar_below_months
    if (
        cutoff is not None
        and profile.age_months is not None
        and profile.age_months < cutoff
        and HealthFlag.STRICT_NO_ADDED_SUGAR not in profile.health_flags
    ):
        out.append(HealthFlag.STRICT_NO_ADDED_SUGAR)
    return out
