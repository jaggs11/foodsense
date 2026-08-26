"""The three demo scenarios from the proposal, as data.

Shared by the CLI, the API and the verification experiment, so that the meal a
faculty demo shows is the same object the tests assert against and the same one
the evaluation measures. Defining them three times would guarantee they drift.

Foods are pinned by ``fdc_id`` rather than looked up by name at load time. The
database is committed, the ids are USDA's own and stable, and a scenario that
silently re-resolved to a different food after a matcher change would be a
demo that quietly stopped demonstrating the thing it was built for. If an id
ever disappears from the curated database, :func:`load_scenario` fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from foodsense.data.fdc import FoodDB, get_food_db
from foodsense.schemas import AgeGroup, Form, Goal, HealthFlag, Meal, MealItem, UserProfile

__all__ = ["SCENARIOS", "Scenario", "ScenarioItem", "load_scenario", "scenario_names"]


@dataclass(frozen=True, slots=True)
class ScenarioItem:
    """One pinned food. ``name`` is documentation; ``fdc_id`` is the contract."""

    fdc_id: str
    name: str
    quantity_g: float
    form: Form | None = None


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named demo case: who is eating, what they planned, what they have."""

    key: str
    title: str
    description: str
    profile: UserProfile
    planned: tuple[ScenarioItem, ...]
    pantry: tuple[ScenarioItem, ...]
    expectation: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def planned_meal(self, db: FoodDB | None = None) -> Meal:
        return _build(self.planned, db)

    def pantry_meal(self, db: FoodDB | None = None) -> Meal:
        return _build(self.pantry, db)


def _build(items: tuple[ScenarioItem, ...], db: FoodDB | None) -> Meal:
    db = db or get_food_db()
    built: list[MealItem] = []
    for item in items:
        record = db.find(item.fdc_id)
        if record is None:
            raise KeyError(
                f"Scenario food {item.fdc_id} ({item.name!r}) is not in the curated "
                f"database. Rebuild it with `make data`, or update the scenario."
            )
        form = item.form if item.form and record.permits(item.form) else record.default_form
        built.append(record.as_item(item.quantity_g, form))
    return Meal(items=built)


# ---------------------------------------------------------------------------
# Scenario 1 -- the toddler choking case
# ---------------------------------------------------------------------------

TODDLER_CHOKING = Scenario(
    key="toddler_choking",
    title="Toddler, 18 months: whole grapes and whole peanuts",
    description=(
        "An 18-month-old's lunch containing two named AAP choking hazards, with a "
        "pantry that has a safe alternative for each."
    ),
    profile=UserProfile(
        age_group=AgeGroup.TODDLER,
        age_months=18,
        weight_kg=11.0,
        goal=Goal.BALANCED_NUTRITION,
        health_flags=[HealthFlag.IRON_FOCUS],
    ),
    planned=(
        ScenarioItem("173040", "Grapes, muscadine, raw", 40.0, Form.WHOLE),
        ScenarioItem("173806", "Peanuts, all types, dry-roasted, without salt", 20.0, Form.WHOLE),
        ScenarioItem("168878", "Rice, white, long-grain, regular, enriched, cooked", 50.0),
    ),
    pantry=(
        ScenarioItem("169339", "Carrots, cooked, boiled, drained, with salt", 0.0),
        ScenarioItem("171117", "Chicken, ground, crumbles, cooked, pan-browned", 0.0),
        ScenarioItem("170886", "Yogurt, plain, low fat", 0.0),
        ScenarioItem("175237", "Beans, black, mature seeds, cooked, boiled, with salt", 0.0),
        ScenarioItem("169704", "Rice, brown, long-grain, cooked", 0.0),
    ),
    expectation=(
        "Grapes are re-formed to quartered rather than removed. Whole peanuts have no "
        "safe form at 18 months, so they are substituted from the pantry. The final meal "
        "contains no choking hazard and every quantity is USDA-verified."
    ),
    notes=(
        "Grapes and peanuts are both on the AAP choking-hazard list, but they resolve "
        "differently: a grape has a safe form and a peanut does not until 24 months. "
        "That asymmetry is the point of the scenario.",
    ),
)


# ---------------------------------------------------------------------------
# Scenario 2 -- the older adult sodium case
# ---------------------------------------------------------------------------

ELDERLY_SODIUM = Scenario(
    key="elderly_sodium",
    title="Older adult, 78, hypertension: canned soup and salted crackers",
    description=(
        "A high-sodium lunch for someone managing blood pressure, with low-sodium "
        "alternatives already in the cupboard."
    ),
    profile=UserProfile(
        age_group=AgeGroup.OLDER_ADULT,
        age_months=78 * 12,
        weight_kg=70.0,
        goal=Goal.BALANCED_NUTRITION,
        health_flags=[HealthFlag.HYPERTENSION],
    ),
    planned=(
        # 300 ml of condensed tomato soup, taken as 300 g.
        ScenarioItem("172882", "Soup, tomato, canned, condensed", 300.0),
        ScenarioItem("172746", "Crackers, saltines (includes oyster, soda, soup)", 30.0),
    ),
    pantry=(
        ScenarioItem("171609", "Soup, chicken broth, low sodium, canned", 0.0),
        ScenarioItem(
            "171477", "Chicken, broilers or fryers, breast, meat only, cooked, roasted", 0.0
        ),
        ScenarioItem("169339", "Carrots, cooked, boiled, drained, with salt", 0.0),
        ScenarioItem("172749", "Crackers, whole-wheat", 0.0),
        ScenarioItem("169704", "Rice, brown, long-grain, cooked", 0.0),
    ),
    expectation=(
        "Sodium is flagged against the 500 mg per-meal ceiling the hypertension flag "
        "imposes, and the meal is edited toward the low-sodium broth, chicken and "
        "carrots already in the pantry."
    ),
    notes=(
        "The planned meal is roughly 1,500 mg of sodium against a 500 mg per-meal "
        "ceiling, so this scenario exercises a soft-rule breach rather than a hazard.",
    ),
)


# ---------------------------------------------------------------------------
# Scenario 3 -- the adult weight-management case
# ---------------------------------------------------------------------------

ADULT_WEIGHT = Scenario(
    key="adult_weight",
    title="Adult, weight management: burger, fries and a cola",
    description=(
        "A third goal profile, to show that the objective is genuinely configurable "
        "rather than hard-coded around safety."
    ),
    profile=UserProfile(
        age_group=AgeGroup.ADULT,
        age_months=34 * 12,
        weight_kg=82.0,
        goal=Goal.WEIGHT_MANAGEMENT,
    ),
    planned=(
        ScenarioItem("170693", "Fast foods, hamburger; single, regular patty; plain", 110.0),
        ScenarioItem("168444", "Potatoes, french fried, steak fries", 120.0),
        ScenarioItem("174852", "Beverages, carbonated, cola, regular", 330.0),
    ),
    pantry=(
        ScenarioItem(
            "171477", "Chicken, broilers or fryers, breast, meat only, cooked, roasted", 0.0
        ),
        ScenarioItem("2346389", "Lettuce, romaine, green, raw", 0.0),
        ScenarioItem("169704", "Rice, brown, long-grain, cooked", 0.0),
        ScenarioItem("175237", "Beans, black, mature seeds, cooked, boiled, with salt", 0.0),
        ScenarioItem("2647437", "Yogurt, plain, nonfat", 0.0),
        ScenarioItem("321900", "Broccoli, raw", 0.0),
    ),
    expectation=(
        "No safety rule fires at all. The edit is driven purely by the weight-management "
        "goal -- a 550 kcal per-meal ceiling with a 25 g protein floor and an 8 g fibre "
        "floor -- which is what makes the generalised-goal objective visible."
    ),
    notes=(
        "Deliberately hazard-free. If every scenario turned on a safety rule, a reader "
        "could reasonably conclude the goal layer does nothing.",
    ),
)


SCENARIOS: dict[str, Scenario] = {s.key: s for s in (TODDLER_CHOKING, ELDERLY_SODIUM, ADULT_WEIGHT)}


def scenario_names() -> list[str]:
    return list(SCENARIOS)


def load_scenario(key: str) -> Scenario:
    try:
        return SCENARIOS[key]
    except KeyError:
        raise KeyError(f"Unknown scenario {key!r}. Available: {', '.join(SCENARIOS)}") from None
