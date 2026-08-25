"""Decision-variable construction -- where availability-awareness actually lives.

This is extension #1 over MetaPlate, and the implementation is the argument: the
search space is built from ``planned_meal + pantry`` and **nothing else**. An
unavailable food is not discouraged by a penalty term that a determined optimiser
could trade away against some other gain -- it has no variable, so no candidate
in the space can contain it. Availability violations are impossible by
construction rather than unlikely by tuning, which is exactly what the comparison
against the baselines is meant to show.

Each candidate food contributes two dimensions:

``quantity_g``
    Continuous, bounded above by what a person would plausibly eat of that food.
``form``
    An index into that food's ``allowed_forms``, decoded through the database so
    the optimiser can only ever propose a preparation the food can physically
    take. This is what lets a choking hazard be repaired by re-forming a food
    instead of deleting it.

Pantry items start at 0 g. Adding one therefore costs both L1 distance and a
sparsity increment, so a substitution only happens when it earns its place --
which is what makes the edit minimal rather than a from-scratch meal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foodsense.data.fdc import FoodDB, FoodRecord, get_food_db
from foodsense.schemas import Form, Meal, MealItem, UserProfile

__all__ = ["SearchSpace", "Variable", "build_space"]

#: Quantities are decoded to this resolution. Sub-gram precision is meaningless
#: for a recommendation a person has to act on, and rounding keeps "unchanged"
#: genuinely unchanged rather than 40.0000001 g.
QUANTITY_RESOLUTION_G = 1.0

#: Smallest amount that counts as being in the meal at all. Distinct from the
#: objective's ``quantity_epsilon_g``, which is the threshold for calling an
#: amount *changed*.
MIN_SERVING_G = 10.0


@dataclass(frozen=True, slots=True)
class Variable:
    """One food the optimiser may use, and the bounds it may use it within."""

    record: FoodRecord
    planned_quantity_g: float
    planned_form: Form
    max_quantity_g: float
    forms: tuple[Form, ...]
    from_pantry: bool
    #: Cost of choosing each form, parallel to ``forms``. Zero for the planned
    #: form, small for the config's declared nearest-safe-form, larger for any
    #: other change. This is what makes "nearest safe form" mean something to the
    #: optimiser: without it, whole grapes get repaired by pureeing them, which
    #: the rules accept but the AAP's map does not recommend.
    form_costs: tuple[float, ...] = ()

    @property
    def food_id(self) -> str:
        return self.record.fdc_id

    @property
    def name(self) -> str:
        return self.record.name

    @property
    def planned_form_index(self) -> int:
        try:
            return self.forms.index(self.planned_form)
        except ValueError:
            return 0


@dataclass(slots=True)
class SearchSpace:
    """The decision space for one recommendation, plus encode/decode."""

    variables: list[Variable]
    db: FoodDB
    #: Food ids the user actually has. Used only for *auditing* baselines that do
    #: not restrict their space; our own optimiser cannot leave it.
    available_ids: frozenset[str] = field(default_factory=frozenset)

    # -- shape --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.variables)

    @property
    def n_dims(self) -> int:
        """Two dimensions per food: quantity, then form index."""
        return 2 * len(self.variables)

    def bounds(self) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for variable in self.variables:
            out.append((0.0, variable.max_quantity_g))
            # Upper bound is len(forms) - eps so that flooring never overruns.
            out.append((0.0, max(len(variable.forms) - 1, 0) + 0.999))
        return out

    def integrality(self) -> np.ndarray:
        """Mask marking the form dimensions as integer-valued."""
        mask = np.zeros(self.n_dims, dtype=bool)
        mask[1::2] = True
        return mask

    # -- encode / decode ----------------------------------------------------

    def encode_planned(self) -> np.ndarray:
        """The planned meal as a point in the space -- the optimiser's origin."""
        x = np.zeros(self.n_dims, dtype=np.float64)
        for i, variable in enumerate(self.variables):
            x[2 * i] = variable.planned_quantity_g
            x[2 * i + 1] = float(variable.planned_form_index)
        return x

    def decode(self, x: np.ndarray, min_serving_g: float = MIN_SERVING_G) -> Meal:
        """Turn a decision vector into a meal, dropping anything below ``min_serving_g``.

        The floor matters more than it looks. Without it the optimiser discovers
        that a two-gram sliver of lentils nudges the fibre score at almost no
        distance cost, and returns a meal with three garnish-sized additions
        nobody would ever serve. A portion either exists or it does not.
        """
        items: list[MealItem] = []
        for i, variable in enumerate(self.variables):
            quantity = round(float(x[2 * i]) / QUANTITY_RESOLUTION_G) * QUANTITY_RESOLUTION_G
            if quantity < min_serving_g:
                continue
            # Plain min/max rather than np.clip: this runs once per item per
            # candidate, and numpy's scalar path costs more than the arithmetic.
            index = min(max(int(x[2 * i + 1]), 0), len(variable.forms) - 1)
            items.append(
                MealItem(
                    food_id=variable.food_id,
                    name=variable.name,
                    quantity_g=min(quantity, variable.max_quantity_g),
                    form=variable.forms[index],
                )
            )
        return Meal(items=items)

    def decode_many(
        self, population: np.ndarray, min_serving_g: float = MIN_SERVING_G
    ) -> list[Meal]:
        return [self.decode(row, min_serving_g) for row in np.atleast_2d(population)]

    # -- auditing -----------------------------------------------------------

    def unavailable_items(self, meal: Meal) -> list[MealItem]:
        """Items in ``meal`` the user does not actually have.

        Always empty for a space built by :func:`build_space`. It is non-empty
        only for the availability-blind baselines, and that is the point -- the
        headline table needs the violation to be *measured*, not asserted.
        """
        if not self.available_ids:
            return []
        return [i for i in meal.items if i.food_id not in self.available_ids]


def _max_quantity(
    record: FoodRecord,
    planned_quantity_g: float,
    from_pantry: bool,
    max_item_quantity_g: float,
    max_growth_factor: float,
) -> float:
    """How much of this food the optimiser may propose.

    A planned item may grow by a bounded factor -- doubling a portion is a
    plausible edit, quintupling it is not a recommendation anyone would follow.
    A pantry item has no planned amount to scale, so it gets the global cap.
    """
    if from_pantry or planned_quantity_g <= 0:
        return max_item_quantity_g
    return float(min(max(planned_quantity_g * max_growth_factor, 30.0), max_item_quantity_g))


#: Cost of moving to the config's declared nearest-safe-form, and of moving to
#: any other form. Both are multiplied by ``lambda_form_preference``, which is
#: small enough that this only ever breaks ties between equally safe options.
NEAREST_SAFE_FORM_COST = 0.3
OTHER_FORM_COST = 1.0


def _form_costs(
    record: FoodRecord, planned_form: Form, profile: UserProfile | None
) -> tuple[float, ...]:
    """Preference over this food's forms: planned, then nearest-safe, then the rest."""
    preferred: tuple[Form, ...] = ()
    if profile is not None and record.hazard_class:
        # Imported lazily: constraints imports the data layer, and the search
        # space is part of the optimiser rather than the constraint layer.
        from foodsense.constraints.age_rules import load_age_config

        ban = next(
            (
                b
                for b in load_age_config(profile.age_group).active_bans()
                if b.hazard_class == record.hazard_class
            ),
            None,
        )
        if ban is not None:
            preferred = ban.safe_forms_for(profile.age_months)

    costs = []
    for form in record.allowed_forms:
        if form == planned_form:
            costs.append(0.0)
        elif form in preferred:
            # Earlier entries in the map are nearer, so cost rises down the list.
            costs.append(NEAREST_SAFE_FORM_COST + 0.05 * preferred.index(form))
        else:
            costs.append(OTHER_FORM_COST)
    return tuple(costs)


def build_space(
    planned_meal: Meal,
    pantry: Meal | list[MealItem] | None = None,
    db: FoodDB | None = None,
    max_item_quantity_g: float = 400.0,
    max_growth_factor: float = 2.5,
    profile: UserProfile | None = None,
) -> SearchSpace:
    """Build the decision space from the planned meal and the pantry, and nothing else.

    Foods appearing in both are merged, keeping the planned quantity as the
    starting point: having spare rice in the cupboard does not mean the rice on
    the plate is a new addition.
    """
    db = db or get_food_db()
    pantry_items = pantry.items if isinstance(pantry, Meal) else list(pantry or [])

    variables: list[Variable] = []
    seen: dict[str, int] = {}

    for item in planned_meal.items:
        record = db.find(item.food_id)
        if record is None:
            # A planned food we have no nutrition for cannot be reasoned about;
            # dropping it silently would be worse than leaving it out of the
            # search, so the caller sees it missing from the trace.
            continue
        planned_form = item.form if record.permits(item.form) else record.default_form
        seen[record.fdc_id] = len(variables)
        variables.append(
            Variable(
                record=record,
                planned_quantity_g=item.quantity_g,
                planned_form=planned_form,
                max_quantity_g=_max_quantity(
                    record, item.quantity_g, False, max_item_quantity_g, max_growth_factor
                ),
                forms=record.allowed_forms,
                from_pantry=False,
                form_costs=_form_costs(record, planned_form, profile),
            )
        )

    for item in pantry_items:
        record = db.find(item.food_id)
        if record is None or record.fdc_id in seen:
            continue
        seen[record.fdc_id] = len(variables)
        variables.append(
            Variable(
                record=record,
                planned_quantity_g=0.0,
                planned_form=record.default_form,
                max_quantity_g=_max_quantity(
                    record, 0.0, True, max_item_quantity_g, max_growth_factor
                ),
                forms=record.allowed_forms,
                from_pantry=True,
                form_costs=_form_costs(record, record.default_form, profile),
            )
        )

    return SearchSpace(
        variables=variables,
        db=db,
        available_ids=frozenset(seen),
    )
