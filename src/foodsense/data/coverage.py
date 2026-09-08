"""Nutrient coverage: which values a source actually reports, and the zero-fill bridge.

## The problem this exists for

``build_food_db.py`` used to end its nutrient handling with ``.fillna(0.0)``. Every
nutrient USDA does not report for a food was stored as a measured zero, and nothing
downstream could tell the two apart. That held for the project's entire life, sits
underneath Stage 1's training labels and Stage 4's verification claims, and is
squarely what ruling R8 forbids.

The database no longer does this: unreported nutrients are ``NULL`` (P1.2), and
:class:`NutrientCoverage` reports what is genuinely present.

## Why the pipeline still zero-fills, for now

Fixing the database and fixing every consumer in one step would move every number
the project has published, which collides with P1.7's requirement that the golden
traces pass unmodified -- and would make it impossible to tell a migration bug from
an intended change.

So ruling R13 splits it:

* **Now (P1)** -- the database is honest. Consumers keep their historical behaviour
  through :func:`zero_filled_vector`, which is *named*, documented, and the only
  place the substitution happens. Golden traces and demo outputs stay byte-identical.
* **P3** -- consumers switch to honest missingness, when the surrogate is retrained
  toddler-only and condition-conditioned and every result is being regenerated
  anyway.

The point of the split is auditability. The database stops lying immediately; the
moment the pipeline's *answers* change becomes one dated, deliberate commit rather
than a side effect of a schema migration.

**This module is scheduled for removal.** When P3 lands, :func:`zero_filled_vector`
goes and its callers read coverage directly. Anything added here in the meantime
should be written expecting to be deleted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from foodsense.schemas import NutrientVector

__all__ = [
    "REQUIRED_FOR_CONDITION",
    "NutrientCoverage",
    "is_missing",
    "unreported_among",
    "zero_filled_vector",
]

#: Nutrients each condition's rules depend on, so Stage 4 can report a rule as
#: ``undetermined`` when the matched food's source does not carry one of them.
#:
#: Empty until P3 writes ``configs/conditions/``. Empty is correct rather than
#: provisional: inventing entries now would mean guessing which nutrients a
#: condition's rules will cite before the evidence behind those rules exists, and
#: ruling R3 makes that evidence the precondition for the condition existing at all.
REQUIRED_FOR_CONDITION: Mapping[str, frozenset[str]] = {}


def is_missing(value: object) -> bool:
    """Whether a raw database value means "this source does not report it".

    A null reaches us in one of three shapes depending on whether the row came from
    sqlite, from parquet, or through pandas' nullable dtypes: ``None``,
    ``pandas.NA``, or float ``nan``. All three mean the same thing here.

    ``pandas.isna`` covers all three, but returns an *array* for array-like input,
    so it is only consulted for scalars it can answer about.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return value != value  # nan is the only value unequal to itself
    if isinstance(value, (int, bool)):
        return False
    result = pd.isna(value)
    return bool(result) if isinstance(result, bool | np.bool_) else False


def zero_filled_vector(
    values: Mapping[str, float | None], reported: Iterable[str]
) -> NutrientVector:
    """Build a dense nutrient vector, substituting 0.0 for unreported nutrients.

    **This preserves pre-repositioning behaviour and is retired in P3.**

    Stage 1 reads all 33 nutrients as a dense array for every candidate in every
    generation of the optimiser's search, and cannot branch per element without
    destroying that loop. So a value has to be there. Until P3 that value is 0.0,
    which is exactly what the database used to store.

    The substitution is only defensible because the caller also keeps the set of
    genuinely reported nutrients alongside the vector -- see
    :attr:`FoodRecord.reported_nutrients`. A nutrient absent from that set is
    **unknown**, and any rule, comparison or verification claim that depends on it
    must say so rather than reading the zero. Summing a vector is fine; asserting
    that a food contains no iron because ``iron_mg == 0.0`` is not.

    Do not re-implement this substitution anywhere. One named function is what makes
    the P3 change a single edit and what makes the current behaviour greppable.
    """
    reported_set = set(reported)
    return NutrientVector(
        **{
            key: (float(values[key]) if key in reported_set and values[key] is not None else 0.0)
            for key in NutrientVector.model_fields
        }
    )


def unreported_among(reported: Iterable[str], needed: Iterable[str]) -> set[str]:
    """Which of ``needed`` is not in ``reported``. The caller's "I cannot answer" set."""
    have = set(reported)
    return {n for n in needed if n not in have}


class NutrientCoverage:
    """What a food's source actually reports.

    Wraps the mask so callers ask a question rather than poking at a set, and so
    the vocabulary is the same everywhere: *reported* means the source states a
    value (including a genuine zero); *unreported* means unknown.
    """

    __slots__ = ("_reported", "food_id")

    def __init__(self, food_id: str, reported: Iterable[str]) -> None:
        self.food_id = food_id
        self._reported = frozenset(reported)

    @property
    def reported(self) -> frozenset[str]:
        return self._reported

    def __contains__(self, nutrient: object) -> bool:
        return nutrient in self._reported

    def __len__(self) -> int:
        return len(self._reported)

    def reports(self, nutrient: str) -> bool:
        return nutrient in self._reported

    def missing(self, needed: Iterable[str]) -> set[str]:
        return unreported_among(self._reported, needed)

    def covers(self, needed: Iterable[str]) -> bool:
        """Whether every nutrient in ``needed`` is reported.

        The question a condition asks before a food is allowed into its candidate
        pool: recommending a food *for* iron-deficiency anemia on the strength of an
        iron value its source never stated is the failure this prevents.
        """
        return not self.missing(needed)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NutrientCoverage({self.food_id!r}, {len(self._reported)} reported)"
