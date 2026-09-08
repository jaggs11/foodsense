"""``FoodRepository`` -- the one way into the food knowledge base.

Before the repositioning every reader went straight to :class:`FoodDB`, which was
fine while there was one source, the table was read-only, and "all foods" and "all
foods you may recommend" were the same set.

None of those hold now. There are several sources of differing confidence; P6 makes
the table writable from the API; and ``verification_status`` means a row can exist,
be searchable by an admin, and still be ineligible for recommendation. Every one of
those distinctions has to be applied consistently by six different callers, and the
failure mode of getting it wrong is not a crash -- it is a pending, unverified,
user-submitted row quietly appearing in a toddler's recommendations.

So the filtering lives here, once, and the callers ask for what they want.

## What this is not

It is not an abstraction over the storage engine, and it does not try to hide
:class:`FoodDB`. ``FoodDB`` remains the in-memory index and the fuzzy matcher, and
this wraps it. Stage 4 still matches with ``FoodDB``'s scorer and Stage 3 still
ranks with BM25, kept deliberately different so verification does not grade the
retriever against the retriever's own notion of similarity.

## Default visibility

:meth:`search` and :meth:`records` return **active, recommendable** foods by
default: ``is_active`` and a ``verification_status`` in
:data:`RECOMMENDABLE_STATUSES`. Callers that genuinely want everything -- an admin
moderation list, a duplicate check that must see pending rows, a coverage report --
pass ``include_unverified=True`` and say so at the call site.

Today this filter is a no-op: every row is migrated USDA, active and verified. It
is written now, with tests, because the first row it will exclude is a user
submission in P6, and a filter added at the moment it first matters is a filter
nobody has ever seen work.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from foodsense.data.coverage import NutrientCoverage
from foodsense.data.fdc import DEFAULT_MATCH_THRESHOLD, FoodDB, FoodRecord, get_food_db
from foodsense.schemas import Meal, MealItem, NutrientVector

__all__ = ["RECOMMENDABLE_STATUSES", "FoodFilters", "FoodRepository", "get_food_repository"]

#: Verification states a food may be recommended from. ``pending`` and ``rejected``
#: are excluded: a row awaiting review has not been reviewed, and one that failed
#: review failed it. ``unverified`` covers data whose provenance could not be
#: established (ruling R5) and is excluded for the same reason.
RECOMMENDABLE_STATUSES = frozenset({"verified"})


@dataclass(frozen=True, slots=True)
class FoodFilters:
    """Metadata filters. ``None`` means "do not filter on this".

    Deliberately not a free-form dict: a mistyped key in a dict silently widens the
    result set, and widening is the direction that hurts here.
    """

    cuisine: str | Sequence[str] | None = None
    category: str | Sequence[str] | None = None
    veg_status: str | Sequence[str] | None = None
    source_type: str | Sequence[str] | None = None
    #: Every one of these nutrients must be *reported* by the food's source -- not
    #: merely non-zero. Recommending a food for iron-deficiency anemia on the
    #: strength of an iron value nobody measured is the failure this prevents.
    requires_nutrients: Sequence[str] | None = None

    def _matches(self, record: FoodRecord) -> bool:
        for value, attr in (
            (self.cuisine, "cuisine"),
            (self.category, "category"),
            (self.veg_status, "veg_status"),
            (self.source_type, "source_type"),
        ):
            if value is None:
                continue
            allowed = {value} if isinstance(value, str) else set(value)
            if getattr(record, attr, None) not in allowed:
                return False
        if self.requires_nutrients and not all(
            record.reports(n) for n in self.requires_nutrients
        ):
            return False
        return True

    @property
    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in ("cuisine", "category", "veg_status", "source_type", "requires_nutrients")
        )


class FoodRepository:
    """Read access to the food knowledge base, with visibility applied once."""

    def __init__(self, db: FoodDB | None = None) -> None:
        self._db = db if db is not None else get_food_db()

    @property
    def db(self) -> FoodDB:
        """The underlying index.

        Exposed rather than hidden: Stage 4's matcher and Stage 3's BM25 index are
        built on it, and pretending otherwise would mean re-exporting its whole
        surface. Reach for this only when you need the matcher or the index.
        """
        return self._db

    # -- visibility ---------------------------------------------------------

    @staticmethod
    def _visible(record: FoodRecord, include_unverified: bool) -> bool:
        if include_unverified:
            return True
        return record.verification_status in RECOMMENDABLE_STATUSES

    def records(
        self,
        *,
        filters: FoodFilters | None = None,
        include_unverified: bool = False,
    ) -> list[FoodRecord]:
        """Every visible food, optionally filtered."""
        out = [r for r in self._db.records if self._visible(r, include_unverified)]
        if filters is not None and not filters.is_empty:
            out = [r for r in out if filters._matches(r)]
        return out

    # -- lookup -------------------------------------------------------------

    def get(self, food_id: str) -> FoodRecord | None:
        """One food by id, or ``None``.

        Not visibility-filtered. A trace that named a food must still be able to
        resolve it after the food is rejected, or the trace becomes unreadable.
        """
        return self._db.find(food_id)

    def exists(self, food_id: str) -> bool:
        return self._db.find(food_id) is not None

    def by_source(self, source_type: str, *, include_unverified: bool = False) -> list[FoodRecord]:
        """Every food from one source. The coverage report's entry point."""
        return self.records(
            filters=FoodFilters(source_type=source_type),
            include_unverified=include_unverified,
        )

    # -- search -------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: FoodFilters | None = None,
        include_unverified: bool = False,
    ) -> list[tuple[FoodRecord, float]]:
        """Fuzzy name search, best first.

        Filtering happens *after* scoring and before truncation, so a filter never
        changes the relative order of what survives it -- only which rows do. The
        underlying scorer is untouched, which is what keeps this equivalent to the
        pre-refactor behaviour when nothing is filtered.
        """
        hits = self._db.search(query, limit=limit if self._unfiltered(filters) else limit * 4)
        out = [
            (record, score)
            for record, score in hits
            if self._visible(record, include_unverified)
            and (filters is None or filters.is_empty or filters._matches(record))
        ]
        return out[:limit]

    @staticmethod
    def _unfiltered(filters: FoodFilters | None) -> bool:
        return filters is None or filters.is_empty

    def match(
        self,
        name: str,
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> tuple[FoodRecord | None, float]:
        """Stage 4's grounding match. Delegates to the scorer Stage 4 has always used.

        Deliberately not visibility-filtered: Stage 4 asks "is this generated name a
        real food we hold", and a pending row is still a real food we hold. Whether
        it may be *recommended* is a separate question, asked where recommendation
        happens.
        """
        return self._db.match(name, threshold=threshold)

    # -- nutrients ----------------------------------------------------------

    def nutrients_for(self, meal: Meal | list[MealItem]) -> NutrientVector:
        """Ground-truth nutrients for a meal, recomputed from stored rows."""
        return self._db.nutrients_for(meal)

    def nutrient_matrix(self, food_ids: Iterable[str]) -> np.ndarray:
        """``(n_foods, 33)`` per-100 g matrix, row-aligned to ``food_ids``.

        Stage 1's feature builder reads this. Unknown ids contribute a zero row,
        which is the pre-existing behaviour and is why :meth:`coverage` exists
        beside it -- the caller can tell a zero row from a measured one.
        """
        ids = list(food_ids)
        keys = list(NutrientVector.model_fields)
        out = np.zeros((len(ids), len(keys)), dtype=float)
        for i, food_id in enumerate(ids):
            record = self._db.find(food_id)
            if record is None:
                continue
            values = record.nutrients_per_100g.as_dict()
            out[i] = [values[k] for k in keys]
        return out

    def coverage(self, food_id: str) -> NutrientCoverage:
        """Which nutrients this food's source reports.

        An unknown food reports nothing -- which is true, and is not the same as
        reporting zeros.
        """
        record = self._db.find(food_id)
        return NutrientCoverage(food_id, record.reported_nutrients if record else frozenset())

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._db)

    def count(self, *, include_unverified: bool = False) -> int:
        return len(self.records(include_unverified=include_unverified))

    def source_types(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._db.records:
            counts[record.source_type] = counts.get(record.source_type, 0) + 1
        return dict(sorted(counts.items()))


@lru_cache(maxsize=1)
def get_food_repository() -> FoodRepository:
    """The process-wide repository, over the process-wide :class:`FoodDB`."""
    return FoodRepository()
