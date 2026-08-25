"""Runtime access to the curated USDA food database.

Three jobs, all of them hot paths:

* **Lookup** -- ``db.get(food_id)`` returns a :class:`FoodRecord` with nutrients,
  category, hazard class, allowed forms and interaction tags.
* **Fuzzy matching** -- ``db.match("whole grapes")`` maps a free-text name onto a
  real ``fdc_id``. Stage 3 uses it to ground generated names; Stage 4 uses it to
  decide whether a generated item exists at all.
* **Nutrient recomputation** -- ``db.nutrients_for(meal)`` recomputes a meal's
  nutrients from database ground truth, which is the whole point of Stage 4.

The database is loaded once per process and held in memory; it is ~2.6k rows, so
the parquet mirror loads in milliseconds and every lookup after that is a dict hit.

Preparation form deliberately does **not** change nutrients. Quartering a grape
does not alter its composition, and the raw/cooked distinction is already encoded
in separate USDA rows ("Carrots, raw" vs "Carrots, cooked, boiled, drained").
Form exists for safety and texture, not for nutrition.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from foodsense import FOOD_DB_PARQUET, FOOD_DB_SQLITE
from foodsense.schemas import NUTRIENTS, Form, Meal, MealItem, NutrientVector

__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "FoodDB",
    "FoodRecord",
    "FoodDatabaseMissingError",
    "get_food_db",
]

#: Stage 4 accepts a fuzzy name match at or above this score and flags anything
#: below it as ``unmatched``. Set by the design brief.
DEFAULT_MATCH_THRESHOLD = 85.0

# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------
#
# Off-the-shelf rapidfuzz scorers do badly here, and it is worth saying why,
# because Stage 4's entire value rests on not mis-matching foods.
#
# USDA descriptions are long and comma-qualified ("Grapes, american type (slip
# skin), raw") while the names Stage 3 emits are short ("grapes"). Every scorer
# with a *partial* component -- WRatio above all -- then scores a short query
# near-perfectly against any long string that happens to contain a substring of
# it: measured on this database, WRatio ranked "Babyfood, Multigrain whole grain
# cereal" as the best match for "whole grapes" at 85.5, and returned 85.5 for
# nine unrelated queries out of ten. token_set_ratio has the mirror problem: it
# returns 100 whenever *any* token overlaps, so "peanuts" matches "Candies, milk
# chocolate coated peanuts" perfectly.
#
# What actually discriminates is scoring both directions at once:
#
#   precision  how much of the QUERY the food name accounts for
#              -- kills "Candies, milk chocolate coated peanuts" for "salted peanuts"
#   recall     how much of the food NAME the query accounts for, weighted so that
#              earlier tokens count more (USDA puts identity first and qualifiers
#              after) -- kills long, over-specified rows when a plain one exists
#
# The final score leans on precision (a query must be satisfied) and uses the
# F1 of the two to break ties toward the least over-qualified food.

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Dropped from both sides: grammatical filler plus the boilerplate USDA staples
#: into its descriptions, none of which carries food identity.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "of", "or", "in", "with", "without", "from", "for",
        "to", "as", "by", "includes", "including", "usda", "s", "commodity",
        "distribution", "program", "type", "types", "all", "prepared", "made",
    }
)

#: Weight of the precision term; the F1 term takes the remainder.
_PRECISION_WEIGHT = 0.65

#: A query token that is not in the vocabulary is expanded to vocabulary tokens
#: at or above this similarity, which is what makes "saltine" find "saltines"
#: (93), "grape" find "grapes" (91) and "fries" find "fried" (80) -- while still
#: refusing "grain" for "grapes" (55) and "beans" for "beets" (60). It is a
#: cheap stand-in for stemming, which USDA's inconsistent phrasing would defeat.
_TOKEN_EXPANSION_THRESHOLD = 80.0


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed, order-preserving and de-duplicated."""
    seen: dict[str, None] = {}
    for token in _TOKEN_RE.findall(text.lower()):
        if token not in _STOPWORDS and len(token) > 1:
            seen.setdefault(token, None)
    return list(seen)


#: Tokens that name a *non-default* preparation. Leaving one unmatched has to be
#: expensive, or "carrot" resolves to "Carrot, dehydrated" (341 kcal/100 g) while
#: "Carrots, raw" (41 kcal/100 g) sits right next to it -- both are one unmatched
#: qualifier away from the query, so plain position weighting cannot separate them.
_HEAVY_QUALIFIERS = frozenset(
    {
        "dehydrated", "dried", "powder", "powdered", "condensed", "concentrate",
        "concentrated", "instant", "canned", "frozen", "syrup", "fried", "breaded",
        "battered", "candied", "pickled", "smoked", "cured", "sweetened", "glutinous",
        "imitation", "creamed", "freeze", "juice", "paste", "puree", "pureed", "flour",
        "meal", "extract", "babyfood", "spread", "substitute", "flavored", "coated",
        # Structural parts. "Orange peel" and "Grape leaves" are not oranges and
        # grapes, but their head token matches those queries exactly.
        "peel", "peels", "rind", "zest", "leaves", "leaf", "skin", "skins",
        "stems", "pits", "hulls", "shells",
    }
)
_HEAVY_WEIGHT = 2.5

#: Tokens naming the food's *default* state. Failing to match one should barely
#: count, so a bare "carrots" is happy to land on "Carrots, raw".
_NEUTRAL_QUALIFIERS = frozenset(
    {
        "raw", "fresh", "cooked", "boiled", "drained", "plain", "regular", "whole",
        "ripe", "unenriched", "enriched", "salt", "unsalted", "table",
    }
)
_NEUTRAL_WEIGHT = 0.3


def _same_stem(a: str, b: str) -> bool:
    """True when two tokens differ only by an English plural suffix.

    Worth special-casing: a plural is not an *approximate* match, it is the same
    word. Scoring "grape"/"grapes" at 0.91 while "grape"/"grape" scores 1.0 makes
    "Grape leaves" the best match for "grape", which is exactly wrong.
    """
    short, long_ = sorted((a, b), key=len)
    if short == long_:
        return True
    if long_ in (short + "s", short + "es"):
        return True
    return short.endswith("y") and long_ == short[:-1] + "ies"


def _position_weights(tokens: list[str]) -> list[float]:
    """Token weights for the recall term.

    Harmonic decay by position, because USDA puts identity first and qualifiers
    after, scaled by how much the token says about *preparation* (see the two
    qualifier sets above).
    """
    weights = []
    for i, token in enumerate(tokens):
        weight = 1.0 / (i + 1)
        if token in _HEAVY_QUALIFIERS:
            weight *= _HEAVY_WEIGHT
        elif token in _NEUTRAL_QUALIFIERS:
            weight *= _NEUTRAL_WEIGHT
        weights.append(weight)
    return weights


class FoodDatabaseMissingError(RuntimeError):
    """Raised when the curated database has not been built yet."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Food database not found at {path}.\n"
            f"Build it with:  python -m foodsense.data.build_food_db\n"
            f"(or `make data` / `./make.ps1 data`). See data/README.md."
        )


@dataclass(frozen=True, slots=True)
class FoodRecord:
    """One curated USDA food, with everything the pipeline needs about it."""

    fdc_id: str
    name: str
    category: str
    hazard_class: str
    default_form: Form
    allowed_forms: tuple[Form, ...]
    tags: frozenset[str] = field(default_factory=frozenset)
    nutrients_per_100g: NutrientVector = field(default_factory=NutrientVector)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def permits(self, form: Form) -> bool:
        return form in self.allowed_forms

    def nutrients_for(self, quantity_g: float) -> NutrientVector:
        """Nutrients contributed by ``quantity_g`` grams of this food."""
        return self.nutrients_per_100g.scaled(quantity_g / 100.0)

    def as_item(self, quantity_g: float, form: Form | None = None) -> MealItem:
        """Build a :class:`MealItem` for this food at its default (or given) form."""
        return MealItem(
            food_id=self.fdc_id,
            name=self.name,
            quantity_g=quantity_g,
            form=form if form is not None else self.default_form,
        )


class FoodDB:
    """In-memory view over ``data/processed/food_db.{parquet,sqlite}``."""

    def __init__(self, records: list[FoodRecord]) -> None:
        self._records = records
        self._by_id: dict[str, FoodRecord] = {r.fdc_id: r for r in records}
        self._names: list[str] = [r.name for r in records]

        # --- name-matching index (built once, ~2.6k rows) ------------------
        self._tokens: list[list[str]] = [_tokenize(name) for name in self._names]
        self._weights: list[list[float]] = [_position_weights(t) for t in self._tokens]
        self._weight_totals: list[float] = [sum(w) or 1.0 for w in self._weights]
        self._token_sets: list[frozenset[str]] = [frozenset(t) for t in self._tokens]

        # Inverted index so a query only scores foods that share a token with it,
        # instead of all 2.6k rows.
        index: dict[str, list[int]] = {}
        for i, tokens in enumerate(self._tokens):
            for token in tokens:
                index.setdefault(token, []).append(i)
        self._index = index
        self._vocab: list[str] = list(index)

        # --- vectorised nutrient lookup -----------------------------------
        # Stage 2 evaluates thousands of candidate meals per run and each one
        # needs a nutrient total. Building a Pydantic NutrientVector per item
        # dominates that loop, so the per-100 g values are also held as one
        # float matrix and totalled with numpy.
        self.nutrient_index: dict[str, int] = {n: i for i, n in enumerate(NUTRIENTS)}
        self._row_of: dict[str, int] = {r.fdc_id: i for i, r in enumerate(records)}
        self._nutrient_matrix = np.asarray(
            [r.nutrients_per_100g.as_tuple() for r in records], dtype=np.float64
        ).reshape(len(records), len(NUTRIENTS))
        self._expansion_cache: dict[str, frozenset[str]] = {}

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> FoodDB:
        """Load from parquet if available, else sqlite. Raises if neither exists."""
        if path is not None:
            frame = cls._read(path)
        elif FOOD_DB_PARQUET.exists():
            frame = cls._read(FOOD_DB_PARQUET)
        elif FOOD_DB_SQLITE.exists():
            frame = cls._read(FOOD_DB_SQLITE)
        else:
            raise FoodDatabaseMissingError(FOOD_DB_PARQUET)
        return cls(cls._to_records(frame))

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FoodDatabaseMissingError(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        with sqlite3.connect(path) as conn:
            return pd.read_sql("SELECT * FROM foods", conn)

    @staticmethod
    def _to_records(frame: pd.DataFrame) -> list[FoodRecord]:
        records: list[FoodRecord] = []
        nutrient_columns = [n for n in NUTRIENTS if n in frame.columns]
        for row in frame.to_dict(orient="records"):
            allowed = tuple(Form(f) for f in json.loads(row["allowed_forms"]))
            records.append(
                FoodRecord(
                    fdc_id=str(row["fdc_id"]),
                    name=row["name"],
                    category=row["category"],
                    hazard_class=row.get("hazard_class") or "",
                    default_form=Form(row["default_form"]),
                    allowed_forms=allowed,
                    tags=frozenset(json.loads(row["tags"])),
                    nutrients_per_100g=NutrientVector(
                        **{n: float(row[n]) for n in nutrient_columns}
                    ),
                )
            )
        return records

    # -- basic access -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, food_id: object) -> bool:
        return str(food_id) in self._by_id

    def __iter__(self):
        return iter(self._records)

    def get(self, food_id: str) -> FoodRecord:
        """Look up by ``fdc_id``. Raises :class:`KeyError` if absent."""
        try:
            return self._by_id[str(food_id)]
        except KeyError:
            raise KeyError(f"No food with fdc_id {food_id!r} in the curated database") from None

    def find(self, food_id: str) -> FoodRecord | None:
        """Like :meth:`get` but returns ``None`` instead of raising."""
        return self._by_id.get(str(food_id))

    @property
    def records(self) -> list[FoodRecord]:
        return list(self._records)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    # -- filtered views -----------------------------------------------------

    def by_category(self, category: str) -> list[FoodRecord]:
        return [r for r in self._records if r.category == category]

    def by_hazard(self, hazard_class: str) -> list[FoodRecord]:
        return [r for r in self._records if r.hazard_class == hazard_class]

    def by_tag(self, tag: str) -> list[FoodRecord]:
        return [r for r in self._records if tag in r.tags]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self._records:
            counts[r.category] = counts.get(r.category, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    # -- fuzzy matching -----------------------------------------------------

    def _expand(self, token: str) -> dict[str, float]:
        """Vocabulary tokens equivalent to ``token``, each with a match *quality*.

        Two things matter here.

        It runs even when the token is already in the vocabulary: "carrot" *is* a
        vocabulary token (from "Carrot, dehydrated"), so an in-vocabulary shortcut
        would never let it reach "carrots", and the query would resolve to
        dehydrated carrot at 341 kcal/100 g.

        And the match is graded, not binary. Near-spellings are genuinely near --
        "orange"/"oranges" is 92 but "orange"/"borage" is 83, and treating both as
        an exact hit makes borage the best match for orange. Quality is the
        similarity itself, so an exact token always outranks an approximate one.
        """
        cached = self._expansion_cache.get(token)
        if cached is not None:
            return cached
        hits = process.extract(
            token, self._vocab, scorer=fuzz.ratio, limit=8, score_cutoff=_TOKEN_EXPANSION_THRESHOLD
        )
        expanded = {
            name: (1.0 if _same_stem(token, name) else score / 100.0) for name, score, _ in hits
        }
        if token in self._index:
            expanded[token] = 1.0
        self._expansion_cache[token] = expanded
        return expanded

    def search(self, query: str, limit: int = 10) -> list[tuple[FoodRecord, float]]:
        """Rank foods by similarity to ``query``, best first.

        See the module-level note on why this is a bespoke scorer rather than one
        of rapidfuzz's: the stock ones are systematically wrong on short queries
        against long USDA descriptions.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        expansions = [self._expand(t) for t in query_tokens]

        # Best quality any query token can offer each vocabulary token.
        matchable: dict[str, float] = {}
        for expansion in expansions:
            for token, quality in expansion.items():
                if quality > matchable.get(token, 0.0):
                    matchable[token] = quality

        candidates: set[int] = set()
        for token in matchable:
            candidates.update(self._index.get(token, ()))
        if not candidates:
            return []

        scored: list[tuple[float, int, FoodRecord]] = []
        for i in candidates:
            name_tokens = self._token_sets[i]

            # Precision: how well each query token is satisfied by this food.
            precision = (
                sum(
                    max((q for t, q in expansion.items() if t in name_tokens), default=0.0)
                    for expansion in expansions
                )
                / len(query_tokens)
            )

            # Recall: how much of the food's name the query accounts for.
            matched_weight = sum(
                w * matchable.get(token, 0.0)
                for token, w in zip(self._tokens[i], self._weights[i], strict=True)
            )
            recall = matched_weight / self._weight_totals[i]

            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            score = 100.0 * (_PRECISION_WEIGHT * precision + (1 - _PRECISION_WEIGHT) * f1)
            # Tie-break toward the shorter, less-qualified name.
            scored.append((score, -len(self._tokens[i]), self._records[i]))

        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [(record, round(score, 1)) for score, _, record in scored[:limit]]

    def match(
        self, name: str, threshold: float = DEFAULT_MATCH_THRESHOLD
    ) -> tuple[FoodRecord | None, float]:
        """Best fuzzy match for ``name``, or ``(None, score)`` if below ``threshold``.

        Returning the score even on failure matters: Stage 4 reports *how close*
        an unmatched generated item was, which is more useful than a bare miss.
        """
        hits = self.search(name, limit=1)
        if not hits:
            return None, 0.0
        record, score = hits[0]
        return (record if score >= threshold else None), score

    # -- nutrient recomputation --------------------------------------------

    def nutrients_for_item(self, item: MealItem) -> NutrientVector:
        """Ground-truth nutrients for one meal item. Unknown foods contribute zero."""
        record = self.find(item.food_id)
        if record is None:
            return NutrientVector.zeros()
        return record.nutrients_for(item.quantity_g)

    def nutrients_for(self, meal: Meal | list[MealItem]) -> NutrientVector:
        """Recompute a whole meal's nutrients from the database.

        This is the Stage-4 ground truth: whatever Stage 3 claimed, *this* is what
        the meal actually contains.
        """
        items = meal.items if isinstance(meal, Meal) else meal
        return NutrientVector.sum([self.nutrients_for_item(i) for i in items])

    def nutrient_totals(self, meal: Meal | list[MealItem]) -> np.ndarray:
        """Meal nutrient totals as a float array in canonical :data:`NUTRIENTS` order.

        The vectorised twin of :meth:`nutrients_for`. Same arithmetic, no object
        allocation -- use this on hot paths and ``nutrients_for`` when a typed
        :class:`NutrientVector` is what the caller actually wants.
        """
        items = meal.items if isinstance(meal, Meal) else meal
        rows: list[int] = []
        weights: list[float] = []
        for item in items:
            row = self._row_of.get(item.food_id)
            if row is None:
                continue
            rows.append(row)
            weights.append(item.quantity_g / 100.0)
        if not rows:
            return np.zeros(len(NUTRIENTS), dtype=np.float64)
        return np.asarray(weights) @ self._nutrient_matrix[rows]

    def allowed_forms(self, food_id: str) -> tuple[Form, ...]:
        record = self.find(food_id)
        return record.allowed_forms if record else (Form.WHOLE,)

    def unknown_ids(self, meal: Meal | list[MealItem]) -> list[str]:
        """Food ids in ``meal`` that are not in the database."""
        items = meal.items if isinstance(meal, Meal) else meal
        return [i.food_id for i in items if i.food_id not in self._by_id]


@lru_cache(maxsize=1)
def get_food_db() -> FoodDB:
    """Process-wide cached food database.

    Every stage calls this rather than constructing its own, so a pipeline run
    parses the parquet file exactly once.
    """
    return FoodDB.load()
