"""Name normalisation and duplicate detection.

Requirements section 22: ``Masala Dosa``, ``masala dosa`` and ``Masala-Dosa`` must
not become three independent foods. That is one problem with two halves.

The first half is a **normal form**. :func:`name_key` collapses the variations that
carry no meaning -- case, Unicode composition, diacritics, separators, surrounding
whitespace, a leading article, a trailing parenthetical -- so that names differing
only in those ways produce the same key and collide on a unique index. This is
exact, cheap, and is what actually prevents duplicates: it runs on every insert and
does not depend on anyone reading a warning.

The second half is **similarity**, for the names a normal form cannot reach.
``Ragi Dosa`` and ``Ragi Dose`` are different strings under any normalisation, and
whether they are the same food is a judgement the system should put to the person
adding it rather than decide alone. :func:`find_similar` produces the candidates
behind the mandate's "Similar food already exists. Did you mean ...?" prompt.

The scorer is :meth:`FoodDB.search` -- the one Stage 4 already matches against.
Reused, not reimplemented, and deliberately not supplemented with a second fuzzy
library: two scorers would eventually disagree about whether two names are the same
food, and the disagreement would surface as a duplicate that the add-flow accepted
and verification then rejected.

Note that this is *not* the Stage-3 retriever. Stage 3 ranks by BM25 and Stage 4
matches with this scorer, kept apart so verification does not grade the retriever
against the retriever's own notion of similarity. Duplicate detection is a matching
question, not a retrieval one, so it belongs on the Stage-4 side of that line.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from foodsense.data.fdc import FoodDB, FoodRecord, get_food_db

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "SimilarFood",
    "alias_key",
    "find_similar",
    "name_key",
]

#: Score at or above which two names are worth showing the user as possibly the
#: same food. Lower than Stage 4's match threshold on purpose: a false positive
#: here costs one dismissed prompt, while a false negative costs a duplicate row
#: that then has to be merged by hand.
DEFAULT_SIMILARITY_THRESHOLD = 60.0

#: Stripped from the front of a name. Only articles -- a leading word that changes
#: no food's identity. Deliberately short: "green" in "green gram" is not noise.
_LEADING_ARTICLES = ("the", "a", "an")

#: A trailing qualifier in brackets: "Dosa (South Indian)", "Rice, cooked (boiled)".
#: Removed from the key so the bracket is not what makes two rows distinct, but
#: kept in ``canonical_name`` so nothing is lost from what the user sees.
_PARENTHETICAL = re.compile(r"\s*[\(\[\{][^\)\]\}]*[\)\]\}]")

#: Separators that are word breaks in one spelling and nothing in another:
#: "Masala-Dosa", "masala_dosa", "Masala  Dosa".
_SEPARATORS = re.compile(r"[-_/\\,.]+")

_WHITESPACE = re.compile(r"\s+")


def name_key(name: str) -> str:
    """The normal form of a food name: two names sharing this key are one food.

    Casefolded, NFKC-composed, stripped of diacritics, with separators and runs of
    whitespace collapsed to single spaces, leading articles and parenthetical
    qualifiers removed.

    >>> name_key("Masala Dosa") == name_key("masala dosa") == name_key("Masala-Dosa")
    True
    >>> name_key("   MASALA   DOSA ")
    'masala dosa'
    """
    if not name:
        return ""

    # NFKC first: it folds compatibility forms (full-width letters, ligatures) into
    # their ordinary equivalents, so the diacritic strip below sees a predictable
    # shape.
    text = unicodedata.normalize("NFKC", name)
    text = _PARENTHETICAL.sub(" ", text)
    text = _SEPARATORS.sub(" ", text)

    # Decompose, drop the combining marks, recompose. This is what makes "Purée"
    # and "Puree" one key. Transliteration of non-Latin scripts is deliberately not
    # attempted -- a wrong transliteration silently merges two different foods,
    # which is worse than leaving them separate for a human to notice.
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = unicodedata.normalize("NFC", text)

    text = text.casefold()
    text = _WHITESPACE.sub(" ", text).strip()

    words = text.split(" ")
    while len(words) > 1 and words[0] in _LEADING_ARTICLES:
        words.pop(0)
    return " ".join(words)


def alias_key(alias: str) -> str:
    """Normal form for an alias. Same rules as :func:`name_key`, by construction.

    A separate function because the two are conceptually different columns that
    must stay comparable; if the rules ever diverge, an alias would stop matching
    the name it is an alias for.
    """
    return name_key(alias)


@dataclass(frozen=True, slots=True)
class SimilarFood:
    """One candidate for "did you mean ...?", with why it was surfaced."""

    record: FoodRecord
    score: float
    #: ``exact_name`` | ``exact_alias`` | ``fuzzy``. The caller renders these
    #: differently: an exact key collision is a duplicate and should block, while a
    #: fuzzy hit is a question and should only warn.
    reason: str

    @property
    def is_exact(self) -> bool:
        return self.reason in ("exact_name", "exact_alias")


def find_similar(
    name: str,
    *,
    cuisine: str | None = None,
    limit: int = 5,
    db: FoodDB | None = None,
    repository: object | None = None,
    aliases: dict[str, str] | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[SimilarFood]:
    """Foods that may already be ``name``, best first.

    Three passes, in order of confidence: exact ``name_key``, then exact
    ``alias_key``, then fuzzy. An exact hit is reported as exact even when the
    fuzzy scorer would rank something else higher -- a key collision is a fact
    about identity, not a similarity judgement, so it must not be outranked.

    ``cuisine`` narrows the fuzzy pass only. Two foods with the same key are the
    same food whatever cuisine they were filed under, and filtering exact hits by
    cuisine would let a mislabelled row become a duplicate.
    """
    # Duplicate detection must see EVERY row, pending and rejected included:
    # warning about a collision with a row awaiting review is the entire point,
    # and hiding it would let the same food be submitted twice.
    if repository is not None:
        records = repository.records(include_unverified=True)
        db = repository.db
    else:
        db = db if db is not None else get_food_db()
        records = db.records
    key = name_key(name)
    if not key:
        return []

    found: list[SimilarFood] = []
    seen: set[str] = set()

    for record in records:
        if name_key(record.name) == key:
            found.append(SimilarFood(record=record, score=100.0, reason="exact_name"))
            seen.add(record.fdc_id)

    if aliases:
        for alias, food_id in aliases.items():
            if food_id in seen or alias_key(alias) != key:
                continue
            record = db.find(food_id)
            if record is not None:
                found.append(SimilarFood(record=record, score=100.0, reason="exact_alias"))
                seen.add(food_id)

    for record, score in db.search(name, limit=limit * 4):
        if record.fdc_id in seen or score < threshold:
            continue
        if cuisine is not None and getattr(record, "cuisine", None) not in (None, cuisine):
            continue
        found.append(SimilarFood(record=record, score=score, reason="fuzzy"))
        seen.add(record.fdc_id)

    return found[:limit]
