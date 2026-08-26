"""Offline BM25 retrieval over the curated USDA food database.

This is the R in Stage 3's RAG. Its job is to give the generator a short list of
*real* foods -- real names, real ``fdc_id`` values -- so that whatever comes back
can be checked against something exact in Stage 4. A generator handed no
candidates invents plausible foods; a generator handed five real ones tends to
pick from them.

BM25 over names and categories, via ``rank_bm25``, entirely local: no embedding
model, no network, no API key. That is a deliberate constraint rather than a
simplification -- the faculty demo has to work with the Wi-Fi off.

Retrieval and Stage-4 matching deliberately use different algorithms. BM25 is a
bag-of-words ranker that answers "which foods are plausibly related to this
phrase"; the matcher in ``data/fdc.py`` answers "is this specific name the same
food as one we hold", weighting precision, recall and preparation qualifiers.
Using one for both would make Stage 4's check partly circular -- it would be
grading the retriever's own notion of similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from foodsense.data.fdc import FoodDB, FoodRecord, get_food_db

__all__ = ["FoodRetriever", "RetrievedFood", "get_retriever"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class RetrievedFood:
    """One candidate, with the score that put it there."""

    record: FoodRecord
    score: float

    @property
    def name(self) -> str:
        return self.record.name

    @property
    def fdc_id(self) -> str:
        return self.record.fdc_id


class FoodRetriever:
    """BM25 index over food names, categories and interaction tags."""

    def __init__(self, db: FoodDB | None = None) -> None:
        self.db = db or get_food_db()
        self._records = self.db.records
        # Category and tags join the document text so that a query like
        # "something low sodium" or "a vegetable" can retrieve on more than the
        # literal name.
        corpus = [
            _tokenize(f"{r.name} {r.category} {' '.join(sorted(r.tags))}") for r in self._records
        ]
        self._bm25 = BM25Okapi(corpus)

    def __len__(self) -> int:
        return len(self._records)

    def search(self, query: str, k: int = 5) -> list[RetrievedFood]:
        """Top ``k`` foods for a free-text query, best first."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            RetrievedFood(record=self._records[i], score=float(scores[i]))
            for i in order
            if scores[i] > 0
        ]

    def candidates_for(self, names: list[str], k: int = 5) -> dict[str, list[str]]:
        """Retrieved names per query, recorded in the trace for provenance.

        Kept as plain strings because this ends up in the ``PipelineTrace`` that
        the API serialises and the UI shows.
        """
        return {name: [hit.name for hit in self.search(name, k)] for name in names if name.strip()}

    def ground(self, name: str) -> FoodRecord | None:
        """Best real food for a generated name, or ``None`` if nothing scores.

        Stage 4 falls back to this when a generated item cannot be matched
        confidently: rather than dropping the item, it is replaced by the closest
        thing that genuinely exists.
        """
        hits = self.search(name, k=1)
        return hits[0].record if hits else None


@lru_cache(maxsize=1)
def get_retriever() -> FoodRetriever:
    """Process-wide cached retriever. Building the index costs ~2.6k tokenisations."""
    return FoodRetriever()
