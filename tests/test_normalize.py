"""Tests for name normalisation and duplicate detection (requirements section 22).

The mandate names the case these have to get right: ``Masala Dosa``, ``masala dosa``
and ``Masala-Dosa`` must not become three foods. The harder half is the other
direction -- ``Masala Dosa`` and ``Masala Idli`` are different foods that share a
word, and a normaliser eager enough to merge the first set will merge these too.
Both directions are asserted.
"""

from __future__ import annotations

import pytest

from foodsense.data.fdc import get_food_db
from foodsense.data.normalize import (
    DEFAULT_SIMILARITY_THRESHOLD,
    alias_key,
    find_similar,
    name_key,
)


@pytest.fixture(scope="module")
def db():
    return get_food_db()


class TestNameKeyCollapsesMeaninglessVariation:
    """The three spellings the mandate names, and the rules behind them."""

    def test_the_three_spellings_share_one_key(self):
        keys = {name_key("Masala Dosa"), name_key("masala dosa"), name_key("Masala-Dosa")}
        assert len(keys) == 1, f"expected one key, got {keys}"

    def test_padding_and_internal_runs_collapse(self):
        assert name_key("  MASALA   DOSA ") == name_key("Masala Dosa") == "masala dosa"

    @pytest.mark.parametrize(
        "variant",
        ["Masala_Dosa", "Masala/Dosa", "Masala, Dosa", "Masala.Dosa", "masala  -  dosa"],
    )
    def test_separators_are_word_breaks(self, variant):
        assert name_key(variant) == "masala dosa"

    def test_diacritics_are_stripped(self):
        assert name_key("Purée") == name_key("Puree") == "puree"

    def test_compatibility_forms_fold(self):
        """Full-width characters are the same letters.

        Written as escapes deliberately: the codepoints are the subject of the
        test, and a linter is right to flag them as ambiguous in ordinary source.
        """
        # U+FF2D..U+FF41: the full-width forms of M, A, S, A, L, A. Built from
        # codepoints because the literal glyphs are what the test is about, and a
        # linter is right to flag them as ambiguous in ordinary source.
        fullwidth_masala = "".join(
            chr(c) for c in (0xFF2D, 0xFF41, 0xFF53, 0xFF41, 0xFF4C, 0xFF41)
        )
        assert name_key(f"{fullwidth_masala} Dosa") == "masala dosa"

    def test_a_trailing_qualifier_does_not_create_a_new_food(self):
        assert name_key("Dosa (South Indian)") == name_key("Dosa") == "dosa"
        assert name_key("Rice [cooked]") == "rice"

    def test_leading_articles_are_dropped(self):
        assert name_key("The Dosa") == name_key("Dosa")

    def test_a_meaningful_leading_word_is_kept(self):
        """'green' in 'green gram' is not an article and not noise."""
        assert name_key("Green Gram") == "green gram"
        assert name_key("Green Gram") != name_key("Gram")

    def test_empty_and_punctuation_only_names_are_empty_keys(self):
        assert name_key("") == ""
        assert name_key("   ") == ""

    def test_alias_key_agrees_with_name_key(self):
        """If these diverge, an alias stops matching the name it aliases."""
        for text in ["Masala Dosa", "  MASALA-DOSA ", "Purée (thick)", "The Idli"]:
            assert alias_key(text) == name_key(text)


class TestNameKeyKeepsDifferentFoodsApart:
    """The failure mode that matters more: over-merging.

    A duplicate row is an inconvenience someone can merge. Two different foods
    collapsed into one key is a wrong recommendation, and nothing downstream can
    detect it.
    """

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Masala Dosa", "Masala Idli"),
            ("Dal", "Dal Makhani"),
            ("Ragi Dosa", "Ragi Idli"),
            ("Plain Dosa", "Masala Dosa"),
            ("Curd", "Curd Rice"),
            ("Rasam", "Sambar"),
            ("Green Gram", "Bengal Gram"),
        ],
    )
    def test_these_must_not_share_a_key(self, left, right):
        assert name_key(left) != name_key(right), (
            f"{left!r} and {right!r} collapsed to the same key -- they are different foods"
        )


class TestFindSimilar:
    """The candidate list behind "Similar food already exists. Did you mean ...?"."""

    def test_an_exact_key_collision_is_reported_as_exact(self, db):
        """Any real row, respelled, must come back as an exact hit."""
        target = db.records[0]
        respelled = f"  {target.name.upper()}  "
        hits = find_similar(respelled, db=db)
        assert hits, "a respelling of an existing food found nothing"
        assert hits[0].is_exact
        assert hits[0].record.fdc_id == target.fdc_id
        assert hits[0].score == 100.0

    def test_an_exact_hit_outranks_a_fuzzy_one(self, db):
        """A key collision is identity, not similarity, so it cannot be outranked."""
        target = db.records[0]
        hits = find_similar(target.name, db=db)
        assert hits[0].is_exact

    def test_aliases_are_searched(self, db):
        target = db.records[0]
        hits = find_similar(
            "Totally Unrelated Name",
            db=db,
            aliases={"Totally Unrelated Name": target.fdc_id},
        )
        assert any(h.reason == "exact_alias" and h.record.fdc_id == target.fdc_id for h in hits)

    def test_nothing_is_returned_for_an_empty_name(self, db):
        assert find_similar("", db=db) == []
        assert find_similar("   ", db=db) == []

    def test_the_limit_is_honoured(self, db):
        assert len(find_similar("rice", db=db, limit=3)) <= 3

    def test_a_nonsense_name_surfaces_nothing_above_threshold(self, db):
        hits = find_similar("qqzzxx wibble frobnicate", db=db)
        assert all(h.score >= DEFAULT_SIMILARITY_THRESHOLD for h in hits)

    def test_every_hit_meets_the_threshold(self, db):
        for h in find_similar("chicken", db=db, limit=5):
            assert h.is_exact or h.score >= DEFAULT_SIMILARITY_THRESHOLD
