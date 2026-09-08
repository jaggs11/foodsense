"""Nutrient bases, and the guard that stops two of them being compared.

Every nutrient number in this project answers a question of the form "how much per
*what*". The corpus is about to hold both answers at once: composition tables report
per 100 g of edible portion, recipe-level sources report per serving, and a user
adding a food types whichever the packet in front of them shows.

Requirements section 27 is unambiguous about the consequence -- *do not compare
incompatible serving bases* -- and it is the kind of error that produces a
confident, plausible, wrong answer. Idli at 150 kcal per serving next to dosa at
250 kcal per 100 g reads as "dosa is worse" whichever way round the truth is, and
nothing about the output looks broken.

So the basis travels with the numbers rather than being remembered by the caller.
:class:`NutrientBasis` is carried on every nutrient payload the API returns,
:func:`to_per_100g` and :func:`to_per_serving` are the only sanctioned conversions,
and :func:`require_same_basis` raises rather than letting a mixed comparison happen
quietly.

**Canonical storage is per 100 g of edible portion.** Everything is converted on the
way in, so the wide nutrient columns can be read as a vector without asking what they
mean -- which is what keeps the Stage-1 hot loop a matrix multiply. Per-serving is a
presentation concern and a property of an ingestion source, never a storage format.

"Edible portion" is part of the definition, not a footnote: composition tables state
whether a value is per 100 g as purchased or per 100 g edible, and the two differ by
the refuse fraction. A value whose source does not say is recorded as its stated
basis and flagged, never silently promoted.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "BasisMismatchError",
    "NutrientBasis",
    "convert",
    "per_serving_factor",
    "require_same_basis",
    "to_per_100g",
    "to_per_serving",
]


class NutrientBasis(StrEnum):
    """What a nutrient number is *per*.

    ``PER_100G``
        Per 100 g of edible portion. The canonical storage basis.
    ``PER_SERVING``
        Per one serving, whose size in grams must be known alongside it. Useless
        without ``serving_size_g`` -- a per-serving number with no serving size is
        not a measurement, and ingestion rejects it rather than guessing.
    ``UNKNOWN``
        The source does not state a basis. Storable, never comparable, and excluded
        from recommendation. Present so that "we do not know" is representable;
        without it the only way to record such a row is to invent a basis for it.
    """

    PER_100G = "per_100g_edible_portion"
    PER_SERVING = "per_serving"
    UNKNOWN = "unknown"


class BasisMismatchError(ValueError):
    """Raised when nutrient values on different bases would have been compared."""


def per_serving_factor(serving_size_g: float) -> float:
    """Multiplier taking a per-100 g value to a per-serving value."""
    if serving_size_g <= 0:
        raise ValueError(f"serving_size_g must be positive, got {serving_size_g!r}")
    return serving_size_g / 100.0


def to_per_100g(
    value: float | None, basis: NutrientBasis, serving_size_g: float | None
) -> float | None:
    """Convert one value to the canonical basis.

    ``None`` in, ``None`` out: a nutrient the source does not report stays
    unreported through conversion. Scaling it would turn a gap into a zero, which
    ruling R8 forbids everywhere in this pipeline.
    """
    if value is None:
        return None
    if basis is NutrientBasis.PER_100G:
        return float(value)
    if basis is NutrientBasis.PER_SERVING:
        if serving_size_g is None:
            raise ValueError("per-serving values need serving_size_g to convert")
        return float(value) / per_serving_factor(serving_size_g)
    raise BasisMismatchError(
        "cannot convert a value whose basis is unknown; record it as-is and "
        "exclude it from comparison instead"
    )


def to_per_serving(
    value: float | None, basis: NutrientBasis, serving_size_g: float | None
) -> float | None:
    """Convert one value to per-serving. ``None`` passes through unchanged."""
    if value is None:
        return None
    if basis is NutrientBasis.PER_SERVING:
        return float(value)
    if basis is NutrientBasis.PER_100G:
        if serving_size_g is None:
            raise ValueError("converting to per-serving needs serving_size_g")
        return float(value) * per_serving_factor(serving_size_g)
    raise BasisMismatchError(
        "cannot convert a value whose basis is unknown; record it as-is and "
        "exclude it from comparison instead"
    )


def convert(
    value: float | None,
    *,
    frm: NutrientBasis,
    to: NutrientBasis,
    serving_size_g: float | None = None,
) -> float | None:
    """Convert between any two known bases."""
    if frm is to:
        return None if value is None else float(value)
    if to is NutrientBasis.PER_100G:
        return to_per_100g(value, frm, serving_size_g)
    if to is NutrientBasis.PER_SERVING:
        return to_per_serving(value, frm, serving_size_g)
    raise BasisMismatchError(f"no conversion to {to!r}")


def require_same_basis(*bases: NutrientBasis) -> NutrientBasis:
    """Assert every payload shares one basis, and return it.

    Called before any comparison, chart or ranking that puts two foods side by
    side. It raises rather than converting on the caller's behalf on purpose: the
    conversion needs a serving size this function does not have, and silently
    picking one is how a wrong number gets a correct-looking label.

    An ``UNKNOWN`` basis anywhere is fatal to a comparison even if every other
    payload agrees -- that is what excluding unverified data from recommendation
    means in practice.
    """
    if not bases:
        raise ValueError("no bases given")
    distinct = set(bases)
    if NutrientBasis.UNKNOWN in distinct:
        raise BasisMismatchError(
            "a value with an unknown basis cannot be compared; requirements "
            "section 27 requires every chart and table to state its basis"
        )
    if len(distinct) > 1:
        raise BasisMismatchError(
            "refusing to compare nutrient values on different bases: "
            + ", ".join(sorted(b.value for b in distinct))
            + ". Convert to one basis first (units.convert)."
        )
    return next(iter(distinct))
