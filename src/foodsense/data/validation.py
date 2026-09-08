"""Validation for foods entering the knowledge base (requirements section 23).

Deliberately outside ``api/``. Two paths write foods -- the HTTP add-food endpoint
(P6) and bulk ingestion of a manifest-listed dataset (P2) -- and if each carried its
own rules they would drift, leaving a row that the API would have rejected sitting
in the database because a loader was more permissive. One model, both callers.

What is enforced here is *structural* honesty, which is the only kind a schema can
enforce. That a value is a non-negative number, that a serving size is positive,
that a required field is present, that an enum is one of its members. Whether the
number is *true* is not knowable at this layer; that is what provenance
(``source_type``, ``source_reference``, ``verification_status``) and Stage-4
verification are for.

Three choices worth stating.

**``extra="forbid"``.** A misspelled nutrient key silently dropped is a food that
looks complete and is missing a nutrient. Rejecting the payload turns a data-loss
bug into a validation error.

**``None`` is valid for every nutrient and is not zero.** Ruling R8. A source that
does not report iron produces ``iron_mg=None``, and the row is honest about it.
Requiring a number here would force every ingestion path to invent one.

**Energy and protein are required.** Not because they matter most, but because a
"food" reporting neither carries no nutritional information at all, and admitting it
would put a row into the recommendation corpus that can never satisfy a condition
target or be verified against anything.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from foodsense.data.normalize import name_key
from foodsense.data.units import NutrientBasis
from foodsense.schemas import NutrientVector

__all__ = [
    "NUTRIENT_KEYS",
    "REQUIRED_NUTRIENTS",
    "Confidence",
    "Cuisine",
    "FoodCreate",
    "FoodUpdate",
    "SourceType",
    "Texture",
    "VerificationStatus",
    "VegStatus",
]

#: The 33 wide nutrient columns, taken from NutrientVector so the two cannot drift.
NUTRIENT_KEYS: tuple[str, ...] = tuple(NutrientVector.model_fields)

#: Without these a row carries no nutritional information worth storing.
REQUIRED_NUTRIENTS: tuple[str, ...] = ("energy_kcal", "protein_g")

Cuisine = Literal["north_indian", "south_indian", "pan_indian", "international", "other"]
VegStatus = Literal["vegetarian", "non_vegetarian", "eggetarian", "unknown"]
Texture = Literal["puree", "soft_mashed", "soft_solid", "firm", "hard", "liquid", "unknown"]
SourceType = Literal[
    "usda",
    "indian_dataset",
    "research_dataset",
    "government",
    "derived_recipe",
    "verified",
    "user_added",
]
Confidence = Literal["high", "medium", "low", "unknown"]
VerificationStatus = Literal["verified", "pending", "rejected", "unverified"]

#: A nutrient is absent-or-unreported (None) or a non-negative number. There is no
#: upper bound -- per 100 g, water tops out near 100 and energy near 900, but a
#: plausibility ceiling here would reject a real outlier and is not what this layer
#: is for.


#: The 33 nutrient fields as a generated mixin. Generated rather than written out,
#: so the set cannot drift from NutrientVector the first time a nutrient is added.
_NutrientFields: type[BaseModel] = create_model(
    "_NutrientFields",
    **{key: (float | None, Field(default=None, ge=0.0)) for key in NUTRIENT_KEYS},
)


class _FoodBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def _blank_string_is_absent(cls, value: Any) -> Any:
        """An empty string is a field the user left blank, not a value."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class FoodCreate(_NutrientFields, _FoodBase):
    """A new food. Every rule in requirements section 23, plus provenance."""

    canonical_name: str = Field(min_length=1, max_length=300)
    display_name: str | None = Field(default=None, max_length=300)
    cuisine: Cuisine
    region: str | None = Field(default=None, max_length=120)
    category: str | None = Field(default=None, max_length=120)
    veg_status: VegStatus = "unknown"

    serving_size_g: float = Field(gt=0.0, description="must be positive (section 23)")
    serving_unit: str | None = Field(default=None, max_length=40)
    serving_description: str | None = Field(default=None, max_length=200)

    ingredients: list[str] | None = None
    preparation_method: str | None = Field(default=None, max_length=200)
    texture: Texture = "unknown"

    basis: NutrientBasis = NutrientBasis.PER_100G

    source_type: SourceType
    source_name: str | None = Field(default=None, max_length=200)
    source_reference: str | None = Field(default=None, max_length=500)
    source_id: str | None = Field(default=None, max_length=120)
    confidence: Confidence = "unknown"
    verification_status: VerificationStatus = "pending"

    @property
    def name_key(self) -> str:
        return name_key(self.canonical_name)

    @model_validator(mode="after")
    def _require_energy_and_protein(self) -> FoodCreate:
        missing = [k for k in REQUIRED_NUTRIENTS if getattr(self, k, None) is None]
        if missing:
            raise ValueError(
                "a food must report at least energy and protein; missing: "
                + ", ".join(missing)
                + ". A row reporting neither carries no nutritional information."
            )
        return self

    @model_validator(mode="after")
    def _per_serving_needs_a_serving_size(self) -> FoodCreate:
        """Enforced although serving_size_g is already required, because the
        constraint is about the *basis*: a per-serving number without the serving
        it is per is not a measurement."""
        if self.basis is NutrientBasis.PER_SERVING and not self.serving_size_g:
            raise ValueError("per-serving nutrients require serving_size_g")
        return self

    @model_validator(mode="after")
    def _user_added_food_is_never_born_verified(self) -> FoodCreate:
        """Requirements section 10: user-entered nutrition is not authoritative.

        source_reference stays optional for user_added -- a parent reading a packet
        has no citation to give and demanding one only teaches people to type
        anything -- but the record is forced to pending regardless of what the
        caller asked for. Trust is granted by review, never claimed on submission.
        """
        if self.source_type == "user_added":
            object.__setattr__(self, "verification_status", "pending")
        return self

    @model_validator(mode="after")
    def _sourced_data_must_cite_something(self) -> FoodCreate:
        """Anything claiming to come from a dataset has to say which."""
        if self.source_type in ("indian_dataset", "research_dataset", "government"):
            if not (self.source_reference or self.source_id):
                raise ValueError(
                    f"source_type={self.source_type!r} requires source_reference "
                    "or source_id -- provenance is mandatory (section 10)"
                )
        return self

    @field_validator("ingredients")
    @classmethod
    def _ingredients_are_non_empty_strings(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [i.strip() for i in value if i and i.strip()]
        return cleaned or None

    def nutrients(self) -> dict[str, float | None]:
        """The nutrient half of the payload, unreported keys included as None."""
        return {key: getattr(self, key) for key in NUTRIENT_KEYS}

    def reported_nutrients(self) -> set[str]:
        """Keys this source actually reports. The complement is not zero."""
        return {key for key in NUTRIENT_KEYS if getattr(self, key) is not None}


class FoodUpdate(_NutrientFields, _FoodBase):
    """A partial edit. Every field optional; absent means "leave alone".

    ``None`` cannot mean "clear this nutrient" here, because it already means "not
    supplied" -- and conflating the two would let a partial update silently erase a
    measured value. Clearing a nutrient is a deliberate act and gets its own path in
    P6 rather than being an accident of omission.
    """

    canonical_name: str | None = Field(default=None, min_length=1, max_length=300)
    display_name: str | None = Field(default=None, max_length=300)
    cuisine: Cuisine | None = None
    region: str | None = Field(default=None, max_length=120)
    category: str | None = Field(default=None, max_length=120)
    veg_status: VegStatus | None = None

    serving_size_g: float | None = Field(default=None, gt=0.0)
    serving_unit: str | None = Field(default=None, max_length=40)
    serving_description: str | None = Field(default=None, max_length=200)

    ingredients: list[str] | None = None
    preparation_method: str | None = Field(default=None, max_length=200)
    texture: Texture | None = None

    basis: NutrientBasis | None = None

    source_name: str | None = Field(default=None, max_length=200)
    source_reference: str | None = Field(default=None, max_length=500)
    confidence: Confidence | None = None
    verification_status: VerificationStatus | None = None

    def changes(self) -> dict[str, Any]:
        """Only the fields the caller actually supplied."""
        return self.model_dump(exclude_unset=True)
