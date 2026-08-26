"""Request models for the HTTP API.

These are deliberately *not* the pipeline's own types. `PipelineTrace` and the
`StageNResult` family are **output** documents: they carry computed fields
(`is_safe`, `n_corrections`, `succeeded`) that exist to be rendered, and they are
declared `extra="forbid"`. Phase 4 established why that matters -- a model that
accepts its own computed output back would also accept a typo in a request body,
because it has no way to tell the two apart.

So the direction of travel is one-way by construction. The API *accepts* the
models below, and *returns* the trace verbatim. Nothing round-trips.

Two ways to ask for a recommendation, and only two:

``ScenarioRequest``
    A key naming one of the built-in demo cases. What the dropdown sends.
``CustomRequest``
    A full profile, planned meal and pantry. What a faculty member poking at the
    system sends.

Both carry the provider name and the seed, because the demo's determinism claim
is only checkable if the seed is an input the user can see and set.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from foodsense.schemas import Meal, MealItem, UserProfile

__all__ = [
    "CustomRequest",
    "ErrorResponse",
    "FoodSummary",
    "HealthResponse",
    "ProviderInfo",
    "ScenarioRequest",
    "ScenarioSummary",
]

#: Upper bound on items in a submitted meal or pantry. Not a security control --
#: the server binds localhost -- but a request with ten thousand items would
#: build a search space that takes minutes, and a demo should refuse that
#: politely rather than appear to hang.
MAX_ITEMS = 60


class _Request(BaseModel):
    """Shared request behaviour: reject unknown fields, loudly."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        default="template",
        description=(
            "Stage-3 text generator. 'template' is fully offline and needs no key. "
            "An unavailable provider is not an error: the run succeeds with "
            "stage3.fallback_used set."
        ),
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Optimiser seed. Omit for the configured default (42).",
    )


class ScenarioRequest(_Request):
    """Run one of the built-in demo scenarios."""

    scenario: str = Field(min_length=1, max_length=64)


class CustomRequest(_Request):
    """Run an arbitrary profile, meal and pantry."""

    profile: UserProfile
    planned_meal: Meal
    pantry: list[MealItem] = Field(default_factory=list, max_length=MAX_ITEMS)


class ScenarioSummary(BaseModel):
    """One demo scenario, as the dropdown needs it."""

    key: str
    title: str
    description: str
    age_group: str
    goal: str
    health_flags: list[str]
    n_planned_items: int
    n_pantry_items: int


class ProviderInfo(BaseModel):
    """Whether a Stage-3 provider can be used right now, and if not why not.

    The UI greys out unavailable providers and shows ``reason`` beside them, so a
    missing key reads as a missing key rather than as a broken demo.
    """

    name: str
    available: bool
    reason: str = ""
    is_default: bool = False


class FoodSummary(BaseModel):
    """One row of the food database, for autocomplete."""

    food_id: str
    name: str
    category: str
    energy_kcal_per_100g: float
    allowed_forms: list[str]
    hazard_class: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    n_foods: int
    default_provider: str


class ErrorResponse(BaseModel):
    """The one error shape the API emits."""

    error: str
    detail: str
