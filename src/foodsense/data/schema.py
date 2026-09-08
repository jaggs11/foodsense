"""The food knowledge base schema, and the provenance defaults for migrated rows.

One place that knows what the database looks like, so the build, the migration and
the repository cannot disagree about it.

## What changed and why

Before the repositioning this was one table of 2,590 USDA rows with 33 nutrient
columns. Every row was USDA, so nothing recorded that it was; every nutrient was
present, so nothing recorded whether it was.

Both assumptions stop holding as soon as a second source lands, and requirements
sections 9 and 10 make provenance mandatory rather than optional. So `foods` grows
provenance and toddler-relevant descriptive columns, four tables join it, and the
nutrient columns become **nullable**.

## Why the nutrient columns stay wide

They could have become an entity-attribute-value table, which is the textbook answer
for sparse optional attributes. They are not, because Stage 1 reads all 33 as a
vector for every candidate in every generation of the optimiser's search, and that
loop is the pipeline's hot path. A join-and-pivot per candidate would cost more than
the entire rest of Stage 2. Nutrients outside the 33 go in
``food_nutrients_extra``, where sparsity is real and the access pattern is
occasional.

## Why the id scheme is namespaced rather than a UUID

``<source_type>:<source_id>`` -- ``indian_dataset:IFCT-A017`` -- with USDA rows
keeping their bare numeric ``fdc_id`` so existing ids, golden traces and pinned
demo scenarios keep working.

Two reasons over a UUID. It is a string either way, so ``MealItem.food_id`` needs no
type change, which is what the choice was to be made on. But a UUID is *random*, and
``make data`` must produce byte-identical output across runs -- the project's
determinism standard predates the repositioning and rebuilding the database must not
break it. A namespaced key is a pure function of the source and its own identifier,
so a rebuild reproduces it exactly.

The primary key column is ``id``. ``FoodRecord.fdc_id`` keeps its attribute name for
now: renaming it touches the verifier, the matcher, the scenarios and every golden
trace, and that is a rename to do once the read path is behind ``FoodRepository``,
not while it is being put there.
"""

from __future__ import annotations

from foodsense.schemas import NutrientVector

__all__ = [
    "FOODS_COLUMNS",
    "MIGRATION_TIMESTAMP",
    "NUTRIENT_COLUMNS",
    "USDA_PROVENANCE",
    "ddl_statements",
    "namespaced_id",
]

#: The 33 wide columns, from NutrientVector so the two cannot drift apart.
NUTRIENT_COLUMNS: tuple[str, ...] = tuple(NutrientVector.model_fields)

#: created_at/updated_at for every row migrated from the pre-repositioning
#: database. A fixed constant, not wall-clock: `make data` must produce identical
#: output across runs, and a timestamp taken at build time would differ every time.
#: The value is the date the repositioning began, which is what these rows' history
#: actually is -- they were not created when the build last ran.
MIGRATION_TIMESTAMP = "2026-09-08T00:00:00Z"

#: Provenance stamped on every row carried over from the USDA-only database.
#: Requirements section 6: USDA is not deleted, it becomes one source among several.
USDA_PROVENANCE = {
    "source_type": "usda",
    "source_name": "USDA FoodData Central",
    "confidence": "high",
    "verification_status": "verified",
    "cuisine": "international",
    "is_active": 1,
}


def namespaced_id(source_type: str, source_id: str) -> str:
    """The food id for a non-USDA row.

    USDA rows keep their bare ``fdc_id`` and do not go through this: their ids are
    load-bearing in committed scenarios and golden traces.
    """
    if not source_type or not source_id:
        raise ValueError("both source_type and source_id are required for a namespaced id")
    if ":" in source_type:
        raise ValueError(f"source_type must not contain ':' -- got {source_type!r}")
    return f"{source_type}:{source_id}"


#: Non-nutrient columns of `foods`, in storage order.
FOODS_COLUMNS: tuple[str, ...] = (
    "id",
    "canonical_name",
    "display_name",
    "name_key",
    "category",
    "fdc_category",
    "data_type",
    "cuisine",
    "region",
    "veg_status",
    "hazard_class",
    "default_form",
    "allowed_forms",
    "texture",
    "tags",
    "serving_size_g",
    "serving_unit",
    "serving_description",
    "ingredients",
    "preparation_method",
    "basis",
    "source_type",
    "source_name",
    "source_reference",
    "source_id",
    "confidence",
    "verification_status",
    "is_active",
    "created_at",
    "updated_at",
)


def _foods_ddl() -> str:
    nutrients = ",\n".join(f"    {c} REAL" for c in NUTRIENT_COLUMNS)
    return f"""
CREATE TABLE foods (
    -- Identity. `id` is <source_type>:<source_id> for new sources and a bare
    -- fdc_id for the migrated USDA rows.
    id                  TEXT PRIMARY KEY,
    canonical_name      TEXT NOT NULL,
    display_name        TEXT,
    -- Normalised form of canonical_name (data/normalize.py). Uniquely indexed
    -- among active rows: this is what actually prevents duplicates, because it
    -- is enforced by the database rather than by anyone reading a warning.
    name_key            TEXT NOT NULL,

    category            TEXT,
    fdc_category        TEXT,
    data_type           TEXT,
    cuisine             TEXT NOT NULL DEFAULT 'other',
    region              TEXT,
    veg_status          TEXT NOT NULL DEFAULT 'unknown',

    -- Safety and preparation. hazard_class/allowed_forms/default_form predate the
    -- repositioning and carry the (food, form) modelling the constraint layer
    -- rests on; texture is new and describes the food as served.
    hazard_class        TEXT,
    default_form        TEXT NOT NULL,
    allowed_forms       TEXT NOT NULL,          -- JSON array
    texture             TEXT NOT NULL DEFAULT 'unknown',
    tags                TEXT NOT NULL,          -- JSON array

    serving_size_g      REAL,
    serving_unit        TEXT,
    serving_description TEXT,
    ingredients         TEXT,                   -- JSON array, or NULL if unknown
    preparation_method  TEXT,

    -- Storage is always per 100 g edible portion; this records what the SOURCE
    -- stated, so a conversion can be audited after the fact.
    basis               TEXT NOT NULL DEFAULT 'per_100g_edible_portion',

    -- Provenance. Mandatory (requirements sections 9, 10).
    source_type         TEXT NOT NULL,
    source_name         TEXT,
    source_reference    TEXT,
    source_id           TEXT,
    confidence          TEXT NOT NULL DEFAULT 'unknown',
    verification_status TEXT NOT NULL DEFAULT 'pending',

    is_active           INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,

    -- The 33 canonical nutrients, per 100 g edible portion.
    -- NULLABLE, and NULL means "this source does not report it" -- never zero.
    -- Ruling R8; the distinction is the whole point of the column being nullable.
{nutrients},

    CHECK (cuisine IN ('north_indian','south_indian','pan_indian','international','other')),
    CHECK (veg_status IN ('vegetarian','non_vegetarian','eggetarian','unknown')),
    CHECK (source_type IN ('usda','indian_dataset','research_dataset','government',
                           'derived_recipe','verified','user_added')),
    CHECK (confidence IN ('high','medium','low','unknown')),
    CHECK (verification_status IN ('verified','pending','rejected','unverified')),
    CHECK (serving_size_g IS NULL OR serving_size_g > 0)
)
"""


def ddl_statements() -> tuple[str, ...]:
    """Every CREATE needed for a fresh database, in dependency order."""
    return (
        _foods_ddl(),
        # Alternative names. alias_key is normalised the same way as name_key, so
        # an alias can be matched against a name without a second normal form.
        """
CREATE TABLE food_aliases (
    food_id   TEXT NOT NULL REFERENCES foods(id) ON DELETE CASCADE,
    alias     TEXT NOT NULL,
    alias_key TEXT NOT NULL,
    source    TEXT,
    PRIMARY KEY (food_id, alias_key)
)
""",
        # Nutrients outside the canonical 33. Sparse and rarely read, so the
        # narrow shape that would be wrong for the hot 33 is right here.
        """
CREATE TABLE food_nutrients_extra (
    food_id          TEXT NOT NULL REFERENCES foods(id) ON DELETE CASCADE,
    nutrient_key     TEXT NOT NULL,
    value            REAL,
    unit             TEXT NOT NULL,
    source_reference TEXT,
    PRIMARY KEY (food_id, nutrient_key)
)
""",
        # Toddler suitability (requirements section 8). Every column is nullable
        # or admits 'unknown' on purpose: section 8 says to mark information
        # unknown rather than invent it, and a NOT NULL default here would be an
        # invented safety claim -- the most dangerous kind in this system.
        """
CREATE TABLE food_toddler_meta (
    food_id          TEXT PRIMARY KEY REFERENCES foods(id) ON DELETE CASCADE,
    min_age_months   INTEGER,
    choking_risk     TEXT NOT NULL DEFAULT 'unknown',
    choking_notes    TEXT,
    safe_forms       TEXT,      -- JSON array, or NULL if not established
    allergens        TEXT,      -- JSON array, or NULL meaning UNKNOWN (not "none")
    added_sugar_flag INTEGER,   -- nullable bool: NULL is unknown
    high_sodium_flag INTEGER,   -- nullable bool: NULL is unknown
    source_reference TEXT,
    CHECK (choking_risk IN ('known_hazard','low','unknown')),
    CHECK (added_sugar_flag IS NULL OR added_sugar_flag IN (0,1)),
    CHECK (high_sodium_flag IS NULL OR high_sodium_flag IN (0,1))
)
""",
        # How a derived_recipe row was computed. Stored so the arithmetic can be
        # audited: ruling R5 permits computing a composite dish's nutrients from a
        # cited standard recipe, and this is what makes that different from
        # fabricating them.
        """
CREATE TABLE recipe_derivations (
    food_id                 TEXT PRIMARY KEY REFERENCES foods(id) ON DELETE CASCADE,
    recipe_source_reference TEXT NOT NULL,
    ingredients             TEXT NOT NULL,  -- JSON [{ingredient_food_id, grams}]
    yield_factor            REAL NOT NULL,
    method                  TEXT,
    computed_at             TEXT NOT NULL
)
""",
        # The source registry. One row per source_type actually in use.
        """
CREATE TABLE sources (
    source_type  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    reference    TEXT,
    license_note TEXT,
    CHECK (source_type IN ('usda','indian_dataset','research_dataset','government',
                           'derived_recipe','verified','user_added'))
)
""",
        "CREATE UNIQUE INDEX idx_foods_name_key_active ON foods(name_key) WHERE is_active = 1",
        "CREATE INDEX idx_foods_category ON foods(category)",
        "CREATE INDEX idx_foods_hazard ON foods(hazard_class)",
        "CREATE INDEX idx_foods_cuisine ON foods(cuisine)",
        "CREATE INDEX idx_foods_source_type ON foods(source_type)",
        "CREATE INDEX idx_foods_verification ON foods(verification_status)",
        "CREATE INDEX idx_aliases_key ON food_aliases(alias_key)",
        "CREATE TABLE build_info (key TEXT PRIMARY KEY, value TEXT)",
    )
