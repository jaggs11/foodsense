# Review traceability

One row per numbered section of `docs/review_requirements.md`. This is how the
review panel checks the mandate was met, so it is maintained as a deliverable and
updated at the end of every phase (ruling R12).

Status values: `not started`, `in progress`, `done`, `blocked: needs human`.

`done` means the requirement is satisfied **and** the evidence column names something
that can be opened and checked -- a file, a test name, a command. A requirement is
never marked `done` on the strength of a plan.

Last updated: end of P1.

| # | Requirement | Status | Phase | Evidence |
|---|---|---|---|---|
| 1 | FINAL PROJECT OBJECTIVE | not started | P0-P10 | -- |
| 2 | VERY IMPORTANT — TODDLERS ONLY | not started | P3 | -- |
| 3 | DISEASE / HEALTH CONDITION SCOPE | not started | P3 | -- |
| 4 | DIABETES EXCLUSION | not started | P3 | -- |
| 5 | EXISTING PROJECT — INSPECT FIRST | done | P0 | docs/inventory_pre_pivot.md -- all 20 items, read from code |
| 6 | CURRENT USDA DATASET PROBLEM | in progress | P1 | data/schema.py (multi-source `foods` + 5 tables); migration keeps all 2590 USDA rows, stamped usda/high/verified; tests/test_coverage.py::TestTheDatabaseStoresRealNulls |
| 7 | INDIAN FOOD DATASET — HIGH PRIORITY | blocked: needs human | P2 | data/indian/manifest.yaml is empty -- awaiting real files (human task) |
| 8 | TODDLER-APPROPRIATE FOOD DATA | not started | P2 | -- |
| 9 | UNIFIED FOOD DATA MODEL | in progress | P1 | data/schema.py DDL; data/validation.py::FoodCreate; tests/test_validation.py (51). Toddler metadata table exists, populated in P2 |
| 10 | DATA PROVENANCE | done | P1 | provenance columns NOT NULL on every row + `sources` table (data/schema.py); data/validation.py forces user_added -> pending and requires a reference for dataset sources; tests/test_validation.py::TestProvenance |
| 11 | TODDLER-SPECIFIC DISEASE KNOWLEDGE BASE | blocked: needs human | P3 | knowledge/sources/sources.yaml is empty -- no condition can be enabled without evidence |
| 12 | RAG ARCHITECTURE | not started | P4, P5 | -- |
| 13 | RAG KNOWLEDGE SOURCES | in progress | P2.0 | knowledge/sources/README.md + empty sources.yaml (manifest schema) |
| 14 | RETRIEVAL MUST BE TODDLER-SPECIFIC | not started | P4 | -- |
| 15 | AGE + CONDITION INTERSECTION | not started | P5 | -- |
| 16 | RECOMMENDATION ENGINE | not started | P5 | -- |
| 17 | MEDICAL SAFETY | not started | P5 | -- |
| 18 | TODDLER FOOD SAFETY | not started | P3, P5 | -- |
| 19 | DYNAMIC FOOD ADDITION | not started | P6 | -- |
| 20 | USER-ADDED FOOD → RAG PIPELINE | not started | P6 | -- |
| 21 | FOOD UPDATE | not started | P6 | -- |
| 22 | DUPLICATE DETECTION | done | P1 | data/normalize.py::name_key + find_similar (reuses the Stage-4 scorer); unique index idx_foods_name_key_active enforces it in the DB; tests/test_normalize.py (28), incl. the three Masala Dosa spellings and 7 must-not-match pairs |
| 23 | DATA VALIDATION | done | P1 | data/validation.py::FoodCreate/FoodUpdate, extra=forbid, nutrients >=0 or None, serving_size_g>0; shared by ingestion and API; tests/test_validation.py (51) |
| 24 | FOOD SEARCH | not started | P6 | -- |
| 25 | FOOD FILTERING | not started | P6 | -- |
| 26 | FOOD COMPARISON | not started | P7 | -- |
| 27 | NORMALIZATION | in progress | P1, P7 | data/units.py::NutrientBasis + require_same_basis() raises on a mixed comparison; canonical storage per_100g_edible_portion; tests/test_units.py (22). Chart-side basis labelling is P7 |
| 28 | RESEARCH-QUALITY GRAPH / VISUALIZATION | not started | P7 | -- |
| 29 | RESEARCH VISUALIZATION EXAMPLE | not started | P7 | -- |
| 30 | RESEARCH ANALYSIS | not started | P7 | -- |
| 31 | RAG EVALUATION | not started | P9 | -- |
| 32 | RAG TRANSPARENCY / EXPLAINABILITY | not started | P5 | -- |
| 33 | MODERN UI/UX | not started | P8 | -- |
| 34 | RECOMMENDATION USER FLOW | not started | P8 | -- |
| 35 | MOBILE RESPONSIVENESS | not started | P8 | -- |
| 36 | ERROR HANDLING | not started | P8, P10 | -- |
| 37 | PERFORMANCE | not started | P10 | -- |
| 38 | SECURITY | not started | P6 | -- |
| 39 | IMPORTANT DATA INTEGRITY RULES | in progress | P0-P10 | R13: fillna(0.0) removed, NULLs stored, one named zero_filled_vector() bridge (data/coverage.py); measured scale + archived-regime note in docs/evaluation.md; scripts/hooks/commit-msg; results/README.md |
| 40 | END-TO-END EXAMPLE | not started | P5 | -- |
| 41 | ANOTHER EXAMPLE | not started | P5 | -- |
| 42 | NEW FOOD EXAMPLE | not started | P6 | -- |
| 43 | ARCHITECTURAL PRINCIPLE | in progress | P1, P4 | structured half built: data/schema.py, data/repository.py, docs/architecture.md section 7. Vector half is P4 |
| 44 | DO NOT TURN IT INTO A GENERIC CHATBOT | not started | P5 | -- |
| 45 | FINAL RESEARCH POSITIONING | in progress | P0, P10 | README.md front door; docs/pivot.md 'What changed' |
| 46 | IMPLEMENTATION ORDER | in progress | P0-P10 | docs/pivot.md roadmap, mapped to mandate phases |
| 47 | TESTING REQUIREMENTS | not started | P10 | -- |
| 48 | FINAL UI REQUIREMENT | not started | P8 | -- |
| 49 | FINAL IMPLEMENTATION REPORT | not started | P10 | -- |
| 50 | FINAL NON-NEGOTIABLE RULES | not started | P0-P10 | -- |

## Summary after P1

- **done**: 4
- **in progress**: 8
- **blocked: needs human**: 2
- **not started**: 36

The two blocked rows are blocked on the same thing: a human placing real files and
filling `data/indian/manifest.yaml` and `knowledge/sources/sources.yaml`. Nothing is
ingested that is not listed in a manifest, and no manifest entry is written for a
document that has not been read from disk (R9), so neither can be unblocked by the
agent. Section 11 in particular gates P3 entirely: a condition cannot be enabled
without at least one source behind every rule it scores.

Sections marked `in progress` after P1 have their data layer built and are waiting on
a later phase for the half a user can see -- section 27's basis labelling lands with
the charts in P7, section 43's vector store in P4, section 9's toddler metadata in P2.
