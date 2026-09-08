# Inventory before the repositioning

State of the repository at `6928d39`, the commit tagged `v1-multiage-counterfactual`.
Written by reading the code, not by paraphrasing `docs/architecture.md`.

Covers every item in section 5 of `docs/review_requirements.md`. For each: where it
lives, what it does, and what the repositioning does to it.

Two conventions. "Survives" means the component stays and is re-pointed at the new
domain. "Retired" means it is removed from `main` in P3 or P9 and remains reachable
from the tag.

---

## 1. Project directory structure

`src/foodsense/` is an installed package (`pyproject.toml`, `pip install -e .`), laid
out one directory per pipeline stage: `stage1_prediction/`, `stage2_optimizer/`,
`stage3_rag/`, `stage4_verification/`, plus `constraints/` (the rule engine and its
configs), `data/` (database build and access), `api/`, and three top-level modules —
`schemas.py` (every pydantic model), `pipeline.py` (the stage orchestrator),
`cli.py`. Around 15,780 lines of Python including tests; `data/build_food_db.py` (867)
and `data/corpora.py` (735) are the largest single files.

Supporting directories: `configs/` (YAML, no code), `experiments/` (five evaluation
scripts), `frontend/` (React), `scripts/`, `tests/` (543 tests before P0),
`docs/`, `results/`, `models/`, `data/`.

**Repositioning:** structure is kept exactly (ruling R1). P1 adds `data/repository.py`,
`data/units.py`, `data/normalize.py`, `data/validation.py`. P4 adds a vector-index
module under `stage3_rag/`. Nothing is rewritten from scratch.

## 2. Frontend framework

React 18.3 with Vite 5.4 and Tailwind 3.4, in `frontend/`. Five components:
`Stepper.jsx`, `MealDiff.jsx`, `Metrics.jsx`, `Verification.jsx`,
`CustomBuilder.jsx`, assembled by `App.jsx`; `api.js` is the fetch layer. Tested
with Vitest and Testing Library (`App.test.jsx`). Built to `frontend/dist/` and
served by the API from a single origin, so the demo needs one process, not two.
No component library, no state-management library, no chart library.

**Repositioning:** framework and build stay (requirements section 5: do not
introduce a new framework). P8 rebuilds the flow into the five steps of section 34
and adds a design system, empty/loading/error/success states, and the 320–1440
breakpoints of section 35. `CustomBuilder` is the closest existing thing to the new
step-1/step-2 input and is the natural starting point.

## 3. Backend framework

FastAPI, in `src/foodsense/api/main.py` (327 lines) with `api/models.py` (132) for
request and response shapes. Five endpoints (item 18). Two global exception handlers
convert `HTTPException` and unhandled exceptions into a stable JSON error shape so a
stack trace never reaches the browser. Uvicorn serves it; `foodsense serve` and
`make serve` both start it.

**Repositioning:** survives. P6 adds the food search/filter/add/update/compare
endpoints and the `pending`/`verified`/`rejected` flow behind a single admin token.

## 4. Database

SQLite plus a Parquet mirror, at `data/processed/food_db.{sqlite,parquet}`, built by
`data/build_food_db.py` and read by `data/fdc.py`. One table, `foods`: **2,590 rows,
42 columns** — nine metadata columns (`fdc_id`, `name`, `category`, `fdc_category`,
`data_type`, `hazard_class`, `default_form`, `allowed_forms`, `tags`) and **33
nutrient columns** stored per 100 g. A second table, `build_info`, is a key/value
record of how the build ran.

`FoodDB` loads the whole table into memory once (Parquet preferred — it "loads in
milliseconds"), then every lookup is a dict hit. There is no ORM and no connection
pool; the database is read-only at runtime.

**Repositioning:** the wide nutrient columns stay wide and stay per-100 g — the
Stage-1 surrogate reads them as a vector and that loop must stay vectorised — but
become **nullable** (P1.1). Seven new tables join `foods`: `food_aliases`,
`food_nutrients_extra`, `food_toddler_meta`, `recipe_derivations`, `sources`.
The read-only-at-runtime assumption ends in P6, when user-added foods make it a
read-write store.

## 5. Existing USDA integration

`data/fdc.py` (499 lines) is the access layer; `data/build_food_db.py` (867) is the
build. The build pulls USDA FoodData Central, filters to a curated subset, maps FDC
nutrient ids to the 33 project nutrient keys, and attaches project-specific columns
(`hazard_class`, `allowed_forms`, `default_form`, `tags`) that USDA does not supply.
`fdc_id` is the primary key and is a string.

`FoodDB` also carries the fuzzy matcher (`search`, `match`) used by Stage 4:
token-based, with stemming, position weighting and preparation-qualifier handling.
It is deliberately a different algorithm from the Stage-3 BM25 retriever, so Stage 4
does not grade the retriever's own notion of similarity.

**Repositioning:** USDA stays and is never deleted (requirements section 6). Every
row is migrated in P1.2 to `source_type=usda`, `confidence=high`,
`verification_status=verified`, `cuisine=international`. The matcher is reused as-is
for duplicate detection in P1.5 — the brief is explicit that a second fuzzy library
must not be added.

## 6. Existing food schema/model

`FoodRecord` in `data/fdc.py` wraps one row: nutrients as a `NutrientVector`, plus
`permits(form)`, `nutrients_for(quantity_g)` and `as_item(...)`. `schemas.py` (568
lines) holds the pydantic layer: `MealItem` (`food_id`, `name`, `quantity_g`,
`form`), `Meal`, `NutrientVector` (the 33 keys), `UserProfile`, `Violation`,
`RuleEvaluation`, `ItemCorrection`, `SafetyFix`, `VerificationReport`, `MealDiff`,
`PipelineTrace`. `extra="forbid"` throughout.

There is no provenance on a food record at all: no source, no confidence, no
verification status, no timestamps. Every row is implicitly USDA because every row
is USDA.

**Repositioning:** this absence is the single largest gap against the mandate
(requirements sections 9 and 10, where provenance is called mandatory). P1.1 adds
the provenance columns and the toddler-suitability metadata of section 8.
`MealItem.food_id` stays a string, which is what permits a namespaced key for
non-USDA sources without a type change anywhere.

## 7. Existing RAG implementation

Retrieval exists; it is BM25 over food names and categories, and that is the whole of
it. `stage3_rag/retriever.py` (108 lines) builds a `rank_bm25.BM25Okapi` index over
the 2,590 curated names, and `candidates_for(names, k)` returns real foods with real
`fdc_id`s for the generator to pick from. `stage3_rag/translate.py` (180) assembles
the request; `providers.py` (608) renders it.

There is no evidence corpus. Nothing is retrieved except food names — no guidelines,
no research documents, no condition knowledge. The "augmented generation" half is
real (retrieved candidates constrain the generator); the knowledge half does not
exist yet.

**Repositioning:** this is the largest single build in the roadmap. P4 adds dense
retrieval alongside BM25 with reciprocal-rank fusion and hard metadata filters, over
two document types: food cards and evidence chunks. The BM25 index is kept, not
replaced. The circularity separation from the Stage-4 matcher is preserved.

## 8. Embedding model

**None.** BM25 only, and deliberately so: the module docstring calls it "a deliberate
constraint rather than a simplification — the faculty demo has to work with the Wi-Fi
off."

**Repositioning:** P4 adds a local sentence-embedding model cached under gitignored
`models/embeddings/` by `make setup`. The offline guarantee is preserved by
degradation, not by exception: if the model is absent the retriever falls back to
BM25-only, logs it, and surfaces it in `GET /api/health`. It never raises. The
Stage-3 design rationale already anticipated this ("if the corpus grew to free-text
recipes, revisit and go hybrid"); P4 is that trigger firing, and `docs/pivot.md`
quotes it so the reversal is explained rather than hidden.

## 9. Vector database/vector store

**None.** There is no vector store, no index directory, no embedding persistence.

**Repositioning:** P4, behind a `VectorIndex` interface (`upsert`, `delete`, `query`,
`verify`). Preference is `sqlite-vec` inside the existing SQLite file, so a food row
and its vector commit in one transaction and cannot drift apart; ChromaDB under
gitignored `data/index/` is the fallback if the extension will not load.
`make verify-index` asserts one-to-one correspondence between active rows and
vectors. Requirements sections 19–21 make that synchronisation a hard requirement,
which is why the single-file option is preferred.

## 10. Retrieval pipeline

`translate()` computes the diff between planned and optimised meals, takes the names
of the **edited** foods only, and retrieves `k=5` candidates for those. Retrieving
over edits rather than over the whole meal is deliberate: the edited foods are the
ones the write-up has to name and the ones a generator would otherwise invent.

There is no reranking, no metadata filtering, no query construction beyond the food
names, and no notion of a population or a condition anywhere in retrieval.

**Repositioning:** P4 and P5. Requirements section 14 is emphatic that retrieval must
be toddler-specific rather than generic, so population, condition tags, cuisine,
document type and verification status become hard pre-ranking filters.

## 11. LLM/API integration

`stage3_rag/providers.py` (608 lines). Four providers behind one `LLMProvider`
interface: `TemplateProvider` (deterministic, offline, the default),
`AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`. The Anthropic provider
pins its model in `configs/pipeline.yaml` and handles the post-Opus-4.6 constraint
that `temperature` must be 1.0. `build_prompt` and `parse_response` are shared, so
provider output is parsed into the same structure whatever produced it.

The offline-first rule is architectural, not aspirational: the template provider is
the default and the pipeline never requires a key or a network.

**Repositioning:** survives unchanged in structure. P5 adds citation validation on
top: an LLM provider's JSON is checked so that every cited chunk id is in the
retrieved set and every claimed nutrient figure matches the recomputed value within
tolerance. Failing either, the output is discarded and the template renders. The
template provider cites by construction and so cannot fail this check.

## 12. Recommendation logic

There is no recommendation engine in the "suggest foods" sense. Stage 2
(`stage2_optimizer/`, four modules) is a **counterfactual editor**: differential
evolution over `(quantity, form)` for each food in `planned_meal ∪ pantry`, minimising
an objective of suitability shortfall + L1 distance + sparsity + form-change cost +
hard-safety penalty. `space.py` builds the decision variables, and the availability
guarantee is structural — a food the user does not have has no variable, so it
cannot be proposed. `baselines.py` (635 lines) holds DiCE, a Wachter-style ablation
and a greedy heuristic for the comparison table.

**Repositioning:** this machinery is not removed; it is re-pointed. P5 adds
`mode=recommend` where the decision variables are built from the **retrieved
candidate pool** instead of the pantry, and the objective becomes condition-target
shortfall + sparsity + toddler-portion realism + hard safety. `mode=fix_meal` keeps
today's behaviour. Requirements sections 43 and 44 require that the LLM not be the
only intelligence in the system; Stage 2 is that intelligence, which is why it
survives the pivot rather than being replaced by ranking.

## 13. Existing disease logic

There is none. What exists is **goal** logic: `configs/goals/` holds
`glycemic_control`, `weight_management` and `balanced_nutrition` as YAML rule sets,
loaded by `constraints/goals.py` (404 lines) into `Rule` objects with thresholds,
severities and weights. Health state is modelled separately and thinly, as
`UserProfile.health_flags` (an enum: dysphagia, MAOI, hypertension and similar),
which drives food exclusions and texture limits through `constraints/age_rules.py`.

`configs/age_groups/` holds `toddler`, `adult`, `older_adult`, each with nutrient
targets, choking-hazard bans as `(hazard_class, form)` pairs, texture notes and a
nearest-safe-form mapping. The toddler config already defines 12–36 months and
already cites the NASEM DRI 1–3 year band.

**Repositioning:** `configs/goals/` is retired in P3 and replaced by
`configs/conditions/`. `glycemic_control` is not carried forward (requirements
sections 3 and 4). `adult` and `older_adult` are retired. The toddler config's
existing 12–36 month definition is reused rather than re-derived, which is what
requirements section 2 asks for ("use the project's existing definition"). Health
flags generalise into cross-cutting restrictions (allergy, vegetarian status).

## 14. Nutritional calculations

Three layers. `FoodDB.nutrients_for(meal)` sums per-100 g values times quantity into
a `NutrientVector`. `constraints/goals.py: meal_metrics()` derives the quantities
rules are written against (energy, protein, sodium, macronutrient shares, an
estimated glycemic load). `constraints/engine.py: RuleEngine.evaluate()` scores those
against thresholds with a soft-margin satisfaction function, producing `soft_score`
(the numeric guidelines) and `score` (`soft_score` driven to zero by any hard
violation).

**A missing nutrient is currently indistinguishable from zero.**
`build_food_db.py:740` runs `pd.to_numeric(..., errors="coerce").fillna(0.0)` across
every nutrient column, so a nutrient USDA does not report for a food is stored as
`0.0` and every downstream layer reads it as a measured zero. That is invisible today
because the corpus is single-source and largely complete.

**Repositioning:** this becomes the central correctness problem of P1. Multi-source
data has real gaps, and ruling R8 forbids treating a missing value as zero anywhere.
P1.3 makes the columns nullable, adds a `NutrientCoverage` helper that every consumer
of a nutrient vector uses, and either removes each coercion site or makes it emit a
missingness mask alongside. Stage 4 then reports `undetermined` rather than `pass`
for a rule whose nutrient the source does not carry.

## 15. Existing charts

None in the application. Two experiment scripts produce matplotlib figures for the
paper — `experiments/run_cf_eval.py` and `run_lambda_sweep.py`, output to
`results/figures/`. The React frontend renders numbers in tables and cards and draws
no charts at all; there is no chart library in `package.json`.

**Repositioning:** P7 builds the comparison charts of requirements sections 26–29:
per-100 g default with a per-serving option, basis and source stated on every figure,
grouped bar and nutrient-profile charts, radar only where axes are normalised, PNG
and SVG export. The matplotlib figure scripts are kept for the paper.

## 16. Existing UI components

`Stepper.jsx` (linear step navigation), `MealDiff.jsx` (the before/after edit list),
`Metrics.jsx` (nutrient and score readouts), `Verification.jsx` (the Stage-4 report:
corrections, safety fixes, final pass), `CustomBuilder.jsx` (manual case
construction). Tailwind utility classes inline; no design tokens, no shared primitives,
no empty/loading/error state components.

**Repositioning:** P8. `Verification.jsx` is the component that must carry the exact
sentence requirements section 18 mandates for unknown safety attributes —
"Insufficient evidence available for this safety attribute." — so it is where the
honesty requirement becomes visible to a user.

## 17. Authentication

**None**, and deliberately. `api/main.py` states it: "there is no auth, and there
should not be, because there is nothing here to authenticate against." API keys are
read from the server's own environment and never reach the browser. The service is
read-only and stateless.

**Repositioning:** that reasoning stops holding in P6, when the API gains write
endpoints and the database stops being read-only. Requirements section 38 asks for
`pending`/`verified`/`rejected` states rather than for user accounts, so the plan is a
single admin token guarding the write and moderation endpoints — not a new auth
framework, which section 5 warns against.

## 18. API routes

Five, all under `/api`:

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | GET | liveness, model presence, provider availability |
| `/api/scenarios` | GET | the three demo scenarios |
| `/api/providers` | GET | which Stage-3 providers are configured |
| `/api/foods?q=` | GET | food search for the builder |
| `/api/recommend` | POST | the full four-stage pipeline; returns a `PipelineTrace` |

Plus `GET /` serving the built frontend. The `PipelineTrace` is the sole contract
between backend and frontend — every stage's inputs, outputs and diagnostics travel
in one object.

**Repositioning:** all five survive. `/api/health` gains the embedding-model and
vector-index status in P4. `/api/foods` moves behind `FoodRepository` in P1.7 and
gains filters in P6. `/api/recommend` gains `mode` in P5, and rejects ages outside
12–36 months with an explicit out-of-scope message (ruling R2).

## 19. Environment configuration

`.env.example` documents seven optional variables: `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `KAGGLE_USERNAME`, `KAGGLE_KEY`.
Nothing is required — the file says so in its first line and the pipeline honours it.
Non-secret configuration lives in `configs/pipeline.yaml` (stage parameters, model
pin, rule-engine scoring constants) and the YAML under `configs/age_groups/` and
`configs/goals/`.

**Repositioning:** unchanged in shape. P3 replaces the `configs/goals/` half.
P6 adds the admin token as a new optional variable.

## 20. Deployment configuration

Multi-stage `Dockerfile` (Node 20 builds the frontend, Python serves it via uvicorn)
and `docker-compose.yml` passing every LLM variable through with empty defaults so the
container runs offline. CI is `.github/workflows/ci.yml`: ruff check, ruff format
`--check`, and `pytest -m "not slow"` on Python 3.11 and 3.12, plus an npm job that
builds and tests the frontend.

**Repositioning:** survives. P1.8 makes the schema migration run in a clean CI
environment. P4 needs CI to work without the embedding model present, so a
deterministic stub embedder tests the synchronisation logic and retrieval-quality
tests are reported as skipped rather than silently passing. P10 verifies both.

---

## What this inventory changes about the plan

Three findings worth the architect's attention.

**The `fillna(0.0)` at `build_food_db.py:740` is load-bearing.** Every consumer of a
nutrient vector currently assumes completeness. P1.3 is not a defensive tidy-up; it
is a change to an invariant the rule engine, the surrogate's feature builder and the
Stage-4 comparison all rely on. It should be done before any second source lands,
not after.

**There is no disease logic to extend — only goal logic to replace.** Requirements
section 11 asks for a condition knowledge layer with pediatric considerations,
contraindications and evidence references. `configs/goals/` has thresholds and
weights and nothing else. P3 is a build, not an edit, and the schema in
`configs/conditions/README.md` reflects that.

**Retrieval has no knowledge corpus at all.** "RAG" today means retrieving food names
to constrain a generator. Requirements sections 12–14 ask for evidence retrieval over
guidelines and research documents. That corpus does not exist and cannot be created
by the agent — it is acquired by a human placing files and filling
`knowledge/sources/sources.yaml`. Until then P3 cannot enable a condition and P4 has
only food cards to index.
