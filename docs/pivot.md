# The repositioning

How FoodSense changes after the project review, and the rules governing how it
changes.

`docs/review_requirements.md` is the mandate. It governs **what must exist**. This
document governs **how it is built**. Where they appear to conflict, the mandate wins
on outcomes and the rulings below win on architecture.

Status: in progress. `docs/review_traceability.md` tracks all fifty requirement
sections.

---

## What changed

FoodSense **was**: availability-aware, verification-guided counterfactual meal editing
across three age groups and three goals, over a curated USDA subset.

FoodSense **is now**: an evidence-grounded, retrieval-augmented, verification-guided
food recommendation system for toddlers (12–36 months) with nutrition-related health
conditions, over a multi-source food knowledge base in which Indian regional food data
is primary and USDA is supplementary.

The four stages persist. Each gets a domain swap, not a replacement.

| Stage | Was | Becomes |
|---|---|---|
| 1 — Prediction | LightGBM surrogate over nutrient vector + (age group, goal, flags) one-hots | Same surrogate over nutrient vector + (condition, restriction) one-hots, toddler-only, **missingness-aware** (multi-source data has nulls) |
| 2 — Optimisation | DE over (quantity, form) for planned meal ∪ pantry; minimal-edit objective | Same DE. **Recommend mode** (new default): decision variables built from the retrieved candidate pool; objective = condition-target shortfall + sparsity + toddler-portion realism + hard safety. **Fix-my-meal mode** (existing): unchanged |
| 3 — Retrieval + translation | BM25 over USDA names → provider (template default) | **Hybrid** BM25 + dense retrieval over food documents *and* condition-evidence documents, metadata-filtered on population/condition/cuisine → provider, with **citation validation** against retrieved chunk ids |
| 4 — Verification | USDA re-grounding, ±10% correction, rule-engine re-scan, two-pass repair | Same five steps against **whichever source** a matched food came from; condition/allergen/celiac scan added; nutrients the source lacks are flagged **unverifiable**, never silently passed |

The panel wrote its pipeline as retrieve → rank → generate. Every listed requirement
will be satisfied, and the optimiser and rule engine will stay between retrieval and
generation, because the mandate itself (sections 43 and 44) says the LLM must not be
the only intelligence in the system. Stage 2 is that intelligence.

**This is the single most important thing to understand about the pivot: the
counterfactual machinery is not being removed, it is being pointed at a new problem.**

---

## Binding rulings

**R1 — Repositioning, not rebuild.** `src/foodsense/` keeps its four stage packages,
`constraints/`, `schemas.py`, `pipeline.py`, `cli.py`, `api/`. Nothing is rewritten
from scratch. Retired code (adult/older-adult profiles, goal configs, multi-age
evaluation) is removed from `main` in P3/P9 and preserved under the git tag created in
P0.1.

**R2 — Population.** Toddler only, defined as 12–36 months, matching the existing
`configs/age_groups/toddler` definition and the NASEM DRI 1–3 year band already cited
there. `age_months` remains a first-class input (safety rules use it). Ages outside
12–36 months are rejected at the API boundary with an explicit out-of-scope message;
infants (<12 months) are not a target population.

**R3 — Goals become conditions.** `configs/goals/` is retired in P3 and replaced by
`configs/conditions/`. Target set: iron-deficiency anemia; undernutrition
(mild/moderate, guidance-level); vitamin D deficiency (with calcium co-monitoring);
functional constipation; celiac disease. Food allergies and vegetarian/non-vegetarian
are **cross-cutting restrictions**, not conditions. A condition is `enabled: true`
only when every rule it scores cites a source present in
`knowledge/sources/sources.yaml`. A condition with no sources exists as a skeleton,
`enabled: false`, hidden from the UI. `glycemic_control` is not carried forward as a
condition. The architecture must allow a new condition to be added as one YAML file
plus one manifest entry, with no code change.

**R4 — Two modes, one pipeline.** `run_pipeline` accepts `mode=recommend` (candidate
pool from retrieval) or `mode=fix_meal` (planned meal + pantry, existing behaviour).
Both produce a `PipelineTrace`. The structural availability guarantee
(`stage2_optimizer/space.py` builds decision variables only from the supplied pool)
holds in both modes: in recommend mode the pool is the retrieved, filtered candidate
set.

**R5 — Multi-source food knowledge base.** USDA stays. Every food row carries
`source_type`, `source_name`, `source_reference`, `source_id`, `confidence`,
`verification_status`. Permitted source types: `usda`, `indian_dataset`,
`research_dataset`, `government`, `derived_recipe`, `verified`, `user_added`. **No
nutritional value is ever fabricated.** A dish whose nutrients are computed from
ingredient composition using a cited standard recipe is permitted as
`derived_recipe`, and the derivation (ingredients, grams, yield factor, method,
ingredient source rows) is stored so it can be audited. Data whose provenance cannot
be established (e.g. unsourced community datasets) may enter only as
`verification_status=unverified` and is excluded from recommendation by default.

**R6 — Hybrid retrieval.** BM25 stays. Dense retrieval over a local embedding model is
added alongside it, fused by reciprocal rank fusion, with **hard metadata filters**
(population, condition tags, cuisine, document type, verification status) applied
before ranking. The circularity argument is preserved: Stage 4 matching continues to
use its own fuzzy scorer, never the retriever's similarity. The offline claim is
preserved: the embedding model is cached locally under gitignored `models/embeddings/`
by `make setup`; if absent, the retriever degrades to BM25-only, logs it, and surfaces
it in `GET /api/health`. It never raises.

*On explaining the reversal.* Ruling R6 as issued attributed a hybrid-retrieval
rationale to `docs/architecture.md`, reading "if the corpus grew to free-text recipes,
revisit and go hybrid". **That sentence has never existed in this repository at any
commit** — not on `main`, not at `phase-4.5`, not at `v1-multiage-counterfactual`; the
words "hybrid" and "revisit" appear nowhere in the tracked tree. The architect has
since confirmed the sentence was his own, written from memory. Nothing is missing from
`docs/architecture.md`, and no earlier plan was lost.

It is not quoted here, because quoting it would mean manufacturing a citation, which
sections 13 and 39 of the mandate forbid — and which would be a poor way to begin a
document about not hiding things.

The consequence is worth stating plainly, because it changes what this reversal *is*.
The project never planned to go hybrid and then deferred it. Going hybrid is a
decision being taken now, and it has to stand on its merits rather than on a
back-reference.

Those merits are real and sufficient on their own: the corpus is changing. Sections 12
to 14 of the mandate require retrieving pediatric guidelines and research documents,
which are free text, and BM25 alone over free-text evidence is weaker than BM25 fused
with dense retrieval.

What the repository actually argues is narrower, and both arguments are constraints
the new design must keep meeting rather than objections to it.
`stage3_rag/retriever.py` justifies BM25-only on offline grounds:

> BM25 over names and categories, via `rank_bm25`, entirely local: no embedding model,
> no network, no API key. That is a deliberate constraint rather than a simplification
> — the faculty demo has to work with the Wi-Fi off.

and `docs/architecture.md` justifies keeping retrieval and matching distinct:

> Retrieval (BM25 over names, categories and tags) and Stage-4 matching are
> deliberately different algorithms. BM25 answers "which foods are plausibly related
> to this phrase"; the matcher in `data/fdc.py` answers "is this specific name the
> same food as one we hold". Using one for both would make verification partly
> circular — grading the retriever against the retriever's own notion of similarity.

Neither is a promise never to go hybrid. R6 honours the first by degrading to
BM25-only when the model is absent rather than by requiring it, and the second by
leaving Stage 4's matcher alone.

**R7 — Vector store.** Preference order: `sqlite-vec` inside the existing SQLite
database (one file, one transaction covers the food row and its vector,
synchronisation is atomic by construction); if the extension cannot be loaded on the
development machine, ChromaDB persistent client under gitignored `data/index/`. Either
way it sits behind a `VectorIndex` interface (`upsert(id, text, metadata)`,
`delete(id)`, `query(text, k, filters)`, `verify()`), and `make verify-index` asserts a
one-to-one correspondence between active food rows and vectors. For CI, a
deterministic stub embedder tests the sync logic without a model download; retrieval-
quality tests run only when the real model is present and are reported as skipped
otherwise, never silently green.

**R8 — Verification is multi-source and honest about gaps.** Stage 4 recomputes
nutrients from the matched row in whichever source it came from. If the source lacks a
nutrient a rule depends on, the rule's result for that item is `undetermined`, not
`pass`; the trace records `unverifiable_nutrients` per item; the UI shows the
mandate's exact sentence, "Insufficient evidence available for this safety attribute,"
for safety attributes that are unknown. A missing value is **never** treated as zero
anywhere in the pipeline.

**R9 — Citations only from ingested sources.** The explanation cites evidence chunks
by id. Chunks come only from documents listed in `knowledge/sources/sources.yaml` that
physically exist in `knowledge/sources/`. The template provider cites by construction.
LLM providers' JSON is validated: every cited chunk id must be in the retrieved set and
every claimed nutrient figure must match the recomputed value within tolerance, else
the provider output is discarded and the template renders. You never author a manifest
entry for a document you have not read from disk. You never write a URL, paper title,
author, or year that does not come from a manifest entry.

**R10 — Evaluation.** The 300-case multi-age comparison, the λ₁ sweep, validity
decomposition, and surrogate-boundary study are retired with the population they
measured. They are archived, not deleted (P0.5), and re-run on the toddler/condition
harness in P9. The new primary ablation ladder is: LLM-only (no retrieval) → retrieval
+ rank (no optimiser) → retrieval + optimiser (no verification) → full pipeline. DiCE
baselines remain only in the fix-my-meal ablation, under the existing budget cap; the
`dice_genetic` runtime problem is thereby closed. No evaluation number appears in the
repository until the script that produces it has run.

**R11 — Git and repository hygiene.** Nothing pushed may contain attribution to an AI
coding assistant: no `Co-Authored-By` trailers naming one, no "Generated with …"
lines, no assistant or tool names in commit messages, PR text, README, docs, comments,
or code. If your own settings expose an option to disable commit co-author
attribution, turn it off; do not rely on it alone. Install a `commit-msg` hook that
rejects these patterns. Any assistant memory or configuration files in the working
tree (an uppercase memory markdown at the repository root, a hidden assistant config
directory) stay local: untracked and gitignored. Commit messages follow
conventional-commit style (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
Refer to the API provider integration as "the Anthropic provider" when a commit
touches it.

**R12 — Review traceability.** `docs/review_traceability.md` has one row per numbered
section of `docs/review_requirements.md` (§1–§50): requirement summary, status
(`not started` / `in progress` / `done` / `blocked: needs human`), owning phase,
evidence (file paths, test names). Updated at the end of every phase. This document is
how the panel will check the mandate was met; it is a deliverable.

**R13 — True missingness is stored now, consumed at P3.** `build_food_db.py` applied
`fillna(0.0)` across every nutrient column, so the database asserted a measured zero
wherever USDA reported nothing. That is an R8 violation, it sits underneath Stage 1's
training labels and Stage 4's verification claims, and it cannot be fixed in one step
without moving every number the project has published — which collides with P1.7's
requirement that the golden traces pass unmodified, and would make a migration bug
indistinguishable from an intended change. So it is split:

* **P1 — the database stops lying.** Nutrients a source does not report are stored
  `NULL`. `data/coverage.NutrientCoverage` reports what is genuinely present, and
  `FoodRecord.reported_nutrients` carries the mask alongside every vector.
* **P1 — the pipeline keeps answering identically.** Every consumer reads through one
  named, documented function, `data/coverage.zero_filled_vector()`, which applies the
  historical zero-fill and says in its own docstring that it preserves
  pre-repositioning behaviour and is retired in P3. No call site re-implements it.
  Golden traces and demo-scenario outputs are byte-identical.
* **P3 — consumers switch to honest missingness**, when the surrogate is retrained
  toddler-only and condition-conditioned and every result is being regenerated anyway.

The point of the split is auditability: the moment the pipeline's answers change is
one dated, deliberate commit rather than a side effect of a schema migration. The
measured scale of the fabrication, and the bound on how much P3 can move, are in
`docs/evaluation.md`. Archived results stay exactly as produced; the note there
records which regime produced them.

---

## Roadmap

Each phase ends with a phase report and waits for approval. Bracketed numbers map to
the mandate's own phase numbering (section 46).

- **P0 — Freeze, hygiene, inventory.** Tag current `main`; git attribution hygiene;
  full inspection per section 5; land the two Phase 4.5 correctness fixes; archive
  results; scaffold data and knowledge directories; README repositioned with an
  in-progress banner; traceability skeleton. [Phase 1]
- **P1 — Food knowledge base data model.** Multi-source, provenance-bearing schema;
  migration of the 2,590 USDA rows; unit normalisation; null semantics; duplicate
  detection; validation; `FoodRepository` read path with unchanged behaviour. [Phase 2]
- **P2 — Indian food ingestion.** Loaders for the manifest-listed Indian datasets;
  recipe-derivation module with audit trail; aliases; toddler metadata where sources
  support it; coverage report. Blocked until `data/indian/manifest.yaml` lists real
  files. [Phase 3]
- **P3 — Conditions, restrictions, surrogate retrain.** `configs/conditions/`; rule
  engine extended (allergen, gluten, unknown-hazard handling); adult/older-adult and
  goals retired from `main`; Stage 1 retrained toddler-only, condition-conditioned,
  missingness-aware; new golden scenarios. Blocked until
  `knowledge/sources/sources.yaml` lists at least one source per condition to be
  enabled. [Phases 4, 6]
- **P4 — Hybrid retrieval and indexing.** Embedding model; `VectorIndex`; food-card
  and evidence-chunk document builders; metadata filtering; RRF fusion; auto-embed on
  create/update/delete; `verify-index`. [Phases 5, 8]
- **P5 — Recommendation engine.** Recommend mode in Stage 2; candidate filtering
  (cuisine, veg status, allergens, celiac, required-nutrient coverage);
  portion-realism term; Stage 3 explanation with citation validation; Stage 4
  multi-source verification; insufficient-evidence path that returns a message, not a
  recommendation. [Phase 6]
- **P6 — Dynamic food addition and API.** Search/filter/add/update/compare endpoints;
  validation; duplicate warning flow; `pending`/`verified`/`rejected` states with a
  single admin token (no new auth framework); persistence proven across restart.
  [Phases 7, 13]
- **P7 — Comparison and research charts.** Per-100g default with per-serving option;
  basis and source stated on every figure; grouped bar, nutrient profile, radar only
  where axes are normalised; PNG/SVG export in the UI; matplotlib figure scripts for
  the paper. [Phases 9, 10]
- **P8 — UI/UX and responsiveness.** Five-step flow; design system;
  empty/loading/error/success states; breakpoints 320–1440; touch targets.
  [Phases 11, 12]
- **P9 — Evaluation framework and results.** Retrieval metrics (P@K, R@K, MRR), RAG
  metrics (context relevance, faithfulness via citation check), recommendation metrics
  (constraint satisfaction, evidence-grounded rate); labelled-set template; new
  ablation ladder; re-run of retired studies on the toddler harness; manifest
  regenerated. [Phase 14]
- **P10 — Hardening, end-to-end tests (section 47, T1–T14), final implementation
  report (section 49), Docker/CI verification.** [Phase 15]
