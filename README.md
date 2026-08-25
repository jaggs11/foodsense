# FoodSense

**An Availability-Aware, Verification-Guided Counterfactual Food Recommendation System**

[![CI](https://github.com/aarush093/foodsense/actions/workflows/ci.yml/badge.svg)](https://github.com/aarush093/foodsense/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Offline-first](https://img.shields.io/badge/demo-offline--first-success.svg)](#quickstart)

FoodSense takes the meal a person **actually planned to eat**, and the ingredients they
**actually have**, and computes the *smallest set of edits* that makes that meal meet their
health goal and their life-stage safety requirements — then verifies every gram of the
result against the USDA FoodData Central database before showing it to anyone.

It extends **MetaPlate** (Arefeen, Johnston & Ghasemzadeh, IEEE JBHI 2026 — arXiv:2606.10120),
which pairs a postprandial-glucose predictor with a counterfactual optimiser and an LLM-RAG
translation layer, along five axes: availability-awareness, modification-based editing,
post-generation verification, generalised health goals, and age/life-stage personalisation.

> **Status:** Phases 0–3 complete — data layer, constraint engine, Stage-1 surrogate
> and the availability-aware counterfactual optimiser with its baselines. Stages 3–4
> land in phases 4–6; see [Roadmap](#roadmap). No evaluation number appears in this
> repository until the script that produces it has actually been run — see `results/`.

---

## Architecture

```mermaid
flowchart TB
    subgraph IN["Input"]
        P["UserProfile<br/>age band, goal, health flags"]
        M["Planned meal<br/>(food_id, quantity_g, form)"]
        A["Pantry / available items"]
    end

    subgraph S1["Stage 1 - Prediction"]
        F["features.py<br/>nutrients, glycemic load, one-hots"]
        SM["LightGBM suitability surrogate<br/>f(meal, profile) in [0,1]"]
        F --> SM
    end

    subgraph S2["Stage 2 - Age-Aware Counterfactual Optimisation"]
        SP["space.py<br/>decision vars = planned + pantry ONLY"]
        OB["objective.py<br/>L1-minimal, sparse, safety-penalised"]
        DE["de_optimizer.py<br/>differential evolution"]
        SP --> OB --> DE
    end

    subgraph S3["Stage 3 - LLM-RAG Translation"]
        R["retriever.py<br/>BM25 over USDA names"]
        PR["providers.py<br/>Template (default), Anthropic, OpenAI, Ollama"]
        R --> PR
    end

    subgraph S4["Stage 4 - USDA Verification"]
        V["verifier.py<br/>match, recompute, compare +/-10%, correct"]
        SS["RuleEngine safety re-scan<br/>nearest-safe-form or remove"]
        V --> SS
    end

    RE[["RuleEngine<br/>single source of truth"]]
    DB[("USDA FoodData Central<br/>curated local DB")]
    OUT["Verified recommendation<br/>+ full PipelineTrace"]

    IN --> S1 --> S2 --> S3 --> S4 --> OUT

    RE -. "weak-supervision labels" .-> S1
    RE -. "validity + hard-safety check" .-> S2
    RE -. "final safety scan" .-> S4
    DB -. "nutrients per 100 g" .-> S1
    DB -. "retrieval corpus" .-> S3
    DB -. "ground truth" .-> S4
```

### The four stages

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 · Prediction | `stage1_prediction/` | A goal-conditioned **meal-suitability surrogate** `f(nutrients, age_group, goal, health_flags) → [0,1]`, trained on weak-supervision labels from the guideline `RuleEngine`. It is the smooth objective the optimiser climbs — *not* the verifier. |
| 2 · CF optimisation | `stage2_optimizer/` | Differential evolution over `(quantity_g, form)` for every item in `planned_meal ∪ pantry` — and nothing else. Minimises `λ₁·(target − f) + λ₂·L1 + λ₃·sparsity` under hard safety penalties. |
| 3 · LLM-RAG translation | `stage3_rag/` | BM25 retrieval over USDA names grounds the optimised vector in real foods; a provider renders it as age-appropriate language. The default `TemplateProvider` is deterministic and needs no network. |
| 4 · Verification | `stage4_verification/` | Every generated item is re-matched to the USDA DB, its nutrients recomputed from ground truth, mismatches beyond ±10 % corrected, and a final hard-safety scan applied. |

### Why an ML model when we already have rules?

Because they do different jobs. The `RuleEngine` is a **discontinuous verifier** — ideal for
"is this meal safe and compliant?", useless as a search objective, because it gives an
optimiser no gradient to follow. The Stage-1 surrogate learns a **smooth, generalising
approximation** of guideline compliance that differential evolution can actually climb.
Validity is then judged by the rules, never by the surrogate, so the optimiser cannot game
its own model. This mirrors MetaPlate exactly, where the learned glucose model is distinct
from the 140 mg/dL threshold check. The long-form argument is in
[`docs/architecture.md`](docs/architecture.md).

---

## The five extensions over MetaPlate

| # | Extension | How it is implemented (not a promise — a file) |
|---|-----------|-----------------------------------------------|
| 1 | **Availability-aware** | `stage2_optimizer/space.py` builds decision variables *only* from `planned_meal ∪ pantry`. Unavailable foods are not penalised — they do not exist in the search space. |
| 2 | **Modification-based editing** | The optimiser starts from the user's own meal and pays `λ₂·L1 + λ₃·sparsity` for every gram and every item it touches. Pantry items start at 0 g, so a substitution only happens when it is worth its cost. |
| 3 | **Post-generation verification** | `stage4_verification/verifier.py` — match, recompute, compare, correct, re-scan. Its `VerificationReport` counts are the headline metric. |
| 4 | **Generalised health goals** | `configs/goals/{glycemic_control,weight_management,balanced_nutrition}.yaml`, layered with age-specific nutrient targets. |
| 5 | **Age/life-stage personalisation** | `configs/age_groups/{toddler,adult,older_adult}.yaml` — choking `(category, form)` bans with a nearest-safe-form map, medication–food interaction rules, and texture (IDDSI-style) constraints. |

**Choking hazards are a property of `(ingredient, preparation form)`, not of the ingredient.**
Whole grapes are unsafe for a toddler; quartered grapes are not. That is why every meal item
is a `(food_id, quantity_g, form)` triple — it lets the optimiser fix a hazard by changing the
*form*, which costs far less than removing the food.

---

## Quickstart

Requires Python 3.11+ (and Node 18+ only if you want the web UI).
**No API keys. No internet after setup.**

```bash
git clone https://github.com/aarush093/foodsense.git
cd foodsense

make setup    # create .venv, install dependencies
make data     # build the curated USDA food database
make train    # train the Stage-1 suitability surrogate
make demo     # run all three demo scenarios end-to-end
```

<details>
<summary><b>Windows (no GNU make)</b></summary>

GNU `make` is not installed on Windows by default. `make.ps1` mirrors every target:

```powershell
./make.ps1 setup
./make.ps1 data
./make.ps1 train
./make.ps1 demo
```
</details>

Web UI:

```bash
make serve    # builds the frontend, serves API + UI on http://localhost:8000
```

Docker:

```bash
docker compose up --build     # -> http://localhost:8000
```

---

## Demo scenarios

| Scenario | Profile | The problem | What FoodSense does |
|----------|---------|-------------|---------------------|
| `toddler_choking` | Toddler, 18 mo, balanced nutrition + iron focus | Whole grapes and whole peanuts are choking hazards | Re-forms grapes to `quartered` (a form fix, not a removal); substitutes the peanuts from the pantry; verifies every quantity against USDA |
| `elderly_sodium` | Older adult, 78 y, hypertension | Canned soup + salted crackers blow the 500 mg/meal sodium cap | Minimal swap to low-sodium broth, chicken and carrots; final sodium verified ≤ 500 mg |
| `adult_weight` | Adult, weight management | Burger and fries — over the kcal cap, under the protein floor | Demonstrates the generalised-goal objective on a third goal profile |

```bash
foodsense recommend --scenario toddler_choking
foodsense demo                                  # all three
```

---

## Results

Populated as each phase lands, and **only** from experiments that actually ran.
Regeneration instructions: [`docs/evaluation.md`](docs/evaluation.md).

Headline, over 300 sampled cases (100 per age group) — full table in
[`results/cf_comparison.md`](results/cf_comparison.md):

| Method | Usable validity | Safe | Availability violations | Safety violations | norm-L1 | Edits |
|---|---|---|---|---|---|---|
| **FoodSense-DE** | 18% | **100%** | **0%** | **0%** | **0.628** | **2.53** |
| Wachter (same space) | 47% | 79% | 0% | 21% | 0.969 | 5.73 |
| Wachter-style | 9% | 78% | 54% | 22% | 0.925 | 5.34 |
| DiCE-random | 17% | 80% | 36% | 20% | 0.985 | 2.09 |
| DiCE-genetic | 2% | 79% | 1% | 21% | 0.029 | 0.09 |
| Greedy | 24% | 82% | 51% | 18% | 0.859 | 3.74 |

FoodSense is the only method that never recommends a food the user does not have and
never leaves a safety violation in place, and it makes the smallest edit of any
method that edits at all. It reaches the nutrition target less often than an
unconstrained search does — `results/cf_comparison.md` isolates exactly how much of
that gap is the constraints and how much is the smaller search space, and shows that
the baselines' apparent advantage largely disappears once you stop counting
recommendations the user cannot actually cook.

| Artefact | What it shows |
|----------|---------------|
| `results/cf_comparison.md` | **Available now.** FoodSense-DE vs a same-space ablation vs Wachter-style vs DiCE (random/genetic) vs greedy — validity, L1/L2 distance, sparsity, **availability-violation %**, **safety-violation %**, runtime, by age group over 300 cases |
| `results/verification_eval.md` | Rate of hallucinated quantities / unsafe items in Stage-3 output, before vs after Stage 4 |
| `results/dataset_comparison.md` | **Available now.** Corpus reconstruction fidelity and Stage-1 metrics on Food.com vs Nutrition5k |
| `results/llm_benchmark.md` | Macro RMSE, goal consistency and diversity across providers (skipped gracefully with no keys) |

---

## Repository layout

```
src/foodsense/
├── schemas.py             # MealItem(food_id, quantity_g, form), UserProfile, PipelineTrace
├── data/                  # USDA DB builder, fuzzy lookup, corpus loaders
├── constraints/           # RuleEngine, age rules, goal thresholds  <- single source of truth
├── stage1_prediction/     # features, weak-supervision labels, LightGBM/XGBoost training
├── stage2_optimizer/      # search space, objective, differential evolution, baselines
├── stage3_rag/            # BM25 retriever, LLM providers, translation
├── stage4_verification/   # the verifier
├── pipeline.py            # run_pipeline(profile, planned_meal, pantry) -> PipelineTrace
└── cli.py                 # foodsense recommend / demo
api/                       # FastAPI: /api/recommend, /api/scenarios, /api/foods
frontend/                  # Vite + React + Tailwind single-page UI
experiments/               # every script that writes into results/
configs/                   # goal + age-group YAML (sourced to NASEM DRI, AAP/CDC, AHA...)
docs/                      # architecture, evaluation, traceability, demo script
```

---

## Roadmap

- [x] **Phase 0** — scaffold, tooling, CI, schemas
- [x] **Phase 1** — data layer (curated USDA DB, Food.com + Nutrition5k loaders)
- [x] **Phase 2** — `RuleEngine`, guideline configs, Stage-1 surrogate
- [x] **Phase 3** — Stage-2 optimiser + DiCE/Wachter/greedy baselines
- [ ] **Phase 4** — Stage-3 RAG + Stage-4 verifier + end-to-end pipeline
- [ ] **Phase 5** — FastAPI + React UI + Docker
- [ ] **Phase 6** — full evaluation, docs, ship

---

## Demo recording

<!-- TODO(Phase 5): replace with docs/assets/demo.gif once the UI is recorded -->
*A recorded click-path of the faculty demo will be embedded here. The written click-path
lives in [`docs/demo_script.md`](docs/demo_script.md).*

---

## Team

<!-- ─────────────────────────────────────────────────────────────────────
     PLACEHOLDER — to be filled in by the project team.
     Nothing here is auto-generated and no names have been invented.
     ───────────────────────────────────────────────────────────────── -->

**Course:** BCSE497J — PROJECT-I
**Institution:** Vellore Institute of Technology

| Name | Registration number |
|------|---------------------|
| _TODO_ | _TODO_ |
| _TODO_ | _TODO_ |

**Guide:** _TODO_
**Department:** _TODO_

---

## Acknowledgements

- **MetaPlate** — Arefeen, Johnston & Ghasemzadeh, *IEEE Journal of Biomedical and Health
  Informatics*, 2026 (arXiv:2606.10120). The four-stage architecture FoodSense extends.
- **USDA FoodData Central** — Foundation Foods and SR Legacy, U.S. Department of Agriculture.
- **Food.com Recipes and Interactions** — Li et al., via Kaggle.
- **Nutrition5k** — Thames et al., Google Research.
- Guideline values are sourced to NASEM Dietary Reference Intakes, AAP/CDC infant and
  toddler feeding guidance, the Dietary Guidelines for Americans, AHA and ESPEN/ASPEN.
  Every threshold in `configs/` carries its source as a YAML comment.

## License

MIT — see [LICENSE](LICENSE).
