# FoodSense architecture

> **Phase 0 skeleton.** Sections marked _(Phase N)_ are filled in as those phases land.
> The Stage-1 justification below is final and is the answer to the question faculty
> are most likely to ask.

## 1. The four stages

FoodSense reproduces MetaPlate's separation of concerns and extends it:

| Stage | MetaPlate | FoodSense |
|-------|-----------|-----------|
| 1 | Postprandial glucose predictor (CGM-trained, RMSE ~16.46 mg/dL) | Goal-conditioned **meal-suitability surrogate** `f(nutrients, age_group, goal, health_flags) -> [0,1]` |
| 2 | Counterfactual macronutrient optimiser, glucose <= 140 mg/dL | **Availability-aware, age-aware** CF optimiser over `planned_meal union pantry`, editing `(quantity_g, form)` |
| 3 | LLM-RAG translation over USDA FoodData Central | Same, with a **deterministic offline template provider** as the default and LLMs as an optional enhancement |
| 4 | (threshold check) | A full **post-generation verification layer**: match, recompute, correct, re-scan |

## 2. Why a learned model when we already have a rule engine?

This is a deliberate design decision, not an artefact.

MetaPlate had continuous-glucose-monitor data from 25 adults. FoodSense targets
toddlers and older adults, for whom no comparable postprandial dataset exists, and
targets goals (weight management, balanced nutrition) that are not glucose at all.
So Stage 1 cannot be a glucose model. It is instead a **goal-conditioned meal-suitability
surrogate**, trained by weak supervision: labels are produced by the exact guideline
rule engine in `src/foodsense/constraints/`, softened with margin functions and
perturbed with Gaussian noise (sigma = 0.05), and LightGBM learns a smooth surrogate of
guideline compliance.

The obvious objection is: *if you already have the rules, why train a model on them?*

Because they do different jobs, and the pipeline needs both:

- **The rule engine is a verifier.** It is discontinuous by nature -- a meal is over the
  sodium cap or it is not; a grape is whole or it is quartered. That is exactly what you
  want when deciding whether an output is safe to show a user, and exactly what you cannot
  optimise against. It provides no gradient, no notion of "closer", and no ordering among
  infeasible candidates.
- **The surrogate is a search objective.** Model-based counterfactual search needs a fast,
  smooth, everywhere-defined score so the optimiser can tell that 620 mg of sodium is
  *better* than 900 mg even though both fail. Learning that surface from the rules gives
  the optimiser something to climb, and lets it generalise between the explicit thresholds
  rather than teleporting between them.

This mirrors MetaPlate precisely: its learned glucose model is a distinct object from the
140 mg/dL threshold check that decides validity. FoodSense keeps the same discipline, and
enforces it structurally -- **Stage-2 validity is judged by the RuleEngine, never by the
surrogate**, so the optimiser cannot manufacture success by exploiting its own model. The
surrogate proposes; the rules dispose; Stage 4 audits the result against USDA ground truth.

For the `glycemic_control` goal, an estimated per-meal glycemic load is included as an
explicit feature, so that goal remains faithful to MetaPlate's glucose origin.

## 3. Why `(food, form)` and not just `food`

A choking hazard is a property of the *pair*. Whole grapes are unsafe for a toddler;
quartered grapes are not. Encoding the ban as `(hazard_class, form)` gives the
optimiser a cheap repair -- change the form -- rather than forcing it to delete the
food, which is what makes the toddler worked example come out the way the proposal
describes.

Three cases fall out of that encoding, all of them data in
`configs/age_groups/toddler.yaml` rather than branches in code:

- **A safe form exists.** Grapes move to `quartered`, hard raw vegetables to
  `soft_cooked`, hot dogs from `sliced_rounds` to `minced`.
- **No safe form exists.** Popcorn, marshmallows, hard candy and gum are banned in
  every form, and the curated database gives those foods exactly one allowed form,
  so the search space offers no escape hatch. The only repair is removal.
- **A safe form exists, but only above a certain age.** Whole nuts can be ground for
  a three-year-old and not for an eighteen-month-old. `safe_form_min_age_months`
  carries this, and an unknown age is treated as the youngest -- guessing wrong in
  the other direction costs a choking hazard rather than an inconvenience.

Medication and condition rules live in `configs/health_flags.yaml` rather than inside
each age group, because they are age-independent: a 40-year-old on warfarin needs the
same vitamin-K consistency as an 80-year-old.

### The added-sugar proxy — a documented substitution, not a measurement

One rule in the system is not evaluated against measured data, and it is worth being
explicit about which one and why.

Neither USDA release FoodSense consumes — SR Legacy (2018) or Foundation Foods
(2025-04) — reports FDC nutrient 1235, "Added Sugars". Every food in the curated
database therefore carries `added_sugars_g = 0.0`. Taken literally, the AAP's "no
added sugars below 24 months" and the DGA's daily ceiling could never fire: a toddler
could be served a 330 ml cola and the rule would report full compliance.

Rather than drop the rule, FoodSense substitutes a **documented proxy**: in the
`sweets`, `snack`, `baked` and `cereal` categories, and in any food tagged
`sweetened_beverage`, a food's *total* sugars are counted as added sugars. Where a
food does carry a measured added-sugar value, that value is used and the proxy is not
applied.

Three properties make the substitution defensible:

- **It is conservative in the protective direction.** In those categories sugar is
  overwhelmingly added rather than intrinsic, so the proxy can overstate added sugar
  but rarely understates it. For a safety rule aimed at toddlers, erring toward
  restriction is the correct direction to err in.
- **Fruit and dairy are deliberately excluded.** Their sugar is intrinsic, and
  counting it would penalise exactly the foods a toddler should be eating. 100 g of
  grapes and 200 g of whole milk both score 0.0 g of added sugar; 330 ml of regular
  cola scores 32.8 g, above the entire 25 g/day toddler ceiling.
- **`sweetened_beverage` is category-scoped and negation-aware**, assigned at
  database-build time. Diet colas and unsweetened almond milk do not carry it. Adding
  the tag mattered: only 4 of 110 curated beverages carried the generic
  `added_sugar_source` keyword tag, so sodas and lemonades were invisible to the rule.

The proxy is labelled as such in `configs/age_groups/toddler.yaml`,
`configs/age_groups/adult.yaml`, `configs/health_flags.yaml`, `constraints/goals.py`
and `data/README.md`. It is the one place where a threshold is compared against an
estimate rather than a measurement, and nothing else in the system does this.

### Calibrating a per-meal target

Daily reference intakes have to become per-meal ones, and floors and ceilings do not
divide the same way. Exceeding a sodium ceiling in one meal genuinely matters, so
ceilings are the proportional share `daily / meals_per_day`. Floors are different:
micronutrient adequacy is assessed across a day and is heavily skewed between meals.
Measured on 400 Food.com meals, a median single serving supplies 0.22 of a
proportional calcium floor, 0.29 of an iron one and 0.02 of a vitamin-D one --
so demanding the full share would mark essentially every real meal non-compliant, and
did, until `per_meal_floor_attainment` was introduced.

The softening has one subtlety worth recording. Scaling each bound of a *band* by its
own magnitude makes narrow bands unsatisfiable: a meal sitting exactly in the middle
of a 25-35% fat-share band scored 0.57. Bands are therefore softened by the width of
the band, which puts the centre near 1.0 and keeps each edge at exactly 0.5. (That
particular band is now the NASEM AMDR's 20-35%; the arithmetic problem was general.)

## 4. Search space and objective

### Availability is structural, not a penalty

Extension #1 is implemented by what the search space *contains*. `space.py` builds
decision variables from `planned_meal + pantry` and nothing else, so an unavailable
food has no variable and no point in the space can contain one. The alternative --
letting every food into the space and penalising the unavailable ones -- would make
availability a quantity the optimiser can trade away against some other gain, at
whatever exchange rate the weights happen to imply. Here there is no exchange rate,
because there is nothing to trade.

This is why `tests/test_stage2.py` tests the property rather than a run: it samples
hundreds of random points in the space and asserts that every decoded meal stays
inside the available set. A test that merely checked one optimiser run had not used
an unavailable food would be testing luck.

Each food contributes two dimensions: a continuous `quantity_g`, and a `form` index
decoded through that food's `allowed_forms`. The second is what lets a choking hazard
be repaired by re-forming a food rather than deleting it, and it is also something a
generic tabular counterfactual method cannot express at all.

### The objective, and two things measurement changed

    lambda1 * max(0, target - f(x))          get the meal over the line
  + lambda2 * L1(x, x0) / scale              stay close to what was planned
  + lambda3 * (items changed)                change as few things as possible
  + lambda_form * (form-change cost)         prefer the declared nearest safe form
  + big_penalty * (hard safety violations)   never trade safety for anything

The validity term is one-sided on purpose. Once `f(x)` clears the target there is
nothing more to gain, so the optimiser stops improving nutrition and starts
minimising distance -- which is what keeps the output an *edit* rather than a
replacement meal.

Two terms were added or changed on evidence rather than taste, and both are recorded
with their measurements in `configs/pipeline.yaml`:

- **`lambda_validity` is 5.0, not the brief's 1.0.** Because `f` is in [0,1], the
  validity term is bounded by the target (0.70) while sparsity costs 0.15 per edit.
  At 1.0 the optimiser cannot afford the edits a meal needs: across 24 sampled cases
  it made 0.58 edits and improved the guideline score by +0.019 -- it repaired safety
  and then stopped. The full sweep is in the config. Safety was 100% at every setting,
  which is the point: that guarantee does not depend on tuning.
- **`lambda_form_preference` (0.05) and a per-form cost.** Without it the optimiser
  repaired whole grapes by *pureeing* them. That is safe, and the rules accept it, but
  it is not what the AAP's nearest-safe-form map says to do. The cost is small enough
  that it only ever decides between forms the rules consider equally safe.

A `min_serving_g` floor of 10 g was also needed: without it the optimiser discovered
that a two-gram sliver of lentils nudges the fibre score at almost no distance cost,
and returned meals with three garnish-sized additions nobody would serve.

### Validity is judged by the rules, never by the surrogate

The optimiser climbs `f`, but whether it has succeeded is decided by
`RuleEngine.is_valid`. An optimiser marked by its own model can always win by finding
that model's blind spots; this is the structural guarantee that it cannot. It is also
why the early-stopping condition is "valid by the rules *and* plateaued" rather than
"the objective stopped moving".


## 5. Retrieval and generation contract

Stage 3 turns the optimiser's decision into language. It does not make decisions --
the diff between the planned and optimised meals is computed in `translate.py` and
handed to the provider as a fact, along with the rule-engine message that caused each
edit. A generative step describing a decision is a much smaller trust surface than one
making it.

**The default provider is the offline one.** `TemplateProvider` is deterministic,
needs no key and no network, and is what `make demo` uses. That ordering is the
project's central robustness claim: the LLM is an enhancement layer, never a
dependency. It is also the fallback for every other provider, so its output quality
sets the floor for the whole system.

Concretely: **every command runs offline unless you ask otherwise.** `make demo`,
`foodsense recommend`, `foodsense scenarios`, the test suite and every experiment in
`experiments/` use the template provider and touch no network. An LLM is reached only
by passing `--provider anthropic|openai|ollama` explicitly, and even then a missing
key, a missing package or an unreachable endpoint degrades to the template with the
reason recorded in `trace.warnings` rather than raising.

The Anthropic model id is pinned in `configs/pipeline.yaml` and was checked against
the published model list on 2026-08-26; `providers.py` records the date and the
lineup it was checked against. Sampling parameters are the one place the model id
changes the request shape -- models after Claude Opus 4.6 reject a `temperature`
other than 1.0 with a 400 -- so the provider sends the brief's 0.2 only to models
known to accept it, and omits it for anything unrecognised. That direction is
deliberate: omitting always works, sending can fail, and an unknown id is more
likely to be newer than the list than older.

Every provider answers the same contract:

```json
{"items": [{"name", "food_id", "quantity_g", "form"}], "text", "rationale"}
```

The LLM providers send the BM25-retrieved USDA candidates so the model picks real
foods rather than inventing them, demand strict JSON, validate the reply, retry once,
and fall back to the template. Parsing is strict about *structure* and forgiving about
*packaging*: a markdown fence or a preface is cosmetic and should not waste the retry,
while a missing field or an unknown form is a real breach Stage 4 should never see.

Retrieval (BM25 over names, categories and tags) and Stage-4 matching are deliberately
different algorithms. BM25 answers "which foods are plausibly related to this phrase";
the matcher in `data/fdc.py` answers "is this specific name the same food as one we
hold". Using one for both would make verification partly circular -- grading the
retriever against the retriever's own notion of similarity.

## 6. Verification loop

Extension #3, and the premise is that **nothing a generator says is trusted**. Whatever
Stage 3 produces is re-checked from scratch:

1. **Match** each item's name to a real food; below the threshold it is recorded as
   `unmatched` and replaced by the retriever's best real candidate -- not silently
   dropped, not silently kept.
2. **Recompute** nutrients from the database row times the quantity. The generator's
   own figures are never used, only compared against.
3. **Compare** claims at ±10% and correct to the database value, recording both.
4. **Re-scan** with the same `RuleEngine` that produced the Stage-1 labels and judged
   Stage-2 validity.
5. **Repair** a survivor by moving to the nearest safe form, or removing it when no
   safe form exists.

Step 4 is the one that matters. A generative rewrite can undo a safety decision the
optimiser made, and re-running the rules on the final list is what stops that being
the last word.

### Measured

`experiments/run_verification_eval.py` runs two studies, because either alone would
mislead. The observed-rate study finds near-zero errors for the template provider --
true, and only because that path emits the optimiser's own items unchanged. The
fault-injection study measures the verifier against generators that *do* err, with
every fault labelled as injected:

| Injected fault | Cases | Detected | Reached the user |
|---|---|---|---|
| `hallucinated_food` | 90 | 100% | 0% |
| `impossible_form` | 90 | 100% | 0% |
| `inflated_claim` | 90 | 100% | 0% |
| `reintroduced_hazard` | 30 | 100% | 0% |
| `quantity_drift` | 90 | 94% | 6% |

`quantity_drift` is the only leaky class, and for a defensible reason: a ±10% tolerance
is a ±10% tolerance, so a drift on a small item that moves the meal total by less than
that is inside the stated bound rather than missed.

## 7. Data flow and artefacts

_Rewritten in P1 of the repositioning. See `docs/pivot.md`._

### The food knowledge base

One SQLite database at `data/processed/food_db.sqlite`, with a Parquet mirror that
`FoodDB` prefers because it loads in milliseconds. The DDL lives in one place,
`data/schema.py`, so the build, the migration and the repository cannot disagree
about the shape of the table.

`foods` holds 2,590 rows, all currently migrated USDA. Beyond the descriptive and
safety columns it carries **provenance on every row** — `source_type`,
`source_name`, `source_reference`, `source_id`, `confidence`, `verification_status`
— because requirements sections 9 and 10 make provenance mandatory rather than
optional, and because the moment a second source lands, "which source said this"
stops being answerable from context.

Five tables join it: `food_aliases`, `food_nutrients_extra` (nutrients outside the
canonical 33), `food_toddler_meta` (age appropriateness, choking risk, allergens —
every column nullable or admitting `unknown`, because inventing a safety claim is
the most dangerous thing this system could do), `recipe_derivations` (how a
`derived_recipe` row was computed, stored so the arithmetic can be audited), and
`sources`.

### Why the nutrient columns stay wide

The 33 canonical nutrients are 33 columns rather than an entity-attribute-value
table. EAV is the textbook answer for sparse optional attributes and would be the
wrong one here: Stage 1 reads all 33 as a dense vector for every candidate in every
generation of the optimiser's search, and a join-and-pivot per candidate would cost
more than the rest of Stage 2 put together. Nutrients outside the 33 go in
`food_nutrients_extra`, where sparsity is real and the access pattern is occasional.

### Identity

The primary key is `id`: `<source_type>:<source_id>` for new sources, and a bare
`fdc_id` for the migrated USDA rows, whose ids are pinned by committed scenarios and
golden traces. Namespaced rather than a UUID because both are strings — so
`MealItem.food_id` needs no type change either way — but a UUID is random, and
`make data` must reproduce byte-identical output. A namespaced key is a pure
function of the source and its own identifier.

Duplicate prevention is enforced by the database, not by review: `name_key` (the
normal form from `data/normalize.py`) carries a unique index over active rows, so
`Masala Dosa`, `masala dosa` and `Masala-Dosa` collide on insert rather than
depending on someone reading a warning.

### NULL means unknown, and never zero

Nutrient columns are nullable. `NULL` means the source does not report the nutrient;
it does not mean the food contains none of it. This is ruling R8, and it is the
single most consequential change P1 makes to the data layer — the build used to end
with `fillna(0.0)`, which made the two indistinguishable for the project's entire
life.

Because Stage 1 needs a dense array, the value 0.0 is still substituted when a
record is materialised — but in exactly one named place,
`data/coverage.zero_filled_vector()`, and only alongside
`FoodRecord.reported_nutrients`, the mask saying which values are real. Summing a
vector is fine. Concluding that a food contains no iron because `iron_mg == 0.0` is
not, and `record.reports("iron_mg")` is how a caller asks. Ruling R13 retires the
substitution in P3; `docs/evaluation.md` records the measured scale of what it was
hiding.

### Reading it

Everything goes through `data/repository.FoodRepository`. It applies visibility once
— `search()` and `records()` return only foods that may actually be recommended —
so that six callers cannot disagree about whether a pending row is eligible. Callers
that genuinely need everything, such as the duplicate check, pass
`include_unverified=True` and say so at the call site. `FoodDB` remains underneath
as the in-memory index and the fuzzy matcher; the repository wraps it rather than
hiding it, because Stage 4 matches with that scorer and Stage 3 ranks with BM25, and
those are deliberately different algorithms (see section 5).

### Determinism

`make data` is reproducible: two runs produce identical content across all six
tables. Migrated rows carry a fixed `created_at`/`updated_at` constant rather than
wall-clock time. The only value that differs between two builds is
`build_info.built_at`, which records when the build ran rather than what it
produced.
