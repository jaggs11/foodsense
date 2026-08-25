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

_(Phase 4)_

## 6. Verification loop

_(Phase 4)_

## 7. Data flow and artefacts

_(Phase 1-2)_
