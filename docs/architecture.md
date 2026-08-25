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

_(Phase 2)_ Choking hazard is a property of the pair, not the ingredient. See
`constraints/age_rules.py`.

## 4. Search space and objective

_(Phase 3)_

## 5. Retrieval and generation contract

_(Phase 4)_

## 6. Verification loop

_(Phase 4)_

## 7. Data flow and artefacts

_(Phase 1-2)_
