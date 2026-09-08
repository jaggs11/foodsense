# configs/conditions/

Toddler-relevant nutrition conditions. Replaces `configs/goals/`, which is
retired in P3 along with the adult and older-adult age groups.

Empty for now: no condition file is written until the evidence behind it is in
`knowledge/sources/sources.yaml` (P2.0). This README describes the schema P3 will
implement.

## The target set

Iron-deficiency anemia; undernutrition (mild to moderate, guidance-level);
vitamin D deficiency with calcium co-monitoring; functional constipation; celiac
disease.

Food allergies and vegetarian/non-vegetarian status are **cross-cutting
restrictions**, not conditions -- they apply across every condition and are
modelled separately. `glycemic_control` is not carried forward.

## Schema

A superset of the goal YAML it replaces:

```yaml
condition_id: iron_deficiency_anemia
name: <display name>
description: <one paragraph, plain language>
population: toddler
age_months: {min: 12, max: 36}

priority_nutrients: [<keys the condition is trying to raise>]
limit_nutrients:    [<keys the condition is trying to hold down>]
monitor_nutrients:  [<keys reported but not optimised -- calcium under vitamin D>]

exclude_tags:        [<food tags the condition forbids outright>]
exclude_ingredients: [<ingredient keys the condition forbids>]

rules:
  - rule_id: <stable id>
    quantity: <a key meal_metrics() can measure>
    threshold: {minimum: <n>} | {maximum: <n>}
    severity: hard | soft
    weight: <float, soft rules only>
    message: <what the user is told>
    sources: [<source_id>, ...]     # REQUIRED -- see below

clinical_notice: <when to see a clinician; shown with every recommendation>
out_of_scope_severity: <presentations this system must not advise on at all>

sources: [<source_id>, ...]
enabled: true | false
```

## The two rules that matter

**Every rule cites its evidence.** `sources` on a rule is not optional. A rule
whose `sources` are not all present in `knowledge/sources/sources.yaml` makes the
whole condition ineligible for `enabled: true`, and `verify_sources.py
--schema-only` fails the build. This is what mechanically enforces R3 and R9 --
a condition cannot be switched on by someone who forgot the evidence.

**Adding a condition is one YAML file plus one manifest entry.** No code change.
If a new condition needs code, the schema is wrong and the schema gets fixed.

## What belongs in clinical_notice rather than in a rule

Anything the evidence says that a food recommendation cannot deliver. Vitamin D
is the clearest case: the evidence is expected to state that diet alone is
frequently insufficient, so that belongs in the notice shown alongside the
recommendation, not encoded as a nutrient target the optimiser will happily claim
to have met. Severe acute malnutrition, red-flag constipation symptoms and
coeliac diagnosis all sit in `out_of_scope_severity` for the same reason: the
right output is a referral, not a meal.

Coeliac carries one further caution. Diagnosis requires a clinician, and gluten
must not be withdrawn before testing -- a system that implies otherwise causes
real harm by making the diagnosis harder to reach.
