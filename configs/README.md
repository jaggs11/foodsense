# Configuration

All guideline thresholds live here as YAML, never hard-coded in Python, and **every
threshold carries its source as a comment** so it can be checked against the literature.

> **Phase 0 skeleton** -- the YAML files below are authored in Phase 2 together with the
> `RuleEngine` that reads them.

| File | Contents | Sources |
|------|----------|---------|
| `age_groups/toddler.yaml` | 1-3 y per-meal targets, choking `(category, form)` bans, nearest-safe-form map, texture phrasing | NASEM DRI; AAP/CDC infant & toddler feeding guidance |
| `age_groups/adult.yaml` | Default adult targets | NASEM DRI; Dietary Guidelines for Americans |
| `age_groups/older_adult.yaml` | 65+ targets, medication-food interaction rules, IDDSI-style texture limits | NASEM DRI; DGA older-adults chapter; AHA; ESPEN/ASPEN |
| `goals/glycemic_control.yaml` | Per-meal glycemic load and sugar caps | MetaPlate; ADA standards of care |
| `goals/weight_management.yaml` | kcal cap, protein floor, fibre floor | DGA |
| `goals/balanced_nutrition.yaml` | Acceptable macronutrient distribution ranges | NASEM AMDR |
| `pipeline.yaml` | Objective weights (lambda 1-3), DE hyperparameters, tolerances, seed | -- |
