# Faculty demo script (5 minutes)

> **Phase 0 skeleton.** The exact click-path is recorded here in Phase 5, once the UI
> exists. What is fixed already: the demo runs **offline**, with **no API keys**, and
> uses the three scenarios from the proposal.

## Before the demo

```bash
make setup && make data && make train    # once, with internet
```

## The run

1. `make serve` -> open `http://localhost:8000`
2. Scenario 1 -- `toddler_choking` _(Phase 5: click-path)_
3. Scenario 2 -- `elderly_sodium` _(Phase 5: click-path)_
4. Scenario 3 -- `adult_weight` _(Phase 5: click-path)_

## Fallback if the UI misbehaves

```bash
foodsense demo        # same three scenarios, terminal output, no browser
```
