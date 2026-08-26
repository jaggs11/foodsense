# Faculty demo script (5 minutes)

The demo runs **offline**, with **no API keys**, on **localhost**. Turn the Wi-Fi
off before you start — not as a precaution, but because doing it in front of the
room is the most direct way to make the point.

## Before the demo (once, with internet)

Windows PowerShell, from the repository root:

```powershell
.\make.ps1 setup     # venv + dependencies
.\make.ps1 data      # build the curated USDA database
.\make.ps1 train     # fit the Stage-1 surrogate
cd frontend; npm install; npm run build; cd ..
```

macOS / Linux: `make setup && make data && make train && make frontend`.

Verify it is ready — this needs no network:

```powershell
.\.venv\Scripts\python.exe -m foodsense.cli serve --no-open
# then in another terminal:
curl http://127.0.0.1:8000/api/health
# {"status":"ok", ... ,"model_loaded":true,"n_foods":2590, ...}
```

`model_loaded: false` means `train` has not run and Stage 2 will be skipped. Fix
it before the demo, not during.

## Starting it

```powershell
.\.venv\Scripts\python.exe -m foodsense.cli serve
```

Binds `127.0.0.1:8000` and opens the page. One process, one origin, no network.
`.\make.ps1 serve` does the same and rebuilds the frontend first.

## The run

**Open with the claim, then let the screen make it.** "This gives dietary advice
that is safe for the person in front of you, only uses food they already have,
and never states a nutrition figure it has not re-checked against USDA."

### 1. `toddler_choking` — the headline (2 min)

An 18-month-old's lunch: whole grapes, whole peanuts, rice.

1. Leave the provider on **template** and point out the label: *offline default*.
   The dropdown shows `anthropic — unavailable`, and beneath it the reason,
   `ANTHROPIC_API_KEY not set`. That is the architecture, not a fault: an LLM is
   an enhancement here and never a dependency.
2. Click **Run pipeline**. It takes about a second.
3. **Stage 1** — two red `hard` violations: `toddler.choking.grape` and
   `toddler.choking.nut`.
4. **Stage 2** — the point of the whole project. The grapes are **amber**, not
   red: `whole → quartered`. The hazard is repaired by *re-forming* the food, not
   by deleting it. The peanuts are **red and struck through**, because at 18
   months a nut has no safe form — and ground chicken appears in **green**,
   substituted in **from the pantry**.
   > Say the line: *"It could not have suggested anything else, because the search
   > space is built from the planned meal and the pantry and contains nothing
   > else. Availability is structural here, not a penalty term."*
5. **Stage 4** — verified safe, with the checked/corrected/fixed counts visible.
   Explain the zeroes: on the offline path Stage 3 emits the optimiser's own
   items, so there is nothing to correct. The zero means the reported meal *is*
   the database recomputation, not that the check was skipped.

### 2. `elderly_sodium` — a numeric limit (1 min)

78, hypertension, canned soup and salted crackers.

1. Run it. Sodium falls from about 1,400 mg to under the 500 mg per-meal ceiling.
2. Note the honest part: the metrics strip shows suitability improved a long way
   but Stage 2 reads **"best safe edit found"**, not "reached the target". The
   composite score also carries per-meal micronutrient floors that no single meal
   can meet. Do not oversell it — that is what `results/` is for.

### 3. `adult_weight` — no safety rule at all (1 min)

Burger, fries, cola, for weight management. Shows the goal layer working with no
hazard involved: energy comes down, broccoli comes in from the pantry.

### 4. Determinism (30 s)

Run the same scenario twice with **seed 42** — identical output. Change the seed,
run again, and the answer may differ. Every trace shows the seed that produced it,
so anything on screen can be reproduced exactly.

### 5. The raw trace (30 s)

Expand **Raw PipelineTrace**. Everything on screen came from this one document —
there is no second path and no post-processing between the pipeline and the page.

## If the UI misbehaves

The terminal path renders the same three scenarios and depends on nothing in the
browser:

```powershell
.\.venv\Scripts\python.exe -m foodsense.cli demo
.\.venv\Scripts\python.exe -m foodsense.cli recommend --scenario toddler_choking
```

## Questions you should expect

**"Why is validity only 29%?"** Because the optimiser is charged for every edit it
makes, and most invalid cases are meals it knew were short and would not buy the
edits to fix. `results/validity_decomposition.md` splits all 213 of them: none
were left unsafe, 90% are that trade, 10% are Stage-1 calibration.
`results/lambda_sweep.md` shows validity moving 2% → 66% as that weight changes,
with safety at 100% at every setting. Validity is a dial; safety is not a setting.

**"Isn't the LLM doing the real work?"** No — it is off by default and the demo
you just watched never called one. The optimiser decides; Stage 3 only describes
the decision; Stage 4 re-checks the description against USDA.

**"How do you know Stage 4 works if it corrected nothing?"**
`results/verification_eval.md` injects faults deliberately and measures what is
caught, split into faults caught by construction and faults that need genuine
re-derivation. The re-derivation block is the honest number.
