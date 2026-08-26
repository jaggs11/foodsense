# Frontend

Vite + React + Tailwind single-page UI for the pipeline.

The bundle is **fully self-contained**: no CDN, no webfonts, no telemetry. That is
a requirement rather than a preference — the demo's central claim is that it runs
with the Wi-Fi off, and a Google Fonts link would quietly make that false.

```bash
npm install && npm run build   # emits dist/, served by FastAPI at /
npm test                       # 4 smoke tests (vitest + jsdom)
npm run dev                    # dev server on :5173, proxies /api to :8000
```

`dist/` is gitignored. The demo story is `foodsense serve`, which builds nothing
itself but is documented alongside `make serve` / `./make.ps1 serve`, and those do
build first — so there is no need to commit a bundle.

## Dev mode

`npm run dev` runs a second server on :5173 and proxies `/api` to :8000, so the
API has to allow that origin. It does not by default. Start the API with
`FOODSENSE_DEV=1` to enable CORS, which is restricted to `localhost:5173` and
`127.0.0.1:5173` and nothing else:

```powershell
$env:FOODSENSE_DEV = "1"; .\.venv\Scripts\python.exe -m foodsense.cli serve --no-open
```

The shipped shape needs none of this: FastAPI serves the built assets itself, so
there is one origin and no cross-origin request to permit.

## Layout

| Component | What it renders |
|---|---|
| `Stepper` | the four stages, with each stage's own `runtime_s` |
| `MealDiff` | Stage-2 before/after — removed red + struck, added green, changed amber with old → new grams, kept grey. Colour is never the only signal; every row also carries a word and a symbol, because a projector washes out exactly the distinction the colours are making |
| `Verification` | Stage 4 in full: counts checked/corrected/fixed/unmatched, the hazards it found, the repairs it applied, the claims it overwrote, and `final_pass` |
| `Metrics` | suitability before → after, edits, L1 distance, safety, total time |
| `App` | scenario dropdown, provider dropdown (unavailable ones disabled with their reason shown), seed input, warnings, raw trace viewer |
