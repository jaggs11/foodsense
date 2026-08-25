# Frontend

Vite + React + Tailwind single-page UI for the pipeline.

> **Phase 0 skeleton** -- scaffolded in Phase 5.

Planned layout: a four-step pipeline stepper; a Stage-2 before/after meal diff (unchanged
grey, removed red strikethrough, modified amber with old -> new grams, added green); a
Stage-4 panel with per-item verification badges and a metrics strip (suitability before ->
after, L1 distance, number of edits); a raw `PipelineTrace` JSON viewer; and a scenario
dropdown preloaded with the three demo cases.

```bash
npm install && npm run dev     # dev server, proxies /api to :8000
npm run build                  # emits dist/, served by FastAPI at /
```
