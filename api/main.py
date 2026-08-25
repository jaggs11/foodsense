"""FastAPI app: the pipeline over HTTP, plus the built single-page frontend.

Routes (Phase 5):
    POST /api/recommend   -> full PipelineTrace
    GET  /api/scenarios   -> the three preloaded demo scenarios
    GET  /api/foods?q=    -> food-database autocomplete
    GET  /api/health      -> liveness probe (used by docker-compose)
    GET  /                -> frontend/dist/index.html

Only the health route is live in Phase 0, so that the container and CI have
something real to check.
"""

from __future__ import annotations

from fastapi import FastAPI

from foodsense import __version__

app = FastAPI(
    title="FoodSense",
    version=__version__,
    description=("Availability-aware, verification-guided counterfactual food recommendation."),
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}
