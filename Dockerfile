# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build the React frontend
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2 — python runtime serving the API + built frontend
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by LightGBM / XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY data/processed/ ./data/processed/
COPY data/samples/ ./data/samples/
COPY models/ ./models/
RUN pip install --no-deps -e .

# Built single-page app served by FastAPI at "/". Pointed at explicitly rather
# than discovered: the API resolves the bundle by walking up from its own module
# file, which finds /app/frontend/dist for the editable install above but would
# not if this were ever switched to a non-editable one. Stating it removes that
# dependency on how the package happens to be installed.
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
ENV FOODSENSE_FRONTEND_DIST=/app/frontend/dist

EXPOSE 8000
CMD ["uvicorn", "foodsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
