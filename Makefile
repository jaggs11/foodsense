# FoodSense — one-command developer workflow.
#
#   make setup   create .venv and install all dependencies
#   make data    build the curated USDA food database + corpus samples
#   make train   train the Stage-1 suitability surrogate (LightGBM + XGBoost)
#   make demo    run the three offline demo scenarios end-to-end
#   make test    run the test suite
#   make eval    regenerate every table/figure in results/
#   make serve   build the frontend and serve the API + UI on one URL
#
# Windows users without GNU make: use the equivalent `./make.ps1 <target>`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV := .venv
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif
PY   := $(BIN)/python
PIP  := $(BIN)/pip

.PHONY: help setup data train demo test eval serve api frontend lint format clean docker-build docker-up

help:
	@echo "FoodSense targets:"
	@echo "  setup   - create .venv and install dependencies"
	@echo "  data    - build the curated USDA food DB + corpus samples"
	@echo "  train   - train the Stage-1 suitability surrogate"
	@echo "  demo    - run the three demo scenarios offline"
	@echo "  test    - run pytest"
	@echo "  eval    - regenerate results/ tables and figures"
	@echo "  serve   - build frontend + run FastAPI on http://localhost:8000"
	@echo "  lint    - ruff check"
	@echo "  format  - ruff format"
	@echo "  clean   - remove caches and build artefacts"

setup:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo ""
	@echo "Setup complete. Optional extras: $(PIP) install -r requirements-optional.txt"

data:
	$(PY) -m foodsense.data.build_food_db
	$(PY) -m foodsense.data.corpora --prepare

train:
	$(PY) -m foodsense.stage1_prediction.train

demo:
	$(PY) -m foodsense.cli demo

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

eval:
	$(PY) experiments/run_cf_eval.py
	$(PY) experiments/run_verification_eval.py
	$(PY) experiments/run_dataset_comparison.py
	$(PY) experiments/run_llm_benchmark.py

frontend:
	cd frontend && npm install && npm run build

api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

serve: frontend
	@echo "Serving FoodSense on http://localhost:8000"
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

lint:
	$(BIN)/ruff check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

docker-build:
	docker build -t foodsense:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/**/__pycache__ .coverage htmlcov results/tmp
