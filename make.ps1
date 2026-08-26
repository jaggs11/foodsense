<#
.SYNOPSIS
    Windows equivalent of the Makefile targets (GNU make is not installed by
    default on Windows). Every target here mirrors the Makefile 1:1.

.EXAMPLE
    ./make.ps1 setup
    ./make.ps1 data
    ./make.ps1 demo
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'data', 'train', 'demo', 'test', 'test-fast', 'eval',
                 'frontend', 'api', 'serve', 'lint', 'format', 'verify-results',
                 'docker-build', 'docker-up', 'clean')]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Venv = Join-Path $Root '.venv'
$Bin = Join-Path $Venv 'Scripts'
$Py = Join-Path $Bin 'python.exe'
$Pip = Join-Path $Bin 'pip.exe'
$Ruff = Join-Path $Bin 'ruff.exe'

function Require-Venv {
    if (-not (Test-Path $Py)) {
        throw "No virtual environment found at $Venv. Run: ./make.ps1 setup"
    }
}

switch ($Target) {
    'help' {
        Write-Output @'
FoodSense targets (./make.ps1 <target>):
  setup   - create .venv and install dependencies
  data    - build the curated USDA food DB + corpus samples
  train   - train the Stage-1 suitability surrogate
  demo    - run the three demo scenarios offline
  test    - run pytest
  eval    - regenerate results/ tables and figures
  serve   - build frontend + run FastAPI on http://localhost:8000
  lint    - ruff check
  format  - ruff format
  clean   - remove caches and build artefacts
'@
    }
    'setup' {
        python -m venv $Venv
        & $Py -m pip install --upgrade pip
        & $Pip install -r requirements.txt
        & $Pip install -e .
        Write-Output ''
        Write-Output "Setup complete. Optional extras: $Pip install -r requirements-optional.txt"
    }
    'data' {
        Require-Venv
        & $Py -m foodsense.data.build_food_db
        & $Py -m foodsense.data.corpora --prepare
    }
    'train' { Require-Venv; & $Py -m foodsense.stage1_prediction.train }
    'demo' { Require-Venv; & $Py -m foodsense.cli demo }
    'test' { Require-Venv; & $Py -m pytest }
    'test-fast' { Require-Venv; & $Py -m pytest -m "not slow" }
    'eval' {
        # Strictly sequential and strictly ordered: run_validity_decomposition
        # reads the rows run_cf_eval writes. Roughly 90 minutes, most of it
        # DiCE-genetic exhausting its budget without converging by construction.
        Require-Venv
        & $Py experiments/run_cf_eval.py
        & $Py experiments/run_validity_decomposition.py
        & $Py experiments/run_lambda_sweep.py
        & $Py experiments/run_surrogate_boundary.py
        & $Py experiments/run_verification_eval.py
        & $Py experiments/run_dataset_comparison.py
        & $Py experiments/run_llm_benchmark.py
        Write-Host 'Now check results/ against its manifest:  ./make.ps1 verify-results'
    }
    'verify-results' { Require-Venv; & $Py scripts/verify_results.py }
    'frontend' {
        Push-Location frontend
        try { npm install; npm run build } finally { Pop-Location }
    }
    'api' { Require-Venv; & $Py -m foodsense.cli serve --no-open }
    'serve' {
        Require-Venv
        Push-Location frontend
        try { npm install; npm run build } finally { Pop-Location }
        Write-Output 'Serving FoodSense on http://localhost:8000'
        & $Py -m foodsense.cli serve
    }
    'lint' { Require-Venv; & $Ruff check . }
    'format' { Require-Venv; & $Ruff format .; & $Ruff check --fix . }
    'docker-build' { docker build -t foodsense:latest . }
    'docker-up' { docker compose up --build }
    'clean' {
        foreach ($p in @('.pytest_cache', '.ruff_cache', '.coverage', 'htmlcov', 'results/tmp')) {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Get-ChildItem -Path . -Include '__pycache__' -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch '\.venv\' } |
            Remove-Item -Recurse -Force
        Write-Output 'Cleaned.'
    }
}
