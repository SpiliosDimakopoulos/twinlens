# The venv's python on Windows or POSIX; override with `make PY=python3 ...`.
PY ?= $(firstword $(wildcard .venv/Scripts/python.exe .venv/bin/python) python)

.PHONY: help install fmt test test-slow lint web-test build serve dev-api dev-web sweep sweep-small analyze bench e2e demo report e2e-static

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s
", $$1, $$2}'

install:         ## dev install (CPU torch, then the package with dev + server extras, then web deps)
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev]"
	cd web && npm ci

fmt:             ## format Python with ruff (not enforced in CI yet)
	$(PY) -m ruff format twinlens server tests experiments

test:            ## Python unit, reproduction, twin and API tests
	$(PY) -m pytest -q

test-slow:       ## also the HF-parity test (needs ~1 GB extra RAM)
	$(PY) -m pytest -q -m slow

lint:            ## ruff + eslint + tsc, as in CI
	$(PY) -m ruff check twinlens server tests experiments
	cd web && npx tsc -b && npx eslint src

web-test:
	cd web && npx tsc -b && npm test

build:           ## build the dashboard into web/dist (served by the API)
	cd web && npm run build

serve: build     ## API + dashboard on http://localhost:8000
	$(PY) -m uvicorn server.app:app --port 8000

dev-api:
	$(PY) -m uvicorn server.app:app --port 8000 --reload

dev-web:         ## Vite dev server on :5173 (proxies /api to :8000)
	cd web && npm run dev

sweep:           ## full fidelity sweep (resumable, ~30-40 min on a laptop CPU)
	$(PY) experiments/fidelity_sweep.py --n-top 3 --n-random 3

sweep-small:     ## 1 prompt per family (~4 min)
	$(PY) experiments/fidelity_sweep.py --small

analyze:         ## every number in the report, from the committed sweep data
	$(PY) experiments/analyze.py
	$(PY) experiments/analyze_generalization.py
	$(if $(wildcard experiments/results/fidelity_atp.parquet),$(PY) experiments/analyze_atp.py,@echo "analyze_atp.py skipped: run experiments/atp_baseline.py first")
	$(if $(wildcard experiments/results/fidelity_atp_gemma2.parquet),$(PY) experiments/analyze_atp.py --model gemma2,@echo "gemma2 AtP skipped: no fidelity_atp_gemma2.parquet")

bench:
	$(PY) bench/latency.py

e2e:             ## needs `make serve` running
	cd web && npx playwright test

demo:            ## precompute the static live demo (web/public/demo), ~20 min
	$(PY) experiments/build_demo.py

report:          ## report web page + PDF (web/public/report, docs/TwinLens_report.pdf)
	$(PY) experiments/build_report.py
	cd web && node scripts/report-pdf.mjs

e2e-static:      ## static-demo e2e (serve web/dist-static under /twinlens/ on :8001)
	cd web && npx playwright test -c playwright.static.config.ts
