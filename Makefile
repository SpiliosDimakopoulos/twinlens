PY := .venv/Scripts/python

.PHONY: test lint web-test build serve dev-api dev-web sweep sweep-small analyze bench e2e

test:            ## Python unit, reproduction, twin and API tests
	$(PY) -m pytest -q

test-slow:       ## also the HF-parity test (needs ~1 GB extra RAM)
	$(PY) -m pytest -q -m slow

lint:
	$(PY) -m ruff check twinlens server tests experiments

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

analyze:
	$(PY) experiments/analyze.py

bench:
	$(PY) bench/latency.py

e2e:             ## needs `make serve` running
	cd web && npx playwright test
