PY := .venv/Scripts/python

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check twinlens server tests

dev-api:
	$(PY) -m uvicorn server.app:app --port 8000

dev-web:
	cd web && npm run dev
