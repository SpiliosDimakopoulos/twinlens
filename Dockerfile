# TwinLens: API + built dashboard in one image (CPU only).
FROM node:26-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 TWINLENS_CACHE=/cache
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# Dependency layer: only reinstalls when pyproject.toml changes, not on source edits.
COPY pyproject.toml ./
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['server']))" > /tmp/req.txt \
    && pip install -r /tmp/req.txt && rm /tmp/req.txt
COPY README.md ./
COPY twinlens/ twinlens/
COPY server/ server/
# The API reads only these two files at runtime.
COPY experiments/results/summary.json experiments/results/trust_model.json experiments/results/
RUN pip install ".[server]" && useradd --create-home --uid 10001 app && mkdir -p /cache && chown app /cache
COPY --from=web /web/dist web/dist
# GPT-2 (~0.5 GB) and the 12 transcoders (~1.8 GB) download on first use into /cache;
# mount a volume there to keep them between runs.
VOLUME /cache
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
