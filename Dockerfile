# TwinLens: API + built dashboard in one image (CPU only).
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 TWINLENS_CACHE=/cache
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY twinlens/ twinlens/
COPY server/ server/
COPY experiments/ experiments/
RUN pip install .
COPY --from=web /web/dist web/dist
# GPT-2 (~0.5 GB) and the 12 transcoders (~1.8 GB) download on first use into /cache;
# mount a volume there to keep them between runs.
VOLUME /cache
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
