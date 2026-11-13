# Hugging Face Space for TwinLens (CPU basic, free tier).
# The Space repo holds only this file and README.md; the source comes from GitHub at REF.
# GPT-2 and its 12 transcoders (~2.3 GB) are downloaded at build time and baked into the image,
# so a sleeping Space wakes without re-downloading them and needs no paid persistent storage.
FROM python:3.12-slim AS src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG REF=main
RUN git clone https://github.com/SpiliosDimakopoulos/twinlens /src && git -C /src checkout "$REF"

FROM node:22-slim AS web
WORKDIR /web
COPY --from=src /src/web/ ./
RUN npm ci && npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 TWINLENS_CACHE=/cache
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY --from=src /src/pyproject.toml /src/README.md ./
COPY --from=src /src/twinlens/ twinlens/
COPY --from=src /src/server/ server/
COPY --from=src /src/experiments/results/summary.json /src/experiments/results/trust_model.json experiments/results/
RUN pip install ".[server]"
# One audit downloads GPT-2 and all 12 transcoders into /cache.
RUN twinlens audit "When Mary and John went to the store, John gave a drink to" --top 1 \
    && chmod -R a+rwX /cache
COPY --from=web /web/dist web/dist
ENV HF_HUB_OFFLINE=1 \
    TWINLENS_WS_ORIGINS=https://spiliosdmk-twinlens.hf.space
RUN useradd --create-home --uid 1000 user
USER user
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
