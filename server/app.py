"""TwinLens API.

    uvicorn server.app:app --port 8000

The model holds shared hook state, so every request that touches it takes ``LOCK``.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twinlens.causal.attribution import direct_logit_attribution, logit_lens
from twinlens.causal.interventions import Intervention, build_hooks
from twinlens.causal.patching import logit_diff_metric, patch_heads
from twinlens.circuits import known
from twinlens.circuits.datasets import ioi_batch
from twinlens.core.gpt2 import GPT2
from twinlens.fidelity.signals import reliability_signals
from twinlens.fidelity.trust import TrustModel
from twinlens.twin.replacement import FIX_LEVELS, TWIN_LEVELS, FeatureEdit, Twin

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
MAX_TOKENS = 48

app = FastAPI(title="TwinLens", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_twin() -> Twin:
    return Twin(GPT2.load("gpt2"))


@lru_cache(maxsize=1)
def get_trust() -> TrustModel | None:
    path = ROOT / "experiments" / "results" / "trust_model.json"
    return TrustModel.load(path) if path.exists() else None


def _tokens(text: str) -> torch.Tensor:
    model = get_twin().model
    toks = model.to_tokens(text)
    if toks.shape[1] > MAX_TOKENS:
        raise HTTPException(400, f"prompt is longer than {MAX_TOKENS} tokens")
    return toks


def _topk(model: GPT2, logits: torch.Tensor, k: int = 8) -> list[dict]:
    probs = logits.softmax(-1)
    top = probs.topk(k)
    return [
        {
            "token": model.tokenizer.token_str(i),
            "id": int(i),
            "prob": float(p),
            "logit": float(logits[i]),
        }
        for i, p in zip(top.indices, top.values, strict=True)
    ]


# --------------------------------------------------------------------- schemas
class RunRequest(BaseModel):
    prompt: str = Field(min_length=1)
    target: str | None = None  # token to attribute; defaults to the model's prediction


class InterveneRequest(BaseModel):
    prompt: str
    interventions: list[Intervention]


class WhatIfRequest(BaseModel):
    prompt: str
    layer: int = Field(ge=0, le=11)
    pos: int = Field(ge=0)
    feature: int = Field(ge=0)
    value: float = 0.0


class PatchRequest(BaseModel):
    clean: str
    corrupt: str
    correct: str
    wrong: str


# -------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/model")
def model_info() -> dict:
    cfg = get_twin().model.cfg
    return {
        "name": "gpt2-small",
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp,
        "d_sae": 24576,
        "twin_levels": list(TWIN_LEVELS),
        "trust_model": get_trust() is not None,
    }


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    """Everything the dashboard shows for one prompt: tokens, prediction, logit lens,
    attention, per-component attribution."""
    twin = get_twin()
    model = twin.model
    toks = _tokens(req.prompt)
    with LOCK:
        names = [f"blocks.{l}.attn.hook_pattern" for l in range(model.cfg.n_layers)]
        logits, cache = model.run_with_cache(toks, names)
        last = logits[0, -1]
        target = int(last.argmax()) if not req.target else model.to_single_token(req.target)
        lens = logit_lens(model, toks)
        labels, contrib = direct_logit_attribution(model, toks, model.W_E[target])
    return {
        "tokens": model.tokenizer.str_tokens(toks[0].tolist()),
        "top": _topk(model, last),
        "target": {"id": target, "token": model.tokenizer.token_str(target)},
        "logit_lens": {
            "top_token": [
                [model.tokenizer.token_str(i) for i in row[:, 0]] for row in lens["top_ids"]
            ],
            "top_prob": lens["top_probs"][..., 0].tolist(),
            "final_prob": lens["final_prob"].tolist(),
            "final_rank": lens["final_rank"].tolist(),
        },
        "attention": [cache[n][0].tolist() for n in names],  # [layer][head][q][k]
        "dla": [
            {"component": lab, "value": float(v)} for lab, v in zip(labels, contrib, strict=True)
        ],
    }


@app.post("/api/intervene")
def intervene(req: InterveneRequest) -> dict:
    model = get_twin().model
    toks = _tokens(req.prompt)
    with LOCK:
        before = model(toks, last_only=True)[0, -1]
        with model.hooks(build_hooks(model, req.interventions)):
            after = model(toks, last_only=True)[0, -1]
    lp_b, lp_a = before.log_softmax(-1), after.log_softmax(-1)
    return {
        "before": _topk(model, before),
        "after": _topk(model, after),
        "kl": float((lp_b.exp() * (lp_b - lp_a)).sum()),
    }


@app.post("/api/twin/explain")
def explain(req: RunRequest) -> dict:
    """Top features (with labels) by direct effect on the target logit: the reasoning path."""
    twin = get_twin()
    model = twin.model
    toks = _tokens(req.prompt)
    with LOCK:
        trace = twin.trace(toks)
        target = (
            int(trace.logits[-1].argmax()) if not req.target else model.to_single_token(req.target)
        )
        rows = twin.attribute_logit(trace, target, top=24, per_cell=3)
        recon = {
            l: float(trace.error[l][0, -1].norm() / trace.mlp_out[l][0, -1].norm())
            for l in range(model.cfg.n_layers)
        }
    tokens = model.tokenizer.str_tokens(toks[0].tolist())
    for r in rows:
        r["token"] = tokens[r["pos"]]
        r["label"] = twin.feature_label(r["layer"], r["feature"])
    return {
        "tokens": tokens,
        "target": model.tokenizer.token_str(target),
        "features": rows,
        "error_fraction": recon,
    }


def whatif_payloads(toks: torch.Tensor, edits: list[FeatureEdit]) -> list[dict]:
    """Batched what-if: every twin level (plus the proposed fixes) and the real model for each
    edit, with reliability signals and the trust score. Caller holds ``LOCK``."""
    twin = get_twin()
    model = twin.model
    trace = twin.trace(toks)
    clean = trace.logits[-1]
    target = int(clean.argmax())
    lp_clean = clean.log_softmax(-1)
    effects = twin.feature_effects(trace, target)
    outs = {mode: twin.run(trace, mode, edits) for mode in (*TWIN_LEVELS, *FIX_LEVELS, "real")}
    trust = get_trust()
    payloads = []
    for i, edit in enumerate(edits):
        levels = {}
        for mode, out in outs.items():
            lp = out[i].log_softmax(-1)
            levels[mode] = {
                "top": _topk(model, out[i], 5),
                "dlogp": float(lp[target] - lp_clean[target]),
                "kl": float((lp_clean.exp() * (lp_clean - lp)).sum()),
            }
        direct = effects[edit.layer][edit.pos, edit.feature].item()
        signals = reliability_signals(twin, trace, edit, direct)
        payloads.append({
            "target": model.tokenizer.token_str(target),
            "clean": _topk(model, clean, 5),
            "levels": levels,
            "signals": signals,
            "trust": trust.predict(signals) if trust else None,
            "trust_factors": trust.explain(signals) if trust else [],
            "label": twin.feature_label(edit.layer, edit.feature),
        })
    return payloads


@app.post("/api/twin/whatif")
def whatif(req: WhatIfRequest) -> dict:
    """The twin predicts a feature edit's effect at every level; then the real model runs it."""
    toks = _tokens(req.prompt)
    if req.pos >= toks.shape[1]:
        raise HTTPException(400, "pos is out of range")
    with LOCK:
        return whatif_payloads(toks, [FeatureEdit(req.layer, req.pos, req.feature, req.value)])[0]


@app.post("/api/patch")
def patch(req: PatchRequest) -> dict:
    model = get_twin().model
    clean, corrupt = _tokens(req.clean), _tokens(req.corrupt)
    if clean.shape != corrupt.shape:
        raise HTTPException(400, "clean and corrupt prompts must have the same number of tokens")
    c, w = model.to_single_token(req.correct), model.to_single_token(req.wrong)
    with LOCK:
        grid = patch_heads(
            model, clean, corrupt, logit_diff_metric(torch.tensor([c]), torch.tensor([w]))
        )
    return {"heads": grid.tolist()}


@app.get("/api/case-studies/{name}")
def case_study(name: str) -> dict:
    model = get_twin().model
    with LOCK:
        if name == "induction":
            scores = known.induction_scores(model, n_batch=4)
            return {
                "grid": scores.tolist(),
                "known": sorted(known.INDUCTION_HEADS),
                "title": "Induction heads (Olsson et al., 2022)",
            }
        if name == "ioi":
            batch = ioi_batch(model, n=16)
            dla = known.ioi_head_dla(model, batch)
            return {
                "grid": dla.tolist(),
                "known": sorted(known.NAME_MOVER_HEADS),
                "negative": sorted(known.NEGATIVE_NAME_MOVERS),
                "logit_diff": known.ioi_logit_diff(model, batch),
                "example": batch.prompts[0],
                "title": "Indirect object identification (Wang et al., 2022)",
            }
    raise HTTPException(404, f"unknown case study {name!r}")


@app.get("/api/fidelity/summary")
def fidelity_summary() -> dict:
    path = ROOT / "experiments" / "results" / "summary.json"
    if not path.exists():
        raise HTTPException(404, "run experiments/fidelity_sweep.py and analyze.py first")
    return json.loads(path.read_text())


@app.websocket("/ws/generate")
async def generate(ws: WebSocket) -> None:
    """Stream greedy generation token by token, with the logit-lens view of each step."""
    await ws.accept()
    try:
        msg = await ws.receive_json()
        model = get_twin().model
        text, n = msg["prompt"], min(int(msg.get("max_new_tokens", 20)), 40)
        for _ in range(n):
            with LOCK:
                toks = model.to_tokens(text)
                if toks.shape[1] > MAX_TOKENS + 40:
                    break
                lens = logit_lens(model, toks, k=1)
                logits = model(toks, last_only=True)[0, -1]
            nxt = int(logits.argmax())
            tok = model.tokenizer.token_str(nxt)
            await ws.send_json(
                {
                    "token": tok,
                    "top": _topk(model, logits, 5),
                    "lens": [model.tokenizer.token_str(int(i)) for i in lens["top_ids"][:, -1, 0]],
                    "lens_prob": lens["final_prob"][:, -1].tolist(),
                }
            )
            text += tok
        await ws.send_json({"done": True})
    except WebSocketDisconnect:
        return


# Serve the built frontend (web/dist) when present, so `make serve` is a single process.
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(DIST / "index.html")
