"""TwinLens API.

    uvicorn server.app:app --port 8000

The model holds shared hook state, so every request that touches it goes through
``model_session()`` (see ``server.state``).
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from twinlens.causal.attribution import direct_logit_attribution, logit_lens
from twinlens.causal.interventions import (
    DirectionEdit,
    HeadAblation,
    Intervention,
    NeuronEdit,
    Steering,
    build_hooks,
)
from twinlens.causal.patching import logit_diff_metric, patch_heads
from twinlens.circuits import known
from twinlens.circuits.datasets import ioi_batch
from twinlens.core.gpt2 import GPT2
from twinlens.fidelity.signals import reliability_signals
from twinlens.twin.replacement import FIX_LEVELS, TWIN_LEVELS, FeatureEdit

from .schemas import (  # noqa: F401  (re-exported for experiments/build_demo.py)
    D_SAE,
    MAX_CHARS,
    MAX_INTERVENTIONS,
    InterveneRequest,
    PatchRequest,
    RunRequest,
    WhatIfRequest,
)
from .state import LOCK, ROOT, get_trust, get_twin, model_session  # noqa: F401

MAX_TOKENS = 48
GEN_EXTRA = 40  # max tokens the WebSocket may generate
DEV_ORIGIN = "http://localhost:5173"  # Vite dev server
# Browsers do not apply CORS to WebSockets, so the WS check also admits the origins that serve
# the built app itself (same-origin); CORS only needs the cross-origin dev server.
CORS_ORIGINS = [DEV_ORIGIN]
WS_ORIGINS = {DEV_ORIGIN, "http://localhost:8000", "http://127.0.0.1:8000"}

app = FastAPI(title="TwinLens", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _tokens(text: str, limit: int = MAX_TOKENS) -> torch.Tensor:
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"text is longer than {MAX_CHARS} characters")
    toks = get_twin().model.to_tokens(text)
    if toks.shape[1] > limit:
        raise HTTPException(400, f"prompt is longer than {limit} tokens")
    return toks


def _single_token(model: GPT2, s: str) -> int:
    """``model.to_single_token`` with the caller's bad input reported as a 400."""
    try:
        return model.to_single_token(s)
    except ValueError as e:
        raise HTTPException(400, str(e)[:200]) from e


def _check_interventions(cfg, items: list[Intervention], n_pos: int) -> None:
    """Reject out-of-range indices with a 400 instead of failing inside a hook."""
    if len(items) > MAX_INTERVENTIONS:
        raise HTTPException(400, f"at most {MAX_INTERVENTIONS} interventions")

    def need(ok: bool, what: str) -> None:
        if not ok:
            raise HTTPException(400, f"{what} is out of range")

    for it in items:
        if isinstance(it, DirectionEdit):
            raise HTTPException(400, "direction edits are not accepted by the API")
        need(0 <= it.layer < cfg.n_layers, "layer")
        if isinstance(it, HeadAblation):
            need(0 <= it.head < cfg.n_heads, "head")
            if it.mode == "mean":
                raise HTTPException(400, "mean ablation is not available in the API")
        if isinstance(it, NeuronEdit):
            need(0 <= it.neuron < cfg.d_mlp, "neuron")
        if isinstance(it, Steering):
            _tokens(it.plus)
            _tokens(it.minus)
        positions = getattr(it, "positions", None)
        if positions is not None:
            ok = 0 < len(positions) <= n_pos and all(0 <= p < n_pos for p in positions)
            need(ok, "positions")


def _token_str(model: GPT2, i) -> str:
    return model.tokenizer.token_str(int(i))


def _topk(model: GPT2, logits: torch.Tensor, k: int = 8) -> list[dict]:
    probs = logits.softmax(-1)
    top = probs.topk(k)
    return [
        {
            "token": _token_str(model, i),
            "id": int(i),
            "prob": float(p),
            "logit": float(logits[i]),
        }
        for i, p in zip(top.indices, top.values, strict=True)
    ]


def _dlogp(lp_p: torch.Tensor, lp_q: torch.Tensor, token: int) -> float:
    """Change in log-probability of ``token`` going from distribution p to q."""
    return float(lp_q[token] - lp_p[token])


def _kl(lp_p: torch.Tensor, lp_q: torch.Tensor) -> float:
    """KL(p || q) from log-probabilities."""
    return float((lp_p.exp() * (lp_p - lp_q)).sum())


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
        "d_sae": D_SAE,
        "twin_levels": list(TWIN_LEVELS),
        "trust_model": get_trust() is not None,
    }


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    """Everything the dashboard shows for one prompt: tokens, prediction, logit lens,
    attention, per-component attribution."""
    toks = _tokens(req.prompt)
    with model_session() as twin:
        model = twin.model
        names = [f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)]
        logits, cache = model.run_with_cache(toks, names)
        last = logits[0, -1]
        target = int(last.argmax()) if not req.target else _single_token(model, req.target)
        lens = logit_lens(model, toks)
        labels, contrib = direct_logit_attribution(model, toks, model.W_E[target])
    return {
        "tokens": model.tokenizer.str_tokens(toks[0].tolist()),
        "top": _topk(model, last),
        "target": {"id": target, "token": _token_str(model, target)},
        "logit_lens": {
            "top_token": [[_token_str(model, i) for i in row[:, 0]] for row in lens["top_ids"]],
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
    _check_interventions(model.cfg, req.interventions, toks.shape[1])
    with model_session():
        before = model(toks, last_only=True)[0, -1]
        with model.hooks(build_hooks(model, req.interventions)):
            after = model(toks, last_only=True)[0, -1]
    return {
        "before": _topk(model, before),
        "after": _topk(model, after),
        "kl": _kl(before.log_softmax(-1), after.log_softmax(-1)),
    }


@app.post("/api/twin/explain")
def explain(req: RunRequest) -> dict:
    """Top features (with labels) by direct effect on the target logit: the reasoning path."""
    toks = _tokens(req.prompt)
    with model_session() as twin:
        model = twin.model
        trace = twin.trace(toks)
        predicted = int(trace.logits[-1].argmax())
        target = _single_token(model, req.target) if req.target else predicted
        rows = twin.attribute_logit(trace, target, top=24, per_cell=3)
        recon = {
            layer: float(trace.error[layer][0, -1].norm() / trace.mlp_out[layer][0, -1].norm())
            for layer in range(model.cfg.n_layers)
        }
    tokens = model.tokenizer.str_tokens(toks[0].tolist())
    for r in rows:
        r["token"] = tokens[r["pos"]]
        r["label"] = twin.feature_label(r["layer"], r["feature"])
    return {
        "tokens": tokens,
        "target": _token_str(model, target),
        "features": rows,
        "error_fraction": recon,
    }


def whatif_payloads(toks: torch.Tensor, edits: list[FeatureEdit]) -> list[dict]:
    """Batched what-if: every twin level (plus the proposed fixes) and the real model for each
    edit, with reliability signals and the trust score. The caller must hold the model lock
    (i.e. be inside ``model_session()``)."""
    assert LOCK.locked(), "whatif_payloads must run inside model_session()"
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
                "dlogp": _dlogp(lp_clean, lp, target),
                "kl": _kl(lp_clean, lp),
            }
        direct = effects[edit.layer][edit.pos, edit.feature].item()
        signals = reliability_signals(twin, trace, edit, direct)
        payloads.append(
            {
                "target": _token_str(model, target),
                "clean": _topk(model, clean, 5),
                "levels": levels,
                "signals": signals,
                "trust": trust.predict(signals) if trust else None,
                "trust_factors": trust.explain(signals) if trust else [],
                "label": twin.feature_label(edit.layer, edit.feature),
            }
        )
    return payloads


@app.post("/api/twin/whatif")
def whatif(req: WhatIfRequest) -> dict:
    """The twin predicts a feature edit's effect at every level; then the real model runs it."""
    toks = _tokens(req.prompt)
    if req.pos >= toks.shape[1]:
        raise HTTPException(400, "pos is out of range")
    if req.layer not in get_twin().layers:
        raise HTTPException(400, "layer has no transcoder")
    with model_session():
        return whatif_payloads(toks, [FeatureEdit(req.layer, req.pos, req.feature, req.value)])[0]


@app.post("/api/patch")
def patch(req: PatchRequest) -> dict:
    model = get_twin().model
    clean, corrupt = _tokens(req.clean), _tokens(req.corrupt)
    if clean.shape != corrupt.shape:
        raise HTTPException(400, "clean and corrupt prompts must have the same number of tokens")
    c, w = _single_token(model, req.correct), _single_token(model, req.wrong)
    with model_session():
        grid = patch_heads(
            model, clean, corrupt, logit_diff_metric(torch.tensor([c]), torch.tensor([w]))
        )
    return {"heads": grid.tolist()}


def _induction_study(model: GPT2) -> dict:
    scores = known.induction_scores(model, n_batch=4)
    return {
        "grid": scores.tolist(),
        "known": sorted(known.INDUCTION_HEADS),
        "title": "Induction heads (Olsson et al., 2022)",
    }


def _ioi_study(model: GPT2) -> dict:
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


CASE_STUDIES = {"induction": _induction_study, "ioi": _ioi_study}


@app.get("/api/case-studies/{name}")
def case_study(name: str) -> dict:
    if name not in CASE_STUDIES:
        raise HTTPException(404, f"unknown case study {name!r}")
    return _case_study(name)


@lru_cache(maxsize=2)
def _case_study(name: str) -> dict:
    """Deterministic, so computed once per process (each costs dozens of forward passes)."""
    with model_session() as twin:
        return CASE_STUDIES[name](twin.model)


@app.get("/api/fidelity/summary")
def fidelity_summary() -> dict:
    path = ROOT / "experiments" / "results" / "summary.json"
    if not path.exists():
        raise HTTPException(404, "study results are not available in this build")
    return json.loads(path.read_text())


def _gen_step(toks: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """One generation step under the model lock (worker thread, never the event loop)."""
    with model_session() as twin:
        lens = logit_lens(twin.model, toks, k=1)
        logits = twin.model(toks, last_only=True)[0, -1]
    return logits, lens


_WS_USAGE = "expected {prompt: str, max_new_tokens: int}"


async def _reject(ws: WebSocket, msg: str) -> None:
    await ws.send_json({"error": msg})
    await ws.close(code=1003)


def _parse_request(msg) -> tuple[str, int] | None:
    """``(prompt, max_new_tokens)`` from a client message, or None if malformed."""
    if not isinstance(msg, dict):
        return None
    prompt, n = msg.get("prompt"), msg.get("max_new_tokens", 20)
    if not isinstance(prompt, str) or not prompt or not isinstance(n, int):
        return None
    return prompt, n


@app.websocket("/ws/generate")
async def generate(ws: WebSocket) -> None:
    """Stream greedy generation token by token, with the logit-lens view of each step."""
    origin = ws.headers.get("origin")
    if origin is not None and origin not in WS_ORIGINS:
        await ws.close(code=1008)  # browsers do not apply CORS to WebSockets
        return
    await ws.accept()
    try:
        parsed = _parse_request(await ws.receive_json())
        if parsed is None:
            await _reject(ws, _WS_USAGE)
            return
        prompt, n = parsed
        try:
            toks = await asyncio.to_thread(_tokens, prompt)
        except HTTPException as e:
            await _reject(ws, e.detail)
            return
        model = get_twin().model
        for _ in range(max(0, min(n, GEN_EXTRA))):
            logits, lens = await asyncio.to_thread(_gen_step, toks)
            nxt = int(logits.argmax())
            await ws.send_json(
                {
                    "token": _token_str(model, nxt),
                    "top": _topk(model, logits, 5),
                    "lens": [_token_str(model, i) for i in lens["top_ids"][:, -1, 0]],
                    "lens_prob": lens["final_prob"][:, -1].tolist(),
                }
            )
            toks = torch.cat([toks, torch.tensor([[nxt]])], dim=1)
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
