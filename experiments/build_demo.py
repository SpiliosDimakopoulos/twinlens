"""Precompute everything the dashboard needs for a fixed set of prompts -> static live demo.

    python experiments/build_demo.py [--max-minutes 8]

Writes ``web/public/demo/manifest.json`` plus one file per prompt. Payloads come from the real
API code (FastAPI TestClient for run/explain/case studies/generation, the server's batched
``whatif_payloads`` for what-ifs), so the static demo shows exactly what the live server would.
Resumable: prompts already written are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from server import app as server  # noqa: E402
from twinlens.causal.interventions import HeadAblation, build_hooks  # noqa: E402
from twinlens.twin.replacement import FeatureEdit  # noqa: E402

OUT = ROOT / "web" / "public" / "demo"
PRESETS = [0.0, 0.5, 2.0, 3.0]
PROMPTS = [
    ("Michael Jordan plays the sport of", "Factual recall"),
    ("When Mary and John went to the store, John gave a drink to", "Indirect object (IOI)"),
    (
        "Mr. Dursley was the director of a firm called Grunnings. Mr. Dursley was the director "
        "of a firm called",
        "Induction / copying",
    ),
    ("def add(a, b):\n    return", "Code"),
    ("Monday, Tuesday, Wednesday,", "Sequence"),
    ("I loved this movie. The acting was", "Sentiment"),
]


def rounded(x, nd: int = 4):
    """Shrink JSON: round floats, recursively."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [rounded(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    return x


def head_ablations(prompt: str) -> dict:
    """Zero-ablate each of the 144 heads (batched): before/after top-5 and KL."""
    twin = server.get_twin()
    model = twin.model
    toks = server._tokens(prompt)
    H, L = model.cfg.n_heads, model.cfg.n_layers
    out: dict[str, dict] = {}
    with server.LOCK:
        before = model(toks, last_only=True)[0, -1]
        lp_b = before.log_softmax(-1)
        for layer in range(L):
            rows = toks.expand(H, -1)

            def fn(z: torch.Tensor, _: str) -> torch.Tensor:
                z = z.clone()
                for h in range(H):
                    z[h, :, h] = 0
                return z

            with model.hooks([(f"blocks.{layer}.attn.hook_z", fn)]):
                after = model(rows, last_only=True)[:, -1]
            for h in range(H):
                lp_a = after[h].log_softmax(-1)
                out[f"{layer}-{h}"] = {
                    "after": server._topk(model, after[h], 5),
                    "kl": float((lp_b.exp() * (lp_b - lp_a)).sum()),
                }
    # sanity: the batched hook equals the API's single-head ablation
    spec = [HeadAblation(layer=9, head=9)]
    with server.LOCK, model.hooks(build_hooks(model, spec)):
        ref = model(toks, last_only=True)[0, -1].log_softmax(-1)
    kl_ref = float((lp_b.exp() * (lp_b - ref)).sum())
    assert abs(kl_ref - out["9-9"]["kl"]) < 1e-4, (kl_ref, out["9-9"]["kl"])
    return {"before": server._topk(model, before, 5), "heads": out}


def generation(client: TestClient, prompt: str, n: int = 16) -> list[dict]:
    steps = []
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_json({"prompt": prompt, "max_new_tokens": n})
        while True:
            msg = ws.receive_json()
            if msg.get("done"):
                break
            steps.append(msg)
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    client = TestClient(server.app)

    shared = OUT / "shared.json"
    if not shared.exists():
        shared.write_text(
            json.dumps(
                rounded(
                    {
                        "model": client.get("/api/model").json(),
                        "summary": client.get("/api/fidelity/summary").json(),
                        "case_studies": {
                            n: client.get(f"/api/case-studies/{n}").json()
                            for n in ("induction", "ioi")
                        },
                    }
                )
            )
        )
        print("shared.json", flush=True)

    for i, (prompt, kind) in enumerate(PROMPTS):
        path = OUT / f"p{i}.json"
        if path.exists():
            continue
        if args.max_minutes and time.perf_counter() - t0 > args.max_minutes * 60:
            print("time budget reached; rerun to resume", flush=True)
            return
        t = time.perf_counter()
        run = client.post("/api/run", json={"prompt": prompt}).json()
        explain = client.post("/api/twin/explain", json={"prompt": prompt}).json()
        toks = server._tokens(prompt)
        edits, keys = [], []
        for f in explain["features"]:
            for m in PRESETS:
                edits.append(FeatureEdit(f["layer"], f["pos"], f["feature"], f["activation"] * m))
                keys.append((f"{f['layer']}-{f['pos']}-{f['feature']}", str(m)))
        whatif: dict[str, dict] = {}
        for s in range(0, len(edits), 16):
            with server.LOCK:
                payloads = server.whatif_payloads(toks, edits[s : s + 16])
            for (fk, mk), pl in zip(keys[s : s + 16], payloads, strict=True):
                whatif.setdefault(fk, {})[mk] = pl
        data = {
            "prompt": prompt,
            "kind": kind,
            "run": run,
            "explain": explain,
            "whatif": whatif,
            "ablation": head_ablations(prompt),
            "generation": generation(client, prompt),
        }
        path.write_text(json.dumps(rounded(data)))
        print(
            f"p{i} {kind}: {len(edits)} what-ifs, {path.stat().st_size / 1e6:.1f} MB, "
            f"{time.perf_counter() - t:.0f}s",
            flush=True,
        )

    manifest = {
        "prompts": [
            {"prompt": p, "kind": k, "file": f"p{i}.json"} for i, (p, k) in enumerate(PROMPTS)
        ],
        "presets": PRESETS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    print("manifest.json written", flush=True)


if __name__ == "__main__":
    main()
