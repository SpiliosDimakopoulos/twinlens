"""How far can you trust this attribution graph? An audit of a circuit-tracer graph.

For every feature node it compares two first-order predictions of what ablating that feature
does to the graph's target logit:
- the graph's own prediction, the path sum through the graph (`graph_math.total_effects`);
- attribution patching through the real model, one backward pass (`real_gradient`).

On the TwinLens sweeps (report §4.8), attribution patching tracks real interventions at
Spearman ρ 0.95 on GPT-2 and on Gemma-2-2B (PLT and CLT), against 0.20–0.27 for today's graph.
The comparison therefore costs one backward pass and needs no interventions. A sample of nodes
is then ablated for real (`fast.Intervener`) to check both predictions on this prompt.

    from twinlens.ct.audit import audit_graph, to_markdown
    report = audit_graph(model, graph, n_check=20)
    print(to_markdown(report))
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from scipy.stats import spearmanr

from twinlens.ct import graph_math as gm
from twinlens.ct.fast import Intervener


@contextmanager
def real_model(model):
    """Temporarily remove the replacement model's permanent hooks, which detach the MLP path
    and the norms, so a backward pass is a gradient of the real model. Restored on exit."""
    flags = (model.freeze_attention, model.freeze_layernorm)
    model.reset_hooks(including_permanent=True)
    try:
        yield model
    finally:
        model.reset_hooks(including_permanent=True)
        model._configure_gradient_flow()
        model.setup()
        model.set_gradient_flow(*flags)


def load(model_name: str, transcoders: str, hf_model: str | None = None):
    """A circuit-tracer ReplacementModel on the TransformerLens backend (CPU, float32).
    `hf_model` loads the weights from another HF repo, e.g. an ungated mirror."""
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    cpu = torch.device("cpu")
    tcs, _ = load_transcoder_from_hub(
        transcoders, device=cpu, dtype=torch.float32, lazy_encoder=False, lazy_decoder=True
    )
    kw = {}
    if hf_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw = {
            "hf_model": AutoModelForCausalLM.from_pretrained(hf_model, dtype=torch.bfloat16),
            "tokenizer": AutoTokenizer.from_pretrained(hf_model),
        }
    return ReplacementModel.from_pretrained_and_transcoders(
        model_name, tcs, backend="transformerlens", device=cpu, dtype=torch.float32, **kw
    )


def real_gradient(model, prompt, token: int) -> torch.Tensor:
    """∂(demeaned logit of `token`)/∂(feature-output hook) of the real model,
    [n_layers, n_pos, d_model]. Zero leaves are added, so no path through later MLPs is cut."""
    hook = model.feature_output_hook.removesuffix(".hook_out_grad")
    store: dict[int, torch.Tensor] = {}

    def make(layer):
        def fn(x, hook):
            leaf = torch.zeros_like(x, requires_grad=True)
            store[layer] = leaf
            return x + leaf

        return fn

    with real_model(model):
        tokens = model.ensure_tokenized(prompt).reshape(1, -1)
        hooks = [(f"blocks.{l}.{hook}", make(l)) for l in range(model.cfg.n_layers)]
        with torch.enable_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)[0, -1]
            (logits[token] - logits.mean()).backward()
        return torch.stack([store[l].grad[0] for l in range(model.cfg.n_layers)])


def atp_per_unit(model, grads: torch.Tensor, feats) -> np.ndarray:
    """Attribution patching: effect on the target of one unit of each feature's activation."""
    out = np.zeros(len(feats))
    by_layer: dict[int, list[int]] = {}
    for i, (l, _, _) in enumerate(feats):
        by_layer.setdefault(int(l), []).append(i)
    for l, idx in by_layer.items():
        ids = torch.tensor([int(feats[i][2]) for i in idx])
        pos = torch.tensor([int(feats[i][1]) for i in idx])
        dec = model.transcoders._get_decoder_vectors(l, ids).float()
        if dec.ndim == 2:
            out[idx] = (grads[l, pos] * dec).sum(-1).numpy()
        else:
            out[idx] = (grads[l:, pos].transpose(0, 1) * dec).sum((-1, -2)).numpy()
    return out


def _rho(a, b) -> float:
    return float(spearmanr(a, b).statistic) if len(a) > 2 else float("nan")


def audit_graph(
    model, graph, n_check: int = 20, freeze_attention: bool = False, target: int = 0
) -> dict:
    """Audit `graph` (a circuit-tracer Graph built on `model`) for logit target `target`.

    Ablation effects on the demeaned target logit, per feature node:
    - `graph` = −a · path sum;
    - `atp` = −a · attribution patching.
    Then the `n_check` nodes the graph ranks highest and the `n_check` that attribution
    patching ranks highest are ablated one at a time for real: live attention by default, or
    circuit-tracer's frozen-attention intervention.
    """
    feats, acts = gm.feature_nodes(graph)
    feats = [tuple(int(x) for x in f) for f in feats.tolist()]
    a = acts.double().numpy()
    token = int(graph.logit_targets[target].vocab_idx)
    pred_graph = -a * gm.total_effects(graph, target).double().numpy()
    grads = real_gradient(model, graph.input_tokens, token)
    pred_atp = -a * atp_per_unit(model, grads, feats)

    top_g = list(np.argsort(-np.abs(pred_graph))[:n_check])
    top_a = list(np.argsort(-np.abs(pred_atp))[:n_check])
    check = sorted(set(top_g) | set(top_a))
    eng = Intervener(model, graph.input_tokens, [feats[i] for i in check], freeze_attention)
    values = torch.full((len(check) + 1, len(check)), float("nan"))
    for r in range(len(check)):
        values[r + 1, r] = 0.0
    logits = eng.run(values)
    m = (logits[:, token] - logits.mean(-1)).double().numpy()
    real = m[1:] - m[0]
    pg, pa = pred_graph[check], pred_atp[check]
    big = np.abs(real) > 0.05

    def sign(p):
        return float(np.mean(np.sign(p[big]) == np.sign(real[big]))) if big.any() else float("nan")

    k = min(10, len(feats))
    overlap = len(set(np.argsort(-np.abs(pred_graph))[:k]) & set(np.argsort(-np.abs(pred_atp))[:k]))
    nodes = [
        {
            "layer": feats[i][0],
            "pos": feats[i][1],
            "feature": feats[i][2],
            "act": float(a[i]),
            "graph": float(pred_graph[i]),
            "atp": float(pred_atp[i]),
            "real": float(r),
            "graph_ranks_top": bool(i in top_g),
            "atp_ranks_top": bool(i in top_a),
            "graph_wrong_sign": bool(abs(r) > 0.05 and np.sign(pred_graph[i]) != np.sign(r)),
        }
        for i, r in zip(check, real, strict=True)
    ]
    return {
        "prompt": graph.input_string,
        "target": graph.logit_targets[target].token_str,
        "n_feature_nodes": len(feats),
        "graph_vs_atp_spearman_all_nodes": _rho(pred_graph, pred_atp),
        f"top{k}_overlap_graph_atp": overlap / k,
        "checked": len(check),
        "truth": "frozen attention" if freeze_attention else "live attention",
        "graph_vs_real_spearman": _rho(pg, real),
        "atp_vs_real_spearman": _rho(pa, real),
        "graph_sign_agree": sign(pg),
        "atp_sign_agree": sign(pa),
        "n_checked_that_move": int(big.sum()),
        "nodes": sorted(nodes, key=lambda x: -abs(x["real"])),
    }


def to_markdown(r: dict, rows: int = 12) -> str:
    def f(x):
        return "n/a" if x != x else f"{x:.2f}"

    k = next(key for key in r if key.startswith("top") and key.endswith("_overlap_graph_atp"))
    lines = [
        f"## Graph audit: {r['prompt']!r} → {r['target']!r}",
        "",
        f"- **Feature nodes:** {r['n_feature_nodes']}.",
        "- **Graph against attribution patching, over all nodes:**",
        f"  - Spearman ρ {f(r['graph_vs_atp_spearman_all_nodes'])};",
        f"  - {k.split('_')[0]} overlap {f(r[k])}.",
        f"- **Checked by real ablation:** {r['checked']} nodes, {r['truth']}; "
        f"{r['n_checked_that_move']} move the logit by more than 0.05.",
        "",
        "| | ρ with real effect | sign agreement |",
        "|---|---|---|",
        f"| graph (path sums) | {f(r['graph_vs_real_spearman'])} | {f(r['graph_sign_agree'])} |",
        f"| attribution patching | {f(r['atp_vs_real_spearman'])} | {f(r['atp_sign_agree'])} |",
        "",
        "| layer | pos | feature | graph | AtP | real | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in r["nodes"][:rows]:
        note = "graph has the wrong sign" if n["graph_wrong_sign"] else ""
        lines.append(
            f"| {n['layer']} | {n['pos']} | {n['feature']} | {n['graph']:+.3f} | "
            f"{n['atp']:+.3f} | {n['real']:+.3f} | {note} |"
        )
    return "\n".join(lines)
