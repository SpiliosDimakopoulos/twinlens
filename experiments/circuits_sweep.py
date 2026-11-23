"""D10-A (pruned circuits) and D10-B (dose) sweep, as pre-registered in DECISIONS.md D10.

    python experiments/circuits_sweep.py --tc clt --phase atp  [--per-family 20] [--tag pilot]
    python experiments/circuits_sweep.py --tc clt --phase main [--per-family 20] [--tag pilot]

Runs on a server (Gemma-2-2B on CPU), with circuit-tracer installed from the PR #123 branch
(``attribute(..., freeze_attention=, freeze_layernorm=)``). Prompts: the first ``--per-family``
of each family of ``experiments/prompts_external_gemma2.json`` (D9).

``--phase atp`` stores, per prompt, the gradient of the demeaned answer logit with respect to
every layer's feature-output hook of the *real* model. The replacement model detaches the MLP
path and the norms for attribution, so this phase first removes its permanent hooks; it is a
separate process because those hooks cannot be put back. ``--phase main`` then builds the three
graph variants, ranks the candidate features, and runs the interventions. Both phases resume:
a prompt whose output exists is skipped. Output: ``results/raw_d10_<tc><tag>/``.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from common import RESULTS, ROOT, write_parquet

from twinlens.ct import graph_math as gm
from twinlens.ct import interventions as iv

KS = (10, 30, 100, 300, 1000)
DOSE = (-2.0, 0.0, 2.0, 5.0, 10.0)
N_RANDOM_ORDERS = 5
N_DOSE = 10
VARIANTS = {
    "G1": {"freeze_attention": True, "freeze_layernorm": True},
    "G2": {"freeze_attention": True, "freeze_layernorm": False},
    "G3": {"freeze_attention": False, "freeze_layernorm": False},
}
TRUTHS = {"T1": False, "T2": True}  # value of feature_intervention's freeze_attention
TC_REF = {"clt": "mntss/clt-gemma-2-2b-426k", "plt": "gemma"}
HF_MODEL = "unsloth/gemma-2-2b"  # ungated mirror of google/gemma-2-2b


def load_model(tc: str):
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cpu = torch.device("cpu")
    transcoders, _ = load_transcoder_from_hub(
        TC_REF[tc], device=cpu, dtype=torch.float32, lazy_encoder=False, lazy_decoder=True
    )
    hf = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-2-2b",
        transcoders,
        backend="transformerlens",
        hf_model=hf,
        tokenizer=tok,
        device=cpu,
        dtype=torch.float32,
    )
    del hf
    model.eval()
    return model


def prompts(per_family: int) -> list[dict]:
    data = json.loads((ROOT / "experiments" / "prompts_external_gemma2.json").read_text("utf-8"))
    out = []
    for fam, block in data["families"].items():
        for i, p in enumerate(block["prompts"][:per_family]):
            out.append({"pid": f"{fam}-{i:02d}", "family": fam, **p})
    return out


# ------------------------------------------------------------------ phase atp
def atp_grads(model, prompt: str, answer: int) -> torch.Tensor:
    """∂m/∂(feature-output hook) of the real model, [n_layers, n_pos, d_model]."""
    # circuit-tracer's feature-output hook is a gradient hook point ("<hook>.hook_out_grad")
    # called only by the permanent hooks removed for this phase; the edit lands on its parent.
    hook = model.feature_output_hook.removesuffix(".hook_out_grad")
    if f"blocks.0.{hook}" not in model.hook_dict:
        raise KeyError(f"no hook point blocks.0.{hook}")
    store: dict[int, torch.Tensor] = {}

    def make(layer: int):
        def fn(x, hook):  # a zero leaf, not a detached copy, keeps paths through later MLPs
            leaf = torch.zeros_like(x, requires_grad=True)
            store[layer] = leaf
            return x + leaf

        return fn

    tokens = model.ensure_tokenized(prompt)
    hooks = [(f"blocks.{l}.{hook}", make(l)) for l in range(model.cfg.n_layers)]
    with torch.enable_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)[0, -1]
        (logits[answer] - logits.mean()).backward()
    return torch.stack([store[l].grad[0] for l in range(model.cfg.n_layers)])


def atp_per_unit(model, grads: torch.Tensor, feats: list[tuple[int, int, int]]) -> np.ndarray:
    """AtP effect on m of one unit of each feature's activation (all its output layers)."""
    out = np.zeros(len(feats))
    by_layer: dict[int, list[int]] = {}
    for i, (l, _, _) in enumerate(feats):
        by_layer.setdefault(l, []).append(i)
    for l, idx in by_layer.items():
        ids = torch.tensor([feats[i][2] for i in idx])
        pos = torch.tensor([feats[i][1] for i in idx])
        dec = model.transcoders._get_decoder_vectors(l, ids).float()
        if dec.ndim == 2:  # per-layer transcoder writes only to its own layer
            out[idx] = (grads[l, pos] * dec).sum(-1).numpy()
        else:  # cross-layer: [n, n_out_layers, d], output layers l .. n_layers - 1
            g = grads[l:, pos].transpose(0, 1)  # [n, n_out_layers, d]
            out[idx] = (g * dec).sum((-1, -2)).numpy()
    return out


# ------------------------------------------------------------------ phase main
def run_graphs(model, p: dict, max_nodes, batch_size: int):
    from circuit_tracer import attribute
    from circuit_tracer.graph import prune_graph

    info, stats = {}, []
    for v, flags in VARIANTS.items():
        t = time.time()
        g = attribute(
            p["prompt"],
            model,
            attribution_targets=torch.tensor([p["answer_id"]]),
            max_feature_nodes=max_nodes,
            batch_size=batch_size,
            **flags,
        )
        secs = time.time() - t
        feats, acts = gm.feature_nodes(g)
        n = len(feats)
        pr = prune_graph(g)
        info[v] = {
            "feats": [tuple(x) for x in feats.tolist()],
            "acts": acts.float().numpy(),
            "infl": gm.node_influence(g).float().numpy(),
            "total": gm.total_effects(g).numpy(),
            "direct": gm.direct_contributions(g).float().numpy(),
            "pruned": [tuple(x) for x in feats[pr.node_mask[:n].cpu()].tolist()],
        }
        stats.append(
            {
                "variant": v,
                "seconds": secs,
                "n_active": len(g.active_features),
                "n_feature_nodes": n,
                "n_pruned_features": len(info[v]["pruned"]),
                "n_pruned_edges": int(pr.edge_mask.sum()),
            }
        )
        del g, pr
    return info, stats


def main_prompt(model, p: dict, grads: torch.Tensor, args) -> dict[str, pd.DataFrame]:
    info, stats = run_graphs(model, p, args.max_feature_nodes, args.batch_size)
    cand = sorted(set().union(*(set(x["feats"]) for x in info.values())))
    index = {f: i for i, f in enumerate(cand)}
    nc = len(cand)
    act = np.zeros(nc)
    per = {v: {"infl": np.zeros(nc), "total": np.zeros(nc)} for v in VARIANTS}
    direct = np.zeros(nc)
    for v, x in info.items():
        idx = [index[f] for f in x["feats"]]
        act[idx] = x["acts"]
        per[v]["infl"][idx] = x["infl"]
        per[v]["total"][idx] = x["total"]
        if v == "G1":
            direct[idx] = x["direct"]
    atp = atp_per_unit(model, grads, cand)

    rankings = {
        "R1": np.argsort(-per["G1"]["infl"], kind="stable"),
        "R2": np.argsort(-per["G2"]["infl"], kind="stable"),
        "R3": np.argsort(-per["G3"]["infl"], kind="stable"),
        "R4": np.argsort(-np.abs(act * atp), kind="stable"),
        "R5": np.argsort(-np.abs(direct), kind="stable"),
    }
    for s in range(N_RANDOM_ORDERS):
        order = list(range(nc))
        random.Random(f"{p['pid']}-{s}").shuffle(order)
        rankings[f"R6_{s}"] = np.array(order)
    ks = sorted({min(k, nc) for k in KS})
    a, c = p["answer_id"], p["contrast_id"]
    clean = iv.metrics(iv.last_logits(model, p["prompt"]), a, c)

    curves, dose = [], []
    for truth, frozen in TRUTHS.items():
        none = iv.metrics(iv.ablate(model, p["prompt"], cand, freeze_attention=frozen), a, c)
        base = {
            "pid": p["pid"],
            "truth": truth,
            "n_cand": nc,
            **{f"full_{k}": v for k, v in clean.items()},
            **{f"none_{k}": v for k, v in none.items()},
        }
        for name, order in rankings.items():
            cur = iv.faithfulness_curve(
                model,
                p["prompt"],
                cand,
                list(order),
                ks,
                a,
                c,
                freeze_attention=frozen,
                reference=(clean, none),
            )
            for k, kp, rm in zip(cur["k"], cur["keep"], cur["remove"], strict=True):
                curves.append(
                    {
                        **base,
                        "ranking": name,
                        "k": k,
                        **{f"keep_{q}": v for q, v in kp.items()},
                        **{f"remove_{q}": v for q, v in rm.items()},
                    }
                )
        for v, x in info.items():  # the default prune_graph node set of each variant
            keep_set = set(x["pruned"])
            rest = [f for f in cand if f not in keep_set]
            kp = iv.metrics(iv.ablate(model, p["prompt"], rest, freeze_attention=frozen), a, c)
            rm = iv.metrics(
                iv.ablate(model, p["prompt"], sorted(keep_set), freeze_attention=frozen), a, c
            )
            curves.append(
                {
                    **base,
                    "ranking": f"prune_{v}",
                    "k": len(keep_set),
                    **{f"keep_{q}": val for q, val in kp.items()},
                    **{f"remove_{q}": val for q, val in rm.items()},
                }
            )

        top = list(rankings["R1"][:N_DOSE])
        rest = [i for i in range(nc) if i not in set(top)]
        rnd = random.Random(f"{p['pid']}-dose").sample(rest, min(N_DOSE, len(rest)))
        for sel, idxs in (("top", top), ("random", rnd)):
            for i in idxs:
                f, a0 = cand[i], float(act[i])
                for s in DOSE:
                    m = iv.metrics(
                        iv.last_logits(model, p["prompt"], [(f, s * a0)], freeze_attention=frozen),
                        a,
                        c,
                    )
                    da = (s - 1.0) * a0
                    dose.append(
                        {
                            "pid": p["pid"],
                            "truth": truth,
                            "selection": sel,
                            "layer": f[0],
                            "pos": f[1],
                            "feature": f[2],
                            "act0": a0,
                            "s": s,
                            "real_dm": m["m"] - clean["m"],
                            "real_ddld": m["dld"] - clean["dld"],
                            **{f"pred_{v}": da * per[v]["total"][i] for v in VARIANTS},
                            "pred_atp": da * atp[i],
                            **{f"in_{v}": f in set(info[v]["feats"]) for v in VARIANTS},
                        }
                    )
    st = pd.DataFrame(stats).assign(pid=p["pid"], n_cand=nc)
    return {"curves": pd.DataFrame(curves), "dose": pd.DataFrame(dose), "graphs": st}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc", choices=list(TC_REF), required=True)
    ap.add_argument("--phase", choices=["atp", "main"], required=True)
    ap.add_argument("--per-family", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-feature-nodes", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-minutes", type=float, default=None, help="stop cleanly after this")
    args = ap.parse_args()
    out = RESULTS / f"raw_d10_{args.tc}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    model = load_model(args.tc)
    if args.phase == "atp":
        model.reset_hooks(including_permanent=True)  # the real model, every derivative live
    print(f"loaded in {time.time() - t_start:.0f}s", flush=True)
    for p in prompts(args.per_family):
        if args.max_minutes and time.time() - t_start > 60 * args.max_minutes:
            print("time budget reached; rerun to resume", flush=True)
            break
        t = time.time()
        if args.phase == "atp":
            path = out / f"{p['pid']}_grads.pt"
            if path.exists():
                continue
            tokens = model.ensure_tokenized(p["prompt"])
            with torch.no_grad():
                top1 = int(model(tokens)[0, -1].argmax())
            grads = atp_grads(model, p["prompt"], p["answer_id"])
            torch.save({"grads": grads, "top1_ok": top1 == p["answer_id"]}, path)
        else:
            if (out / f"{p['pid']}_curves.parquet").exists():
                continue
            g = torch.load(out / f"{p['pid']}_grads.pt")
            res = main_prompt(model, p, g["grads"], args)
            res["graphs"]["top1_ok"] = g["top1_ok"]
            for kind in ("graphs", "dose", "curves"):  # curves last: it marks the prompt done
                write_parquet(res[kind], out / f"{p['pid']}_{kind}.parquet")
        print(f"{args.phase} {p['pid']}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
