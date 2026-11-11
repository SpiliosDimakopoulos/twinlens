"""Command line: audit an attribution graph's strongest edges against the real model.

    twinlens audit "When Mary and John went to the store, John gave a drink to"
    twinlens audit "The Colosseum is located in the city of" --top 12 --edit double --json

For the ``top`` features with the largest direct effect on GPT-2's prediction (the edges an
attribution graph draws first), it makes the edit three ways and reports each:

- ``graph``: today's linear attribution graph (LayerNorm and attention frozen);
- ``fixed``: the same linear graph with the LayerNorm and attention Jacobians kept;
- ``gpt2``: the real model with the edit applied.

plus a trust score for the graph's claim, computed before the real model runs. Effects are the
change in log-probability of the predicted token (nats).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

EDITS = {"ablate": 0.0, "halve": 0.5, "double": 2.0}
TRUST_PATH = Path(__file__).resolve().parent / "fidelity" / "trust_model.json"

# Verdict thresholds (effects in nats). Below NO_EFFECT an effect counts as zero; a prediction
# is overstated above OVERSTATED times the real effect and understated below UNDERSTATED times.
NO_EFFECT = 0.01
OVERSTATED = 2.0
UNDERSTATED = 0.5
RATIO_FLOOR = 1e-3  # keeps the ratio finite when the real effect is ~0


def _verdict(pred: float, real: float) -> str:
    if abs(real) < NO_EFFECT and abs(pred) < NO_EFFECT:
        return "no effect"
    if (pred > 0) != (real > 0) and abs(real) >= NO_EFFECT:
        return "WRONG SIGN"
    ratio = abs(pred) / max(abs(real), RATIO_FLOOR)
    if ratio > OVERSTATED:
        return f"overstated {ratio:.0f}x" if ratio >= 10 else f"overstated {ratio:.1f}x"
    if ratio < UNDERSTATED:
        return "understated"
    return "ok"


class _Subscriptable:
    # The report used to be a plain dict and the published notebook still indexes it that way.
    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class Edge(_Subscriptable):
    layer: int
    pos: int
    feature: int
    token: str
    direct_effect: float
    graph: float  # effect predicted by the linear attribution graph
    fixed: float  # ... by the graph with the LayerNorm and attention Jacobians kept
    gpt2: float  # real effect
    graph_verdict: str
    fixed_verdict: str
    trust: float | None


@dataclass
class AuditReport(_Subscriptable):
    prompt: str
    prediction: str
    edit: str
    edges: list[Edge]


def audit(prompt: str, top: int = 8, edit: str = "ablate") -> AuditReport:
    """Audit the ``top`` strongest graph edges for ``prompt``. Loads GPT-2 and its transcoders
    (downloaded from the Hugging Face Hub on first use, ~2.3 GB)."""
    from twinlens.core.gpt2 import GPT2
    from twinlens.fidelity.signals import reliability_signals
    from twinlens.fidelity.trust import TrustModel
    from twinlens.twin.replacement import FeatureEdit, Twin

    twin = Twin(GPT2.load("gpt2"))
    model = twin.model
    toks = model.to_tokens(prompt)
    trace = twin.trace(toks)
    target = int(trace.logits[-1].argmax())
    rows = twin.attribute_logit(trace, target, top=top)
    effects = twin.feature_effects(trace, target)
    mult = EDITS[edit]
    edits = [FeatureEdit(r["layer"], r["pos"], r["feature"], r["activation"] * mult) for r in rows]
    lp0 = trace.logits[-1].log_softmax(-1)[target]
    out = {}
    for mode in ("linear", "linear_fulljac", "real"):
        logits = twin.run(trace, mode, edits)
        out[mode] = (logits.log_softmax(-1)[:, target] - lp0).tolist()
    trust = TrustModel.load(TRUST_PATH) if TRUST_PATH.exists() else None
    tokens = model.tokenizer.str_tokens(toks[0].tolist())
    edges = []
    for i, (r, e) in enumerate(zip(rows, edits, strict=True)):
        direct = effects[e.layer][e.pos, e.feature].item()
        sig = reliability_signals(twin, trace, e, direct)
        graph, fixed, real = out["linear"][i], out["linear_fulljac"][i], out["real"][i]
        edges.append(
            Edge(
                layer=r["layer"],
                pos=r["pos"],
                feature=r["feature"],
                token=tokens[r["pos"]],
                direct_effect=r["effect"],
                graph=graph,
                fixed=fixed,
                gpt2=real,
                graph_verdict=_verdict(graph, real),
                fixed_verdict=_verdict(fixed, real),
                trust=trust.predict(sig) if trust else None,
            )
        )
    return AuditReport(prompt, model.tokenizer.token_str(target), edit, edges)


def print_report(report: AuditReport) -> None:
    print(f"\nPrompt: {report.prompt!r}")
    print(f"GPT-2 predicts {report.prediction!r}. Edit: {report.edit} each feature.\n")
    head = f"{'layer':>5} {'word':<12} {'feature':>7} {'graph':>8} {'fixed':>8} {'GPT-2':>8}  "
    print(head + f"{'graph verdict':<15} {'fixed verdict':<15} trust")
    for e in report.edges:
        word = json.dumps(e.token)[1:-1][:12]
        trust = "  -" if e.trust is None else f"{e.trust:.0%}"
        print(
            f"{e.layer:>5} {word:<12} {e.feature:>7} {e.graph:>+8.3f} {e.fixed:>+8.3f} "
            f"{e.gpt2:>+8.3f}  {e.graph_verdict:<15} {e.fixed_verdict:<15} {trust:>5}"
        )
    fine = ("ok", "no effect")
    bad = sum(e.graph_verdict not in fine for e in report.edges)
    fixed_bad = sum(e.fixed_verdict not in fine for e in report.edges)
    n = len(report.edges)
    print(f"\nToday's graph is off on {bad} of {n} edges; with the fix, {fixed_bad} of {n}.")
    print("Effects: change in log-probability of the predicted token (nats).")


_print = print_report  # notebooks/audit_your_prompt.ipynb imports the old name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="twinlens", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="check a prompt's strongest graph edges against GPT-2")
    a.add_argument("prompt")
    a.add_argument("--top", type=int, default=8, help="number of edges to audit (default 8)")
    a.add_argument("--edit", choices=sorted(EDITS), default="ablate")
    a.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    if not 1 <= args.top <= 64:
        ap.error("--top must be between 1 and 64")
    if args.edit == "halve":
        print(
            "warning: the trust model was trained on ablate and double (x2) edits only; "
            "its scores for halve are an extrapolation",
            file=sys.stderr,
        )
    print(
        "loading GPT-2 small and its transcoders (first run downloads ~2.3 GB)...", file=sys.stderr
    )
    report = audit(args.prompt, args.top, args.edit)
    if args.json:
        json.dump(asdict(report), sys.stdout, indent=2)
        print()
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
