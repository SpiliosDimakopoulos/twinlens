import sys, torch
sys.path.insert(0, ".")
from twinlens.core.gpt2 import GPT2
from twinlens.circuits.datasets import known_facts
from twinlens.causal.patching import causal_trace
m = GPT2.load()
facts = known_facts(m)
print(len(facts), "facts")
agg = {k: {"last_subj": torch.zeros(12), "first_subj": torch.zeros(12), "last": torch.zeros(12), "further": torch.zeros(12)} for k in ["mlp", "attn", "resid"]}
for f in facts:
    toks = m.tokenizer.str_tokens(f.tokens[0].tolist())
    print(f.prompt, [toks[i] for i in f.subject_positions])
    for kind in agg:
        r = causal_trace(m, f.tokens, f.subject_positions, f.target, kind=kind, window=3, n_samples=4)
        e = (r["effect"] - r["p_corrupt"]) / (r["p_clean"] - r["p_corrupt"])  # normalised indirect effect
        ls = f.subject_positions[-1]
        agg[kind]["last_subj"] += e[:, ls] / len(facts)
        agg[kind]["first_subj"] += e[:, f.subject_positions[0]] / len(facts)
        agg[kind]["last"] += e[:, -1] / len(facts)
        agg[kind]["further"] += e[:, ls + 1:-1].mean(1) / len(facts)
for kind, d in agg.items():
    print(f"== {kind}")
    for name, v in d.items(): print(f"  {name:10s}", " ".join(f"{x:5.2f}" for x in v.tolist()))
