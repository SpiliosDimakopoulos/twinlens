import sys, time, torch
sys.path.insert(0, ".")
from twinlens.core.gpt2 import GPT2
from twinlens.circuits import known
from twinlens.circuits.datasets import ioi_batch, known_facts
from twinlens.causal.patching import causal_trace, patch_heads, patch_resid, logit_diff_metric
m = GPT2.load(); b = ioi_batch(m, n=8)
t = time.perf_counter(); ph = patch_heads(m, b.tokens, b.corrupt_tokens, logit_diff_metric(b.io, b.s))
print("head patching top:", [(l, h, round(ph[l, h].item(), 2)) for l, h in known.top_k(ph, 5)], f"{time.perf_counter()-t:.1f}s")
t = time.perf_counter(); pr = patch_resid(m, b.tokens, b.corrupt_tokens, logit_diff_metric(b.io, b.s))
print("resid patching: last-pos row by layer", [round(x, 2) for x in pr[:, -1].tolist()], f"{time.perf_counter()-t:.1f}s")
print(m.tokenizer.str_tokens(b.tokens[0].tolist()))
print("resid patching layer 0..3 argmax pos:", [int(pr[l].argmax()) for l in range(12)])
facts = known_facts(m)
t = time.perf_counter()
for f in facts:
    toks = m.tokenizer.str_tokens(f.tokens[0].tolist())
    res = {}
    for kind in ["mlp", "attn"]:
        e = causal_trace(m, f.tokens, f.subject_positions, f.target, kind=kind, window=3, n_samples=4)["effect"]
        i = int(e.argmax()); res[kind] = (i // e.shape[1], toks[i % e.shape[1]], round(e.max().item(), 2))
    print(f.prompt[:30], "| last subj:", toks[f.subject_positions[-1]], "| mlp", res["mlp"], "| attn", res["attn"])
print(f"traces {time.perf_counter()-t:.1f}s")
