import sys, time, torch
sys.path.insert(0, ".")
from twinlens.core.gpt2 import GPT2
from twinlens.circuits import known
from twinlens.circuits.datasets import ioi_batch, known_facts
from twinlens.causal.attribution import direct_logit_attribution, logit_lens
from twinlens.causal.patching import causal_trace, patch_heads, logit_diff_metric
m = GPT2.load()
t = time.perf_counter()
ind = known.induction_scores(m)
print("induction top8:", [(l, h, round(ind[l, h].item(), 2)) for l, h in known.top_k(ind, 8)], f"{time.perf_counter()-t:.1f}s")
b = ioi_batch(m, n=32)
print("IOI logit diff", round(known.ioi_logit_diff(m, b), 2))
print("corrupt (ABC) logit diff", round(known.ioi_logit_diff(m, type(b)(b.prompts, b.corrupt_tokens, b.io, b.s, b.corrupt_tokens, b.template)), 2))
dla = known.ioi_head_dla(m, b)
print("DLA top+:", [(l, h, round(dla[l, h].item(), 2)) for l, h in known.top_k(dla, 6)])
print("DLA top-:", [(l, h, round(dla[l, h].item(), 2)) for l, h in known.top_k(dla, 4, largest=False)])
labels, contrib = direct_logit_attribution(m, b.tokens, m.W_E[b.io] - m.W_E[b.s])
bias = ((m.W_E[b.io] - m.W_E[b.s]) @ m.ln_f[1]).mean().item()
print("DLA sum+bias", round(contrib.sum().item() + bias, 3))
print("ablate name movers:", round(known.ioi_ablate_heads(m, b, known.NAME_MOVER_HEADS), 2))
print("ablate 3 random heads:", round(known.ioi_ablate_heads(m, b, {(1, 3), (2, 7), (4, 4)}), 2))
t = time.perf_counter()
ph = patch_heads(m, b.tokens[:8], b.corrupt_tokens[:8], logit_diff_metric(b.io[:8], b.s[:8]))
print("head patching top:", [(l, h, round(ph[l, h].item(), 2)) for l, h in known.top_k(ph, 6)], f"{time.perf_counter()-t:.1f}s")
facts = known_facts(m)
print("known facts:", len(facts), [(f.prompt[:25], round(f.p_target, 2), f.subject_positions) for f in facts])
t = time.perf_counter()
f = facts[0]
for kind in ["resid", "mlp", "attn"]:
    r = causal_trace(m, f.tokens, f.subject_positions, f.target, kind=kind, window=3, n_samples=4)
    e = r["effect"]; idx = int(e.argmax()); print(kind, "p_clean", round(r["p_clean"], 3), "p_corr", round(r["p_corrupt"], 3), "argmax (layer,pos)", (idx // e.shape[1], idx % e.shape[1]), "max", round(e.max().item(), 3))
print(f"trace {time.perf_counter()-t:.1f}s")
