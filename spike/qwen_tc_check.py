import sys, time, torch
sys.path.insert(0, ".")
from twinlens.core.qwen3 import Qwen3
from twinlens.twin.transcoders import RemoteRowTranscoder
m = Qwen3.load()
L = 16
t = time.time(); tc = RemoteRowTranscoder(L); tc.prefetch_decoder(); print(f"tc ready {time.time()-t:.0f}s d_sae={tc.d_sae}", flush=True)
text = ("The history of the Roman Empire spans many centuries. It began when Augustus became the first emperor, "
        "and the western half collapsed in 476 AD after a long decline. Historians still debate the causes.")
toks = m.to_tokens(text)
b = f"blocks.{L}."
logits, c = m.run_with_cache(toks, [b + "ln2.hook_normalized", b + "hook_mlp_in", b + "hook_mlp_out"])
Y = c[b + "hook_mlp_out"][0, 1:]
def fvu(p, y): return ((p - y).pow(2).sum() / (y - y.mean(0)).pow(2).sum()).item()
best = None
for name, X in [("normalized (no weight)", c[b + "ln2.hook_normalized"][0, 1:]), ("normalized x weight", c[b + "hook_mlp_in"][0, 1:])]:
    acts = torch.relu(tc.pre(X))
    idx = acts.any(0).nonzero()[:, 0]
    pred = acts[:, idx] @ tc.dec_rows(idx) + tc.b_dec
    f = fvu(pred, Y); l0 = (acts > 0).float().sum(-1).mean().item()
    print(f"{name:24s} L0={l0:6.1f} active={len(idx)} FVU={f:.3f}", flush=True)
