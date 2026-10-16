import sys, torch, glob
sys.path.insert(0, ".")
from twinlens.core.gpt2 import GPT2
from twinlens.core.weights import hf_file, MmapSafetensors
m = GPT2.load()
L = 8
tc = MmapSafetensors(hf_file("jacobdunefsky/gpt2small-transcoders", f"gpt2-small-dun-chl-mlp-tc{L}/sae.safetensors"))
W_enc, b_enc, W_dec, b_dec = (tc.get(k) for k in ["W_enc", "b_enc", "W_dec", "b_dec"])
texts = ["The quick brown fox jumps over the lazy dog because it was very tired after a long day of running around.",
         "In 1905 Albert Einstein published four papers that changed physics forever, including special relativity theory.",
         "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(10))"]
xs_pre, xs_aff, ys = [], [], []
for t in texts:
    _, c = m.run_with_cache(m.to_tokens(t), [f"blocks.{L}.hook_resid_mid", f"blocks.{L}.hook_mlp_in", f"blocks.{L}.hook_mlp_out"])
    r = c[f"blocks.{L}.hook_resid_mid"][0]
    r = r - r.mean(-1, keepdim=True); r = r / (r.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
    xs_pre.append(r[1:]); xs_aff.append(c[f"blocks.{L}.hook_mlp_in"][0, 1:]); ys.append(c[f"blocks.{L}.hook_mlp_out"][0, 1:])
X_pre, X_aff, Y = torch.cat(xs_pre), torch.cat(xs_aff), torch.cat(ys)
W_out = m.blocks[L]["W_out"]
# centered target = mlp_out with W_out & b_out mean-centered across d_model
Yc = Y - Y.mean(-1, keepdim=True)
def fvu(pred, y): return ((pred - y).pow(2).sum() / (y - y.mean(0)).pow(2).sum()).item()
for xn, X in [("LNPre-normalized", X_pre), ("ln2 affine", X_aff)]:
    for sub in [False, True]:
        h = torch.relu((X - (b_dec if sub else 0)) @ W_enc + b_enc)
        pred = h @ W_dec + b_dec
        print(f"{xn:18s} sub_bdec={sub!s:5s} L0={(h>0).float().sum(-1).mean():6.1f}  FVU raw={fvu(pred, Y):.3f} centered={fvu(pred, Yc):.3f}")
