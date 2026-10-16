import sys, torch
sys.path.insert(0, ".")
from twinlens.core.gpt2 import GPT2
from twinlens.core.weights import hf_file, MmapSafetensors
m = GPT2.load()
text = ("The history of the Roman Empire spans many centuries. It began when Augustus became the first emperor in 27 BC, "
        "and the western half collapsed in 476 AD after a long period of decline. Historians still debate the causes, "
        "which include economic troubles, military defeats, and political instability. The eastern half, known as the "
        "Byzantine Empire, survived until 1453 when Constantinople fell to the Ottoman Turks.")
toks = m.to_tokens(text)
def loss(logits): return torch.nn.functional.cross_entropy(logits[0, :-1], toks[0, 1:]).item()
base = loss(m(toks))
mean_abl = None
print(f"clean loss {base:.3f}")
for L in [2, 5, 8, 11]:
    tc = MmapSafetensors(hf_file("jacobdunefsky/gpt2small-transcoders", f"gpt2-small-dun-chl-mlp-tc{L}/sae.safetensors"))
    We, be, Wd, bd = (tc.get(k) for k in ["W_enc", "b_enc", "W_dec", "b_dec"])
    store = {}
    def grab(x, n): store["x"] = x
    def lnpre(r):
        r = r - r.mean(-1, keepdim=True); return r / (r.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
    def splice(y, n):
        return torch.relu(lnpre(store["r"]) @ We + be) @ Wd + bd
    def grab_r(r, n): store["r"] = r
    def zero(y, n): return torch.zeros_like(y)
    def meanab(y, n): return y.mean(1, keepdim=True).expand_as(y)
    hooks_tc = [(f"blocks.{L}.hook_resid_mid", grab_r), (f"blocks.{L}.hook_mlp_out", splice)]
    with m.hooks(hooks_tc): l_tc = loss(m(toks))
    with m.hooks([(f"blocks.{L}.hook_mlp_out", meanab)]): l_ab = loss(m(toks))
    print(f"L{L:2d}: transcoder-spliced {l_tc:.3f} | mean-ablated {l_ab:.3f} | loss recovered {(l_ab-l_tc)/(l_ab-base)*100:5.1f}%")
