import sys, torch
sys.path.insert(0, ".")
from twinlens.core.gpt2 import GPT2
from twinlens.twin.replacement import Twin, FeatureEdit
twin = Twin(GPT2.load()); m = twin.model
tr = twin.trace(m.to_tokens("The Colosseum is located in the city of"))
pos = 3
f = int(tr.acts[0][pos].argmax()); a = tr.acts[0][pos, f].item()
print("layer0 feature", f, "act", round(a, 2), "dec norm", twin.tcs[0].W_dec[f].norm().item(), "resid norm", tr.resid_mid[0][0, pos].norm().item())
print("err frac per layer at pos:", [round((tr.error[l][0,pos].norm()/tr.mlp_out[l][0,pos].norm()).item(), 2) for l in range(12)])
e = FeatureEdit(0, pos, f, 0.0)
for mode in ["linear", "replacement", "real"]:
    store = {}
    names = [f"blocks.{l}.hook_resid_post" for l in range(12)]
    hooks = [(n, lambda x, nm: store.__setitem__(nm, x[0, :].clone())) for n in names]
    with m.hooks(hooks):
        twin.run(tr, mode, [e])
    _, clean = m.run_with_cache(tr.tokens, names)
    print(mode.ljust(12), "||Δresid_post|| at last pos per layer:", [f"{(store[n][-1]-clean[n][0,-1]).norm().item():.2g}" for n in names])
