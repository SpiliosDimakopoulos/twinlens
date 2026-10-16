import sys, time, torch
sys.path.insert(0, ".")
torch.set_grad_enabled(False)
from twinlens.core.gpt2 import GPT2
m = GPT2.load()
from transformers import GPT2LMHeadModel
hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", cache_dir=".hf_cache/hub").eval()
toks = m.to_tokens(["The Eiffel Tower is in the city of", "When Mary and John went to the store, the"])
a = m(toks); b = hf(toks).logits
print("max abs diff", (a - b).abs().max().item())
for name, f in [("ours", lambda: m(toks)), ("hf", lambda: hf(toks))]:
    f(); t = time.perf_counter()
    for _ in range(10): f()
    print(name, f"{(time.perf_counter()-t)*100:.1f} ms")
