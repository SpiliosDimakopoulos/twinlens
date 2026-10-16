import os, sys, time, psutil, torch
sys.path.insert(0, ".")
p = psutil.Process(os.getpid())
from twinlens.core.gpt2 import GPT2
t = time.perf_counter(); m = GPT2.load(); print(f"load {time.perf_counter()-t:.2f}s private={p.memory_info().private/1e9:.2f}GB")
toks = m.to_tokens("When Mary and John went to the store, John gave a drink to")
logits, cache = m.run_with_cache(toks)
print(f"cache {len(cache)} tensors, private={p.memory_info().private/1e9:.2f}GB")
pr = logits[0, -1].softmax(-1).topk(5)
print([(m.tokenizer.token_str(i), round(v.item(), 3)) for i, v in zip(pr.indices, pr.values)])
for _ in range(2): m(toks)
t = time.perf_counter()
for _ in range(10): m(toks)
print(f"fwd {(time.perf_counter()-t)*100:.1f} ms ({toks.shape[1]} tok)")
t = time.perf_counter(); ids = [i for i, _ in m.generate("The Eiffel Tower is in the city of", 10)]
print(f"gen {(time.perf_counter()-t)*100:.1f} ms/tok |", repr(m.tokenizer.decode(ids)))
