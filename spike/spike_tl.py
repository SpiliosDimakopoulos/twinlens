"""Spike (a): GPT-2 small through TransformerLens on CPU — load time, forward latency, cache size."""

import time

import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

t0 = time.perf_counter()
model = HookedTransformer.from_pretrained("gpt2", device="cpu")
print(f"load: {time.perf_counter() - t0:.1f}s")

prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)

for _ in range(2):  # warmup
    model(tokens)
t0 = time.perf_counter()
for _ in range(10):
    model(tokens)
print(f"forward ({tokens.shape[1]} tok): {(time.perf_counter() - t0) / 10 * 1000:.1f} ms")

t0 = time.perf_counter()
logits, cache = model.run_with_cache(tokens)
print(f"run_with_cache: {(time.perf_counter() - t0) * 1000:.1f} ms")
n_bytes = sum(v.numel() * v.element_size() for v in cache.values())
print(f"cache: {len(cache)} tensors, {n_bytes / 1e6:.1f} MB")

top = logits[0, -1].topk(5)
print("top-5:", [(model.to_string(i), round(p.item(), 3)) for i, p in zip(top.indices, top.values.softmax(-1))])

t0 = time.perf_counter()
out = model.generate(prompt, max_new_tokens=20, verbose=False, do_sample=False)
print(f"generate 20 tok: {(time.perf_counter() - t0) * 1000 / 20:.1f} ms/token")
print(out)
