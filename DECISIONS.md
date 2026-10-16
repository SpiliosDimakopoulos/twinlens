# Engineering & research decisions

Each entry: the decision, the evidence behind it, and what it rules out.

## D1. Own hookable GPT-2 instead of TransformerLens at runtime (2026-10-16)
**Context.** Target machine: CPU-only laptop (Ryzen 7 5700U, 16 GB RAM), with only ~2.5 GB of
free commit memory while other applications are running.
**Evidence.**
- `import transformer_lens` alone commits **1.39 GB** (it pulls in transformers, datasets, etc.).
- `HookedTransformer.from_pretrained("gpt2")` failed with `DefaultCPUAllocator: not enough memory`
  at ~1 GB RSS, because it keeps the HF model, the converted state dict and folded copies
  alive at the same time.
**Decision.** `twinlens/core/gpt2.py` implements GPT-2 in about 250 lines of PyTorch, using
TransformerLens hook names. `twinlens/core/weights.py` memory-maps safetensors files, so the
weights are served from the OS page cache and never copied.
**Result.** Load plus a full activation cache uses **0.81 GB** of private memory. Logits match HF
`GPT2LMHeadModel` to a max abs diff of 6.9e-5. Speed matches HF (~170 ms per forward pass for
15 tokens on this CPU).

## D2. Model = GPT-2 small; twin = Dunefsky et al. per-layer transcoders (2026-10-16)
**Why GPT-2 small.** It is the most studied model in mechanistic interpretability (IOI,
induction, factual recall), so our measurements can be checked against published results.
It also fits the memory budget. Qwen3-0.6B and Gemma-3-270M transcoders exist, but they would
not fit next to the model in ~2.5 GB. They are left to the Colab notebook.
**Transcoders.** `jacobdunefsky/gpt2small-transcoders`: 12 layers, 24,576 ReLU features each,
fp32, 151 MB per layer, memory-mapped.
**Input convention, verified empirically** (`spike/tc_variant.py`, layer 8): the input is
`LayerNormPre(resid_mid)`, i.e. centred and scaled with **no** LN affine, and `b_dec` is **not**
subtracted from it. The output is `hook_mlp_out`. This gives L0 ≈ 66 and FVU 0.41. Feeding the
affine `ln2` output instead gives FVU > 1, which is wrong.
**Splice test** (`spike/tc_splice.py`, 87-token paragraph, clean loss 2.883). Loss recovered
relative to mean-ablating the MLP:

| layer | spliced loss | mean-ablated loss | recovered |
|---|---|---|---|
| 2 | 2.888 | 2.956 | 94.2% |
| 5 | 2.907 | 2.946 | 63.0% |
| 8 | 2.897 | 2.992 | 87.0% |
| 11 | 2.928 | 3.092 | 78.6% |

The twin is good but imperfect. That gap is exactly what the fidelity study measures.

## D3. Three levels of "twin", from cheapest to most faithful
1. **Linear twin** (attribution graph). Attention patterns and LN scales are frozen, MLPs are
   replaced by transcoders plus error terms, and effects propagate linearly. Cheapest, and
   what attribution-graph UIs show.
2. **Local replacement twin.** Same frozen attention and LN, but downstream transcoder features
   are recomputed with their ReLU nonlinearity.
3. **Real model** (ground truth). The intervention is applied as a feature-direction edit
   (`mlp_out -= a_f * W_dec[f]`) and everything is recomputed.

Twin fidelity = agreement between the effect predicted at level 1 or 2 and the effect measured
at level 3.
