# Engineering & research decisions

Each entry: the decision, the evidence behind it, and what it rules out.

## D1. Own hookable GPT-2 instead of TransformerLens at runtime (2026-10-16)
**Context.** Target machine: CPU-only laptop (Ryzen 7 5700U, 16 GB RAM), with only ~2.5 GB of
free commit memory while other applications are running.
**Evidence.**
- After `import torch, transformer_lens` the process has **1.39 GB** private memory (0.69 GB of it
  is torch); TransformerLens pulls in transformers, datasets, etc.
- `HookedTransformer.from_pretrained("gpt2")` failed with `DefaultCPUAllocator: not enough memory`
  at ~1 GB RSS, because it keeps the HF model, the converted state dict and folded copies
  alive at the same time.
**Decision.** `twinlens/core/gpt2.py` implements GPT-2 in about 250 lines of PyTorch, using
TransformerLens hook names. `twinlens/core/weights.py` memory-maps safetensors files, so the
weights are served from the OS page cache and never copied.
**Result.** Load plus a full activation cache uses **0.81 GB** of private memory (0.82 GB in the later benchmark, `bench/latency.json`). Logits match HF
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

## D3. Three levels of "twin", from cheapest to most faithful (first design)

*Superseded: the study uses five twin levels plus two linear fixes; see §3 of the report. Kept
as the record of the original design.*

1. **Linear twin** (attribution graph). Attention patterns and LN scales are frozen, MLPs are
   replaced by transcoders plus error terms, and effects propagate linearly. Cheapest, and
   what attribution-graph UIs show.
2. **Local replacement twin.** Same frozen attention and LN, but downstream transcoder features
   are recomputed with their ReLU nonlinearity.
3. **Real model** (ground truth). The intervention is applied as a feature-direction edit
   (`mlp_out -= a_f * W_dec[f]`) and everything is recomputed.

Twin fidelity = agreement between the effect predicted at level 1 or 2 and the effect measured
at level 3.

## D4. Gemma-2-2B port (2026-11-11)

- **Weights**: `unsloth/gemma-2-2b`, an ungated mirror of `google/gemma-2-2b` (same single BF16
  `model.safetensors`), converted once to a memory-mapped F32 copy like Qwen3.
  `twinlens/core/gemma2.py` follows HF `modeling_gemma2.py` (transformers 4.57): embeddings
  × sqrt(d_model), four RMSNorms per block with a `1 + weight` factor, GQA, RoPE, alternating
  sliding-window layers (implemented, not just asserted), GeGLU (tanh GELU), tied unembedding.
  On a tiny random Gemma2 it matches HF to < 1e-5 (`tests/test_gemma_tiny.py`).
- **Transcoders**: Gemma Scope 16k-feature JumpReLU transcoders
  (`google/gemma-scope-2b-pt-transcoders`), with the per-layer L0 choice circuit-tracer uses
  (`mntss/gemma-scope-transcoders`). Each `params.npz` (stored F32 arrays `W_enc` [2304, 16384],
  `W_dec` [16384, 2304], `b_enc`, `b_dec`, `threshold`; checked from the zip directory) is
  streamed into an F32 safetensors file and memory-mapped. Features are `pre * (pre > threshold)`
  everywhere: clean trace, replacement levels, and hence the "active" set. The linear levels
  keep each feature's clean on/off gate, as with ReLU.
- **Norms**: every norm on the twin's path (`ln1`, `ln1_post`, `ln2`, and `ln2_post` if the
  transcoder writes before it) is frozen, linearised or live per level, through the same
  `hook_scale`/`hook_normalized` hooks as GPT-2 and Qwen3. Norms the transcoder replaces are
  computed and then overwritten, so their treatment does not matter.
- **Attention softcap** `50 tanh(s / 50)`: frozen levels freeze the pattern; live levels run the
  real cap; `linear_fulljac` multiplies the score perturbation by `1 - tanh²(s0 / 50)` before the
  softmax Jacobian. On the tiny model, leaving that factor out raises the Jacobian error vs.
  the live reference from 0.1% to 1.1%.
- **Final logit softcap** `30 tanh(l / 30)` is part of the read-out, applied identically at every
  level (like the metrics' log-softmax). The linear levels are affine before it; the tests check
  that with the cap switched off (`model.final_softcap = None`).
- **BOS**: Gemma prepends BOS (id 2); `skip_first` is False, since the sweep never edits BOS.

**Open question: which hooks the transcoders read and write.** The input is either the
pre-feedforward norm output with its `1 + weight` (`hook_mlp_in`) or without it
(`ln2.hook_normalized`). The output is the MLP output either after the post-feedforward norm
(`hook_mlp_out`) or before it (`mlp.hook_out`). Both are parameters of
`GemmaScopeTranscoderSet` (`in_hook`, `out_hook`). The defaults are `hook_mlp_in` and
`hook_mlp_out`. Evidence on circuit-tracer's side: its config names `ln2.hook_normalized` and
`hook_mlp_out`. With TransformerLens loaded `fold_ln=False` (as circuit-tracer does),
`hook_normalized` is taken *before* the weight, and its nnsight backend maps it to the inner
`_norm` call. That points to the input **without** the weight, and to the output after the
post-norm. `experiments/gemma_convention.py` settles it on real weights: FVU and single-layer
splice loss recovery for all four combinations at layers 2, 8, 14 and 20. Pick the combination
with low FVU, recovery near 1 and L0 near the published value, and change the defaults to match.
