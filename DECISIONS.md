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

**Settled (2026-11-12, `out/convention.log`, real weights, 86 tokens):** input
`ln2.hook_normalized` (without the `1 + weight`), output `hook_mlp_out`, as circuit-tracer's
config suggests. It has the lowest mean FVU (0.45 vs 0.93 for `hook_mlp_in`; both `mlp.hook_out`
variants are > 1000) and L0 matches the published values (47.7/49, 63.4/52, 8.4/8, 14.4/11 at
layers 2/8/14/20); with `hook_mlp_in`, L0 is 2-3x too high at layers 2 and 8. Those are now the
defaults. Caveat: single-layer splice recovery is uneven (1.00 at layer 8, 0.34-0.45 at 14 and 20,
-0.50 at layer 2), so the Gemma twin starts from a weaker replacement than GPT-2's.

## D5. Cross-layer transcoder format, Gemma-2-2B 426K (2026-11-13)

Read from `decoderesearch/circuit-tracer` (commit `8c35dc2`, 2026-10-17:
`transcoder/cross_layer_transcoder.py`, `replacement_model/replacement_model_transformerlens.py`,
`utils/tl_nnsight_mapping.py`, `graph.py`) and from the safetensors headers of
`mntss/clt-gemma-2-2b-426k` (HTTP range reads, nothing downloaded). Several of these differ from
what the extension plan assumed; the facts below are the ones to build on.

- **Repo**: ungated. 81 files, 33.1 GB: `W_enc_{i}.safetensors` (i = 0..25, 75.5 MB each),
  `W_dec_{i}.safetensors` (1.96 GB for i = 0 down to 75.5 MB for i = 25, 26.5 GB in all),
  `features/layer_{i}.bin` (≈4.6 GB, feature visualisations, not needed) and `config.yaml`
  (`model_name: google/gemma-2-2b`, `model_kind: cross_layer_transcoder`,
  `feature_input_hook: hook_resid_mid`, `feature_output_hook: hook_mlp_out`).
- **Tensors, all BF16**: `W_enc_i` [16384, 2304] (feature-major, the transpose of the Gemma Scope
  PLT layout), `b_enc_i` [16384], `b_dec_i` [2304] (stored in the *encoder* file), `W_dec_i`
  [16384, 26 − i, 2304]: feature f of layer i has one output vector for every layer l' = i..25,
  at index l' − i. 26 layers × 16384 = 425,984 features. No `threshold_i` and no `W_skip_i`.
- **Activation: ReLU, not JumpReLU.** `load_clt` picks JumpReLU only when `threshold_i` exists,
  and it does not here. So `acts = relu(resid_mid @ W_enc_iᵀ + b_enc_i)`.
- **Input: the raw residual `hook_resid_mid`** (the input of `pre_feedforward_layernorm`), with no
  norm in front of the encoder. The Gemma Scope PLTs read `ln2.hook_normalized` instead (D4).
  Our `twinlens/core/gemma2.py` already exposes `hook_resid_mid`.
- **Output: `hook_mlp_out`**, after `post_feedforward_layernorm`, as for the PLTs. Reconstruction:
  `mlp_out[l'] ≈ Σ_{i ≤ l'} Σ_f acts_i[f] · W_dec_i[f, l' − i] + b_dec_{l'}`. The plan's
  assumed shape holds. Since `ln2` and `ln2_post` sit wholly inside the replaced MLP, the only
  norms left on the twin's path are `ln1`, `ln1_post` and the final norm.
- **Size**: 28.5 GB to download (encoders + decoders, BF16); 57 GB if converted to F32. Decision:
  keep BF16 on disk, memory-map, convert only the rows read (active features) to F32. The full
  encoder (1.96 GB BF16) is read per layer for the clean trace.
- **BOS**: attribution zeroes all feature activations at position 0 (`zero_positions =
  slice(0, 1)`); Gemma prepends BOS, so the twin does the same for the CLT.
- **Intervention in the real model** (`feature_intervention`, `constrained_layers=None`, i.e.
  iterative patching): at layer i's `hook_mlp_out` it reads the feature's *current* activation
  (recomputed from the live residual), sets `Δ = value − current`, and adds
  `Δ · W_dec_i[f, l' − i]` to `hook_mlp_out[l']` for every l' ≥ i. All MLPs stay the real ones;
  later features are not clamped. This matches the plan's definition of `real`. One difference:
  the **default is `freeze_attention=True`**, which freezes `hook_pattern` (attention patterns)
  at their clean values; norms and MLPs still run live. With `constrained_layers` covering all
  layers it also freezes norm scales and the MLP outputs, which gives direct effects.
  Decision: `real` stays fully live, as for every existing number in the report, and the CLT
  sweep adds a second ground truth `real_frozen_attn` that reproduces circuit-tracer's default,
  so both comparisons can be reported. Fixed now, before any sweep.
- **Graph conventions** (needed for the parity check, plan 0.6): logit nodes use the *demeaned*
  unembedding column (`W_U[:, t] − mean_t W_U`), and edges are checked with the final softcap
  switched off (`zero_softcap`), so feature→logit edges are pre-softcap and demeaned. Pruning
  defaults: `node_threshold = 0.8`, `edge_threshold = 0.98`; targets: `max_n_logits = 10`,
  `desired_logit_prob = 0.95`; `max_feature_nodes = None`. circuit-tracer itself tests this CLT
  (`tests/test_attributions_gemma.py::test_gemma_2_2b_clt`, CUDA only).

## D6. One twin for per-layer and cross-layer transcoders (2026-11-13)

- **One code path.** A feature of source layer `i` has one decoder vector per output layer
  (`trace.dec[i]` is `[k, n_out, d_model]`; a PLT has `n_out = 1`, a CLT writes to every
  `l >= i`). The twin's output at layer `l` is the sum over the sources writing there, plus
  `b_dec` of layer `l` and its error term. With one source the sum is that single term, so every
  PLT number is unchanged: checked bit for bit against the previous code on the tiny Gemma for
  all levels, `real`, both transcoder hook choices, batched multi-layer edits, `feature_effects`
  and the reliability signals.
- **Sources upstream of the first edit** read an unperturbed input, so their clean contribution
  is used; each later source's activations are computed once per run at its own layer and reused
  by every output layer it writes to (`tests/test_clt_tiny.py` checks the two agree).
- **Edits** set a feature relative to its clean activation and add `Δ · W_dec[f → l]` at every
  output layer `l`, in the twin and in `real` alike. For a single edit this is circuit-tracer's
  iterative patching: the feature's own input is upstream of the edit, so clean = current.
- **`real_frozen_attn`**: `real` with attention patterns frozen at the clean run (D5).
- **Reliability signals and labels for a CLT**: `rel_edit_size` uses the norm of all of a
  feature's output vectors together; `feature_label` reads the sum of them. Both are the old
  values for a PLT.
- **BOS**: the CLT set sets `skip_first`, so position-0 activations are zero and the whole
  position-0 output sits in the error term, like Qwen's attention sink and like circuit-tracer.

## D7. CLT pilot: analysis and decision gate, fixed before the sweep (2026-11-13)

Written before any CLT sweep number was seen (the `--small` run is for timing only and is not
analysed for the decision). Script: `experiments/analyze_clt.py`, section `gemma2_clt` of
`summary.json`, seed 0, 1000 cluster-bootstrap resamples over prompts.

- **Sweep**: all 56 prompts, `--n-top 3 --n-random 3` (edit selection as for the PLT sweep:
  `sample_edits`, ablate and ×2), levels `direct`, `linear`, `linear_lnjac`, `linear_fulljac`,
  plus both ground truths `real` and `real_frozen_attn`. The nonlinear `replacement*` levels are
  left out: they need a full 426K-feature encode of every layer per run. AtP on the same edits
  (`atp_baseline.py --transcoders clt426k`), summed over every output layer of the feature.
- **Primary ground truth `real`** (everything live), as for every existing number; every block is
  repeated against `real_frozen_attn`, and the agreement of the two ground truths is reported.
- **Metrics**: Spearman ρ with the real Δlog p, sign agreement on edits with |real| > 0.05 nats,
  share within tolerance, share with error > 1 nat ("catastrophic"), clipped mean error; paired
  differences Fix B − linear and Fix B − AtP; Spearman per edited layer; the PLT twin's metrics on
  the same prompts.
- **Gate** (plan 0.9). Let `[lo, hi]` be the interval of Δρ = ρ(Fix B) − ρ(linear) on the CLT, and
  `g = ρ(CLT linear) − ρ(PLT Fix B)` with a joint bootstrap over prompts (`[g_lo, g_hi]`):
  - **A** (finding holds on CLTs): `lo > 0` and `g_hi < 0`;
  - **B** (CLTs fix it themselves): `lo ≤ 0` and `g_hi ≥ 0`;
  - **C** (mixed): anything else.
  The script prints the outcome; it is not revisited after seeing the numbers.
