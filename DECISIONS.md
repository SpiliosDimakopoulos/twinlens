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

## D8. Feature-to-feature edges: definition and analysis, fixed before code (2026-11-15)

Plan phase 1. So far only feature → output effects were tested; a graph also asserts
"feature A moves feature B by w". This fixes what that claim means and how it is scored before
any edge number exists.

- **Edits**: exactly the edits of the output sweeps (`sample_edits`, seed = prompt id; GPT-2
  `--n-top 2 --n-random 2`, Gemma PLT and CLT `--n-top 3 --n-random 3`), so every edge row
  joins the output row of its edit. Edits in the last layer have no targets and are skipped.
- **Targets** of an edit at (l, p, f): features (l′, p′, f′) with l′ > l, p′ ≥ p, p′ > 0 (no
  BOS) and clean activation > 0. An edit changes `mlp_out[l]`, which no transcoder at layer ≤ l
  reads, hence l′ > l. Per edit, K = 10 targets with the largest |graph prediction| (the
  `linear` level below) and K = 10 drawn uniformly from the other targets (seeded), so that the
  test is not limited to what the graph itself picked (`selection` = top / random).
- **Quantity**: the target's pre-activation `pre = x · W_enc[f′] + b_enc[f′]`, where x is the
  transcoder input at (l′, p′) in the run. Clean value `pre0`, clean activation `act0 > 0`.
  - *Prediction of a twin level* (`linear`, `linear_lnjac`, `linear_fulljac`): the change of
    `pre` inside that level's run. The gates are frozen and every target is clean-active, so
    this is also the predicted change of the activation. For `linear` it is exactly the
    attribution graph's edge weight times Δa (the linear twin is the graph).
  - *Truth*: `pre` in the real model with the edit (`real`, everything live; and
    `real_frozen_attn`, circuit-tracer's default intervention), passed through the
    transcoder's own nonlinearity (ReLU or JumpReLU) to give `Δact_real = act_real − act0`.
    `Δpre_real` is kept as well, so that the error splits into a linearisation part
    (prediction − Δpre_real) and a gate part (Δpre_real − Δact_real, non-zero only when the
    target turns off).
- **Metrics** (script `experiments/analyze_edges.py`, block `edges_<model>` in `summary.json`,
  seed 0, 1000 cluster-bootstrap resamples over prompts):
  - primary: Spearman ρ between predicted and real Δact over all (edit, target) pairs, per
    level, with the paired difference Fix B − linear;
  - ρ of the relative change Δact / act0 (scale-free across features);
  - sign agreement on targets that really move (|Δact_real| > 0.1 · act0);
  - share of targets that turn off in the real model (gate errors), and the share of the
    squared error that is the gate part;
  - all of the above by selection (top / random) and by distance l′ − l.
- **Reading** (not a gate, the phase continues either way): the graph's internal edges are
  "trustworthy" to the degree their ρ matches the output ρ of the same level; Fix B "helps
  inside the graph" if the interval of its Δρ over linear is above zero.

## D9. External prompts and a contrast target, fixed before the sweep (2026-11-15)

Plan tasks 2.1 and 2.2. All numbers so far come from 56 prompts we wrote ourselves; the
reviewer's first question is whether they hold on prompts we did not choose. This fixes the
sets, the filter and the analysis before any external number exists.

- **Sets** (script `experiments/build_external_prompts.py`, every source pinned to a revision):
  - `ioi`: `mib-bench/ioi` test split (MIT), the MIB release of the Wang et al. (2023)
    templates. Answer = indirect object, contrast = subject.
  - `sva`: `nyu-mll/blimp` (CC BY 4.0), six subject-verb agreement paradigms
    (regular/irregular plural 1-2, distractor relational noun, distractor relative clause),
    pairs with `one_prefix_method`, cut before the single differing word. Answer = first token
    of the grammatical verb, contrast = first token of the ungrammatical one.
  - `factual`: `evandez/relations` (MIT), 13 LRE factual relations (Hernandez et al., 2024),
    first prompt template. Answer = first token of the object, contrast = first token of
    another object of the same relation (seeded).
  - `greater_than` (GPT-2 only): generated from the template of Hanna et al. (2023), seven
    nouns × YY ∈ 02..98; answer = the model's top-1 when it is a two-digit number above YY,
    contrast = YY − 1. Gemma's tokenizer splits numbers into digits, so the family does not
    exist there.
  - CounterFact was left out: its data licence could not be confirmed.
- **Filter**: answer and contrast are single (first) tokens and differ, and the model's top-1
  next token is the answer. Candidates are shuffled with seed 0 per family; the first N that
  pass are kept, separately per model (the sets differ between GPT-2 and Gemma). N = 100 per
  family on GPT-2 (≤ 400 prompts) and N = 60 on Gemma (≤ 180), set by CPU cost (Gemma PLT and
  CLT run ~0.6 s per edit, ~310 edits per prompt). If a family yields fewer, it is reported
  with what it has; the filter is not loosened afterwards.
- **Runs**: the output sweep unchanged (`fidelity_sweep.py --prompts external`, same edit
  selection, levels and seeds as §4.1, §4.9, §4.10) on GPT-2, Gemma PLT and Gemma CLT. The
  only addition is the second target `<level>_dld`: the change of logit(answer) −
  logit(contrast), next to the existing Δ log p of the top-1 (= answer). Existing columns and
  every number of the development set are unchanged.
- **Analysis** (`experiments/analyze_external.py`, block `external_<model>` in
  `summary.json`, seed 0, 1000 cluster bootstraps over prompts): the metrics of
  `analyze_clt.py` per level (ρ, sign agreement and within tolerance on big edits,
  catastrophic misses, mean error) and the paired Fix B − linear differences; for the second
  target, Spearman ρ and sign agreement (on edits with |real Δ| > 0.1 nat) of the predicted
  versus the real `dld`. Also per family.
- **Reading**: the 56 prompts are "development", the external sets are "held-out". The
  development claim "Fix B improves ρ over the graph" replicates on a model if the interval of
  the paired Δρ on the external set is above zero. A failed replication is reported with the
  same weight. The trust model is not refitted here (task 2.6).

## D10-0. Re-analysis of the existing sweeps, fixed before any number (2026-11-24)

Plan v2, phase 0. No new runs: only `fidelity*.parquet`, `fidelity_atp*.parquet` and `edges_*.parquet`
as committed. Script `experiments/analyze_reanalysis.py`, block `reanalysis` in `summary.json`, seed 0,
1000 cluster-bootstrap resamples over prompts. Models: `gpt2`, `qwen3`, `gemma2` (PLT), `gemma2_clt`.
Primary ground truth `real` everywhere; `real_frozen_attn` (circuit-tracer's default intervention:
attention patterns frozen, normalisation live) where it was recorded.

- **0.1 Disagreement as a flag.** On big edits (|real| > 0.05 nats), the label is "the graph
  (`linear`) is outside tolerance". Scores: `dis_full` = |linear − linear_fulljac|, `dis_ln` =
  |linear − linear_lnjac|, and the baseline `size` = |linear| (the graph's own prediction, AUROC 0.87
  on GPT-2, §4.6). AUROC of each, and the paired AUROC(dis_full) − AUROC(size).
  Reading: unfreezing has a use as a *diagnostic* on a model if AUROC(dis_full) ≥ 0.80 **and** the
  paired interval over `size` lies above zero. The decisive model is `gemma2_clt`.
- **0.2 Normalisation alone against circuit-tracer's intervention.** Paired Δρ of `linear_lnjac` −
  `linear`, `linear_fulljac` − `linear` and `linear_lnjac` − `linear_fulljac`, against
  `real_frozen_attn`: at the output on `gemma2_clt` (the only output sweep that recorded it), and on
  the edges of all three edge sweeps. Reading: a `freeze_layernorm`-only change is supported if the
  interval of `linear_lnjac` − `linear` lies above zero at the output and on the edges of `gemma2_clt`.
- **0.3 Where something happens.** Per model and level (`linear`, `linear_lnjac`, `linear_fulljac`,
  `atp` where present, and `zero` = predict no change): ρ, sign agreement, within tolerance and clipped
  mean error on big edits, split by edit selection (top / random). And **top-10 recall**: per prompt,
  on ablations only, the share of the 10 edits with the largest |prediction| that are among the 10 with
  the largest |real| effect, averaged over prompts; chance = 10 / (ablations in the prompt).
- **0.4 Depth, CLT against PLT** (Lange et al., 2026, argue that CLTs rewrite deep circuits as shallow
  ones). On Gemma-2-2B, same prompts: ρ of `linear` and `linear_fulljac` per edited-layer bin
  (0–5, 6–12, 13–19, 20–25), with the joint-bootstrap CLT − PLT difference per bin. Descriptive.
- **0.5 Multiple comparisons.** Bootstrap two-sided p-value p = min(1, 2·min(P*(Δ ≤ 0), P*(Δ ≥ 0)))
  (resolution 0.002) for every paired difference the report states as excluding zero: Fix B − graph
  (ρ, sign agreement, within tolerance on big edits, catastrophic) on the four models, and Fix B −
  attribution patching (same four metrics) on `gpt2`, `gemma2`, `gemma2_clt`: 28 tests, Holm at 0.05.
  A claim that does not survive is reworded in the report.
  *Note, 2026-11-24, after the run:* "states as excluding zero" was loose. Not all 28 had been
  stated so (e.g. Fix B − AtP within tolerance on GPT-2 was reported as indistinguishable). The
  family was run exactly as enumerated; only the rewording rule applies to the ones that had.
- Nothing here changes D10 (circuits, dose, AtP 2×2), which is fixed separately before its runs.

## D10. Pruned circuits, dose and frozen-gradient AtP, fixed before any number (2026-11-24)

Plan v2, phases 1–2. This answers the question left open on circuit-tracer PR #123: are circuits
pruned from graphs that keep the LayerNorm and attention Jacobians more faithful? Every graph and
intervention below goes through circuit-tracer itself. The branch is the PR's
`gradient-flow-options`, rebased onto upstream `82996f1`, with the commit recorded in the results.
The TwinLens engine is used only for the cross-check and for D10-C.

**Shared setup**
- Models: Gemma-2-2B with the 426K CLT (`mntss/clt-gemma-2-2b-426k`), which is primary, and with
  the Gemma Scope PLTs (circuit-tracer's `gemma` set).
- Prompts: the first 20 per family of the Gemma external sets of D9 (`ioi`, `sva`, `factual`),
  60 in total.
- Target: `attribution_targets=[answer token]`, so each graph attributes the quantity that is
  measured.
- Metric m: the demeaned answer logit (circuit-tracer's logit target). The secondary metric is
  logit(answer) − logit(contrast).
- Three graph variants per prompt, with otherwise identical arguments:
  - G1: today's graph (both frozen);
  - G2: `freeze_layernorm=False`;
  - G3: both `False`.
- Node budget: `max_feature_nodes=None` if the pilot shows that a G3 graph takes ≤ 30 min on the
  pilot server. Otherwise the pilot sets one value M ≥ 2000 for every variant.
- Candidate set C: the union of the feature nodes of G1–G3. A feature missing from a variant's
  graph gets influence 0 in that variant (this matters only when M is set).
- Ground truths, both in circuit-tracer's `feature_intervention` with no `constrained_layers`:
  - T1 (primary): `freeze_attention=False`;
  - T2: the default `freeze_attention=True`.
  Following the code, both keep LayerNorm live, and transcoders are recomputed downstream.
- Bootstrap: 95% intervals over prompts, seed 0, 1000 resamples.

**D10-A, circuits**
- Rankings of C:
  - R1, R2, R3: node influence (`compute_node_influence`, as `prune_graph` uses it) in G1, G2, G3;
  - R4: |attribution patching| on the real model (TwinLens `atp_baseline`, CLT summed over output
    layers);
  - R5: |activation × direct edge to the target| in G1;
  - R6: random, averaged over 5 seeds.
- Sizes: K ∈ {10, 30, 100, 300, 1000}, capped at |C|, plus the node set of the default
  `prune_graph(G, 0.8, 0.98)` of each variant.
- Interventions:
  - keep(K) sets every feature in C outside the top K to 0;
  - remove(K) sets the top K to 0.
  - Error nodes and embeddings are untouched.
- F_keep(K) = (m(keep K) − m(keep 0)) / (m(all) − m(keep 0)).
- F_remove(K) = (m(all) − m(remove K)) / (m(all) − m(keep 0)).
- A prompt with |m(all) − m(keep 0)| < 0.5 is excluded, and the count is reported.
- Score: the area under F over log10 K on the grid (trapezoid, normalised to [0, 1]).
- **Primary:** the paired Δ of the keep-area, R3 − R1, CLT, T1.
  - **positive:** lower bound > 0 and mean ≥ 0.03;
  - **negative:** upper bound < 0;
  - **null:** anything else.
- **Secondary:**
  - R2 − R1 under T2;
  - R3 − R4 and R3 − R5;
  - remove-area for the same pairs;
  - the same comparisons on the PLT;
  - nodes and edges kept by the default pruning of G1–G3.
- **PR #123:** positive → comment with the numbers. Null or negative → comment saying so and
  suggest closing.

**D10-B, dose**
- Edits: per prompt, the top 10 of R1 and 10 drawn at random from C (seeded).
- Each feature is set one at a time to s × its clean activation, s ∈ {−2, 0, 2, 5, 10}.
- Predictions:
  - Δa × the signed total effect of the feature on the target in each variant, from the
    graph's adjacency with per-unit edge weights (A_ts / a_s), through features only, with error
    nodes fixed;
  - Δa × attribution patching.
- Truth: T1, and T2 as secondary.
- Per s: ρ, sign agreement and within tolerance on |real| > 0.05, and the departure from the
  linear extrapolation of the true effect at s = 0 and s = 2.
- Descriptive, with no gate.

**D10-C, AtP with frozen gradients**
- Edits: the existing 56-prompt edits.
- AtP four ways: nothing detached (the existing value), LayerNorm scale detached, attention
  pattern detached, and both.
- Models: GPT-2 and Gemma-2-2B, with PLT and CLT features.
- Paired Δρ against the existing `real` truth.
- Reading: if "both detached" falls inside the interval of today's graph, the frozen-derivative
  effect holds without any transcoder.

**Cross-check (before the sweep):** on 3 prompts, the signed total effects from the G1 adjacency
must match the TwinLens `linear` level on the same features at r > 0.99. If they do not, the sweep
does not run until the difference is explained.

**Deviations** are written here, dated, before the analysis is run.

**Deviation, 2026-11-24 (before any D10 number).** The attribution-patching gradient of
`experiments/atp_baseline.py` was wrong: it detached every layer's MLP output, which cuts every
path from a layer through later MLPs. At layer 0 on an IOI prompt its gradient has cosine −0.02
with the real one. Found while testing `circuits_sweep.atp_grads`, which copied the pattern.
Both now add a zero leaf instead (`tests/test_atp.py`, `tests/test_ct.py`). Consequences:
- every attribution-patching number of the report (§4.8, §4.9, §4.10, §4.12, the abstract) is
  recomputed with the fixed gradient before it is used again, for GPT-2 locally and for Gemma
  PLT and CLT in the phase-2 server session;
- in D10-C, "nothing detached" is the recomputed value, not "the existing value";
- the comparison with attribution patching posted on circuit-tracer PR #123 must be corrected
  there once the recomputed numbers exist.

**Addition, 2026-11-24 (before any external number).** Attribution patching is also computed
on the D9 external edits (`atp_baseline.py --prompts external`, GPT-2, Gemma PLT and CLT;
nothing detached), so that "attribution patching predicts far better than the graph" is
tested on prompts we did not write. Reading as for D9: the claim replicates on a model if the
interval of the paired Δρ (AtP − graph today, and AtP − Fix B) on the external set lies above
zero.

**Implementation change, 2026-11-24 (before any D10 number).** Through `feature_intervention`,
one Gemma CLT prompt took about 110 minutes: about 450 single forward passes, with the vocab
unembedded at every position and, for T2, an extra clean pass per call. Interventions now run
through `twinlens/ct/fast.py`, a batched engine with the same semantics: live `encode_layer`
activations, Δ = value − live, and Δ · W_dec added at the feature-output hook. Only the last
position is unembedded, and the T2 patterns are cached once per prompt. `tests/test_ct.py`
checks that it equals `feature_intervention` to 1e-4 (PLT and CLT, T1 and T2). The design of
D10 is unchanged. The one CLT prompt run with the old path is discarded unread.
