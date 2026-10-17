# Attribution graphs predict interventions poorly, and the fix keeps them linear

**TL;DR.** I measured how well the linear replacement model behind attribution graphs predicts what happens when you intervene on a feature in the real model. On 8,064 interventions in GPT-2 small, the rank correlation is ρ = 0.20. Nearly all of the error comes from two derivatives the graph drops: the LayerNorm scale and the attention pattern. Adding both Jacobians back keeps the graph exactly linear and raises ρ to 0.53, the same as the full nonlinear replacement model (0.52). It replicates on Qwen3-0.6B. I have opened a PR to circuit-tracer with opt-in flags.

## The question

Attribution graphs say that feature A drives feature B drives the answer. The implied claim is counterfactual: if you push A, the output changes in the way the graph says. The graph is a local linear model, so this is a testable prediction. The question I wanted answered edge by edge: when you actually intervene, does the real model agree?

## Setup

I built TwinLens, a digital twin of the model's internals, and compared a ladder of twins against the real model. Each rung unfreezes one more thing:

1. direct edges only
2. the linear graph (what attribution graphs describe)
3. the replacement model with frozen LayerNorm scales and attention patterns
4. the same with live LayerNorm
5. the same with live LayerNorm and live attention

For each edit I record the predicted change in log-probability at every rung and the real change from the real model. GPT-2 small is the main testbed: 56 prompts across eight task families, 8,064 interventions, using per-layer MLP transcoders. I wrote GPT-2 myself with hook points and checked it against Hugging Face logits (max abs difference 7×10⁻⁵). Before measuring anything new, it reproduces induction heads, IOI name movers and ROME causal tracing as unit tests. For replication I used Qwen3-0.6B with circuit-tracer's transcoders: 2,304 edits on 16 prompts, layers 10 to 27 of 28.

## Finding

The linear twin is a weak predictor: ρ = 0.20 [0.14, 0.25]. On edits that visibly move the output it is within tolerance only 54% of the time, and 2.2% of its predictions are catastrophic misses. It is worst where attention does the work: ρ is 0.01 on IOI and 0.07 on induction.

Decomposing the replacement twin's error, frozen LayerNorm accounts for 88% of the mean error and almost all catastrophic misses. Frozen attention accounts for most of the remaining ranking error. Both freezes remove a first-order term. That is a different problem from linearity.

## The fix and results

Freezing a scale or a pattern treats it as a constant. The derivative is not zero, so I restore it. For LayerNorm, a perturbation da of the normalised input moves the output by the projection `dn = da − n0(n0·da)/d`, scaled by the inverse standard deviation. For attention, the softmax pattern responds to perturbed queries and keys, and that term is added as a Jacobian. The result is still exactly linear, so effects stay additive over sources and the graph stays cheap to compute.

| GPT-2 small | graph today | + LN and attention Jacobians | nonlinear twin |
|---|---|---|---|
| Spearman ρ | 0.20 | 0.53 [0.49, 0.56] | 0.52 |
| catastrophic misses | 2.2% | 0.05% | 0.2% |
| IOI ρ | 0.01 | 0.59 | |

Paired improvement over the graph today, same edits, 95% cluster bootstrap over prompts:

- GPT-2 small: Δρ +0.33 [0.27, 0.39], Δ sign agreement +19 pp [15, 24], Δ catastrophic −2.2 pp [−2.7, −1.7]
- Qwen3-0.6B: Δρ +0.12 [0.05, 0.20], Δ sign agreement +9 pp [2, 17], Δ catastrophic −2.2 pp [−3.7, −0.8]

All intervals exclude zero. The gain on Qwen3 is smaller and I have not tested why. A possible cause is larger residual transcoder error, which no Jacobian can repair.

![Fix results](https://raw.githubusercontent.com/SpiliosDimakopoulos/twinlens/main/experiments/figures/F5_fix.png)

![Generalisation to Qwen3-0.6B](https://raw.githubusercontent.com/SpiliosDimakopoulos/twinlens/main/experiments/figures/F7_generalization.png)

Since the fixed graph is linear and matches the nonlinear twin, the remaining gap to the real model comes from the transcoders, not from linearisation.

## Trust model

A logistic model over nine signals computed from the clean run alone (edit size, activation, layers to the output, and so on) predicts whether a graph prediction will hold. AUROC is 0.93 on held-out prompts and 0.93 on held-out task families (range 0.88 to 0.98). Among top edges with a noticeable effect, today's graph is right 54% of the time; on edges the trust model clears it is right 71%, and on flagged edges 22%.

## Limitations

- Two models and two transcoder families. Cross-layer transcoders and larger models are untested.
- The intervention moves the feature's decoder direction in the real model. That is one definition of "this edge matters".
- The tolerance threshold (0.1 nats + 25%) is a choice, though rank correlation and cosine use none and give the same ordering.
- 56 prompts, written by me. The Qwen3 replication covers 16 prompts and the upper layers only.
- The gain is not uniform across families: it is roughly flat on code and narrative.

## Links

- Demo (runs in the browser): https://spiliosdimakopoulos.github.io/twinlens/
- Report: https://spiliosdimakopoulos.github.io/twinlens/report/
- Code: https://github.com/SpiliosDimakopoulos/twinlens
- PR to circuit-tracer adding opt-in `freeze_attention` / `freeze_layernorm` flags to `attribute()`: https://github.com/decoderesearch/circuit-tracer/pull/123

I would like to hear from anyone who has run similar intervention checks on cross-layer transcoders.

*Spilios Dimakopoulos (GitHub: SpiliosDimakopoulos)*
