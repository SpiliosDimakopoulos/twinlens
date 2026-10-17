---
license: mit
pretty_name: TwinLens fidelity sweeps
task_categories:
  - other
tags:
  - interpretability
  - mechanistic-interpretability
  - attribution-graphs
  - transcoders
  - gpt2
  - qwen3
size_categories:
  - 1K<n<10K
configs:
  - config_name: gpt2
    data_files: fidelity.parquet
    default: true
  - config_name: qwen3
    data_files: fidelity_qwen3.parquet
---

# TwinLens fidelity sweeps

Per-intervention results from TwinLens, a digital twin of a language model built from transcoders. Each row is one feature edit. It holds the effect predicted by each twin level next to the effect measured in the real model.

- `fidelity.parquet`: GPT-2 small, 8,064 edits on 56 prompts, 12 layers.
- `fidelity_qwen3.parquet`: Qwen3-0.6B, 2,304 edits on 16 prompts, layers 10 to 27.

Headline result (GPT-2): the linear graph twin has Spearman ρ = 0.20 with the real effect. Restoring the LayerNorm and attention Jacobians gives ρ = 0.53 while staying linear. See the [report](https://spiliosdimakopoulos.github.io/twinlens/report/).

## Columns

Metadata (both files):

| column | meaning |
|---|---|
| `prompt_id`, `category`, `prompt` | prompt index, task family, prompt text |
| `feature`, `pos`, `layer`, `kind`, `selection` | edited transcoder feature, token position, layer, edit kind (e.g. ablate), and how the edit was chosen (top attribution edge or random) |
| `layers_after`, `pos_from_end` | layers between the edit and the output; distance of the position from the last token |
| `activation`, `delta_act`, `rel_edit_size` | feature activation, size of the change, change relative to activation |
| `direct_effect`, `abs_direct_effect` | direct-edge effect on the target logit |
| `downstream_err_frac`, `attn_to_pos`, `n_active_downstream`, `clean_top1_prob` | clean-run signals used by the trust model |

Outcome columns come in groups per twin level, with suffixes `_dlogp` (change in target log-probability), `_kl`, `_flip` (top-1 changed) and `_cos` (cosine with the real change; absent for `real_`):

| prefix | level |
|---|---|
| `real_` | the real model (ground truth) |
| `direct_` | direct edges only |
| `linear_` | linear graph twin (attribution graph as published) |
| `linear_lnjac_` | graph + LayerNorm Jacobian (Fix A) |
| `linear_fulljac_` | graph + LayerNorm and attention Jacobians (Fix B) |
| `replacement_`, `replacement_ln_`, `replacement_full_` | nonlinear replacement twins (frozen, live LayerNorm, live LayerNorm and attention); GPT-2 file only |

`fidelity_qwen3.parquet` has the same columns minus the three `replacement_*` groups.

## Usage

```python
from datasets import load_dataset
from scipy.stats import spearmanr

gpt2 = load_dataset("SpiliosDimakopoulos/twinlens-fidelity", "gpt2", split="train").to_pandas()
print(spearmanr(gpt2.real_dlogp, gpt2.linear_fulljac_dlogp).statistic)
```

## Caveats

Prompts were written by the author. Interventions move the feature's decoder direction in the real model. See the report's limitations section.

## Citation

Dimakopoulos, S. (2026). *TwinLens: an interpretability digital twin for LLMs.* https://github.com/SpiliosDimakopoulos/twinlens
