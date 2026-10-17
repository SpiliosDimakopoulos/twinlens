# TwinLens: CV material

Every number below comes from `experiments/results/summary.json`, `bench/latency.json` or the
test suite. Keep them in sync if you re-run anything.

## CV entry (English)

**TwinLens: an interpretability digital twin for LLMs** · Python, PyTorch, FastAPI, React/TypeScript · *2026*
Live demo: spiliosdimakopoulos.github.io/twinlens · Code: github.com/SpiliosDimakopoulos/twinlens
- Built a "digital twin" of GPT-2's internals: a transcoder-based replacement model that
  predicts the effect of any feature intervention. The real model then verifies each
  prediction, and the dashboard shows predicted vs actual live.
- Ran an 8,064-intervention study across 56 prompts. Attribution graphs predict intervention
  effects poorly (Spearman ρ = 0.20), and the error traces to two derivatives they drop
  (LayerNorm scale, attention pattern).
- Proposed and validated a fix that keeps graphs linear: restoring both Jacobians raises ρ from
  0.20 to 0.53 (matching the nonlinear model), cuts catastrophic misses from 2.2% to 0.05%, and
  lifts IOI from 0.01 to 0.59 — and replicated it on Qwen3-0.6B (different architecture).
- Upstreamed the fix to the open-source circuit-tracer library as opt-in exact-gradient
  attribution flags (TransformerLens + nnsight backends, finite-difference tests): PR #123.
- Designed a trust model that predicts, before running the real model, whether an explanation
  will hold: AUROC 0.93 on held-out prompts and on held-out task families.
- Wrote a hookable GPT-2 from scratch with memory-mapped weights (<1 GB RAM on a laptop CPU).
  It reproduces published circuits (induction heads, IOI name movers, ROME causal tracing),
  pinned in a 33-test suite plus Playwright end-to-end tests.

*Short version (1 line):* Built TwinLens, an LLM interpretability "digital twin" for GPT-2.
Showed that attribution graphs fail because of two dropped derivatives, and that restoring them
raises counterfactual accuracy from ρ 0.20 to 0.53 while keeping graphs linear.

## Βιογραφικό (Ελληνικά)

**TwinLens: Digital twin ερμηνευσιμότητας για LLMs** · Python, PyTorch, FastAPI, React/TypeScript · *2026*
Live demo: spiliosdimakopoulos.github.io/twinlens · Κώδικας: github.com/SpiliosDimakopoulos/twinlens
- Ανέπτυξα έναν «ψηφιακό δίδυμο» του εσωτερικού του GPT-2: ένα μοντέλο αντικατάστασης με
  transcoders που προβλέπει το αποτέλεσμα κάθε παρέμβασης σε features. Το πραγματικό μοντέλο
  επαληθεύει κάθε πρόβλεψη, και το dashboard δείχνει πρόβλεψη και πραγματικότητα live.
- Έτρεξα μελέτη με 8.064 παρεμβάσεις σε 56 prompts. Τα attribution graphs προβλέπουν φτωχά
  το αποτέλεσμα των παρεμβάσεων (Spearman ρ = 0,20), και το σφάλμα οφείλεται σε δύο
  παραγώγους που πετάνε (LayerNorm και attention pattern).
- Πρότεινα και επαλήθευσα διόρθωση που κρατά τα graphs γραμμικά: με τις δύο Jacobians το ρ
  ανεβαίνει από 0,20 σε 0,53 (όσο και το μη γραμμικό μοντέλο), οι καταστροφικές αστοχίες
  πέφτουν από 2,2% σε 0,05%, και στο IOI από 0,01 σε 0,59 — και την επιβεβαίωσα και στο Qwen3-0.6B, διαφορετική αρχιτεκτονική.
- Συνεισέφερα τη διόρθωση στη βιβλιοθήκη ανοιχτού κώδικα circuit-tracer ως προαιρετικά flags
  ακριβών gradients (TransformerLens + nnsight, με finite-difference tests): PR #123.
- Σχεδίασα trust model που προβλέπει, πριν τρέξει το πραγματικό μοντέλο, αν μια εξήγηση θα
  ισχύσει: AUROC 0,93 σε prompts και σε κατηγορίες εργασιών που δεν είχε δει.
- Υλοποίησα από το μηδέν GPT-2 με hooks και memory-mapped βάρη (<1 GB RAM σε CPU laptop). Αναπαράγει
  δημοσιευμένα κυκλώματα (induction heads, IOI, ROME causal tracing), καλυμμένα από 33 tests
  και end-to-end tests με Playwright.

## LinkedIn post (draft)

> When an attribution graph says "this feature drives the answer", how often is that true?
>
> I built TwinLens to find out. It's a digital twin of GPT-2's internals: an interpretable
> replacement model that predicts what happens when you switch a feature off. The real model
> then checks the prediction.
>
> Across 8,064 interventions:
> - The linear twin that attribution graphs describe ranks real effects poorly (ρ = 0.20).
> - The error comes from two derivatives the graph drops (LayerNorm and attention). Put them
>   back and the graph stays linear but becomes as faithful as the full nonlinear model:
>   ρ 0.20 → 0.53, catastrophic misses 2.2% → 0.05%.
> - It holds on a second architecture too (Qwen3-0.6B).
> - A small trust model can predict *in advance*, from the clean run alone, when an
>   explanation will hold (AUROC 0.93 on held-out prompts).
>
> Everything runs on a laptop CPU in under 1 GB of RAM, because I wrote the model from scratch
> with memory-mapped weights. The tool first reproduces known circuits (induction heads, IOI,
> ROME) before measuring anything new.
>
> Live demo (runs in the browser): https://spiliosdimakopoulos.github.io/twinlens/
> Report + code: https://github.com/SpiliosDimakopoulos/twinlens

## 60-second interview pitch

"Interpretability tools explain LLMs through a simplified copy of the model, a twin. But nobody
checks, edge by edge, whether that copy behaves like the real thing when you intervene. I built
a system that runs both side by side. The twin predicts the effect of an edit, the real model
answers, and I measured the gap on 8,000 edits. Three things came out. The standard linear twin
is a weak predictor. Most of its error comes from one frozen component, LayerNorm. And a simple
model can flag the bad explanations before you run the experiment. To make it run on my laptop
I wrote GPT-2 myself with memory-mapped weights, and to trust my own measurements I made the
code reproduce three published circuits as unit tests first."

**Likely follow-up questions, and where the answers are:**
- *Why transcoders, not SAEs?* Transcoders replace the MLP's input-to-output map, so the
  replacement model is a forward model you can intervene in. See REPORT §2.
- *Isn't 90% within tolerance high?* Yes, because most edits barely matter. On high-effect edits
  it's 54%. That's why the report shows that subset separately (REPORT §3, §4).
- *How do you know the implementation is right?* HF parity to 7e-5; induction/IOI/ROME
  reproductions; invariants (exact clean reproduction, affinity, attribution = simulation).
- *What would you do next?* Cross-layer transcoders and a larger model (Gemma-2-2B) on a GPU,
  and testing whether the trust model transfers.
