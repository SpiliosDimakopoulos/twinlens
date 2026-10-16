# TwinLens: CV material

Every number below comes from `experiments/results/summary.json`, `bench/latency.json` or the
test suite. Keep them in sync if you re-run anything.

## CV entry (English)

**TwinLens: an interpretability digital twin for LLMs** · Python, PyTorch, FastAPI, React/TypeScript · *2026*
- Built a "digital twin" of GPT-2's internals: a transcoder-based replacement model that
  predicts the effect of any feature intervention. The real model then verifies each
  prediction, and the dashboard shows predicted vs actual live.
- Ran an 8,064-intervention study across 56 prompts. Attribution-graph-style twins rank effects
  poorly (Spearman ρ = 0.20). Frozen LayerNorm causes 88% of their error; unfreezing it cuts
  catastrophic misses from 4.3% to 0.2%, and live attention raises ρ to 0.52.
- Designed a trust model that predicts, before running the real model, whether an explanation
  will hold. AUROC 0.93 on held-out prompts; the trusted half is right 99.8% of the time vs 81%
  for the flagged half.
- Wrote a hookable GPT-2 from scratch with memory-mapped weights (<1 GB RAM on a laptop CPU).
  It reproduces published circuits (induction heads, IOI name movers, ROME causal tracing),
  pinned in a 33-test suite plus Playwright end-to-end tests.

*Short version (1 line):* Built TwinLens, an LLM interpretability "digital twin" that predicts
intervention effects inside GPT-2 and scores how far each explanation can be trusted (AUROC
0.93); found frozen LayerNorm causes 88% of attribution-graph error.

## Βιογραφικό (Ελληνικά)

**TwinLens: Digital twin ερμηνευσιμότητας για LLMs** · Python, PyTorch, FastAPI, React/TypeScript · *2026*
- Ανέπτυξα έναν «ψηφιακό δίδυμο» του εσωτερικού του GPT-2: ένα μοντέλο αντικατάστασης με
  transcoders που προβλέπει το αποτέλεσμα κάθε παρέμβασης σε features. Το πραγματικό μοντέλο
  επαληθεύει κάθε πρόβλεψη, και το dashboard δείχνει πρόβλεψη και πραγματικότητα live.
- Έτρεξα μελέτη με 8.064 παρεμβάσεις σε 56 prompts. Οι twins τύπου attribution graph
  κατατάσσουν τα αποτελέσματα φτωχά (Spearman ρ = 0,20). Το «παγωμένο» LayerNorm προκαλεί το
  88% του σφάλματός τους· όταν γίνει live, οι καταστροφικές αστοχίες πέφτουν από 4,3% σε 0,2%,
  και η live attention ανεβάζει το ρ στο 0,52.
- Σχεδίασα trust model που προβλέπει, πριν τρέξει το πραγματικό μοντέλο, αν μια εξήγηση θα
  ισχύσει: AUROC 0,93 σε prompts που δεν είχε δει. Στο μισό με την υψηλότερη εμπιστοσύνη
  πετυχαίνει 99,8%, έναντι 81% στο μισό που επισημαίνει ως αναξιόπιστο.
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
> - 88% of its error comes from one assumption: frozen LayerNorm. Unfreezing it cuts
>   catastrophic misses from 4.3% to 0.2%.
> - A small trust model can predict *in advance*, from the clean run alone, when an
>   explanation will hold (AUROC 0.93 on held-out prompts).
>
> Everything runs on a laptop CPU in under 1 GB of RAM, because I wrote the model from scratch
> with memory-mapped weights. The tool first reproduces known circuits (induction heads, IOI,
> ROME) before measuring anything new.
>
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
