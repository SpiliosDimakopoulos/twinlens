/** Plain-language definitions shown by <Info> popovers. */
export const GLOSSARY = {
  feature: (
    <>A <b>transcoder feature</b> is one of 24,576 interpretable directions per layer that replace GPT-2's MLP. Each fires on a recognisable pattern (a name, a topic, a syntax cue).</>
  ),
  directEffect: (
    <>The <b>direct effect</b> is how much a feature pushes the predicted token's logit through the residual stream, with everything else held fixed. It is the edge weight an attribution graph draws.</>
  ),
  twin: (
    <>The <b>twin</b> is an interpretable copy of GPT-2: MLPs replaced by transcoders, other parts frozen or live depending on the level. Level 2 ("linear") is what attribution graphs show.</>
  ),
  trust: (
    <>The <b>trust score</b> is a model trained on 8,064 real interventions. From the clean run alone it predicts whether the graph's prediction will be within tolerance of what GPT-2 really does (AUROC 0.93 on unseen prompts).</>
  ),
  error: (
    <>The part of each MLP's output the transcoder does not reconstruct. The twin carries it as a fixed "error" term, which is computation it cannot explain.</>
  ),
};
