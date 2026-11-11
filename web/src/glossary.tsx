/** Plain-language definitions shown by <Info> popovers. */
export const GLOSSARY = {
  feature: (
    <>A <b>transcoder feature</b> is one of 24,576 interpretable directions per layer that replace GPT-2's MLP. Each fires on a recognisable pattern (a name, a topic, a syntax cue).</>
  ),
  directEffect: (
    <>The <b>direct effect</b> is how much a feature pushes the predicted word up or down by itself, ignoring what it does to later features. It is the edge weight an attribution graph draws. Measured in logits: +1 makes the word about 2.7× more likely relative to the alternatives.</>
  ),
  graph: (
    <>Each dot is a <b>feature</b>: a pattern GPT-2 detects at one word (column) and one layer (row; layer 0 is the first, 11 the last). The line from a dot to the answer shows its direct push on the prediction. Bigger dot, bigger push.</>
  ),
  twin: (
    <>The <b>twin</b> is an interpretable copy of GPT-2: MLPs replaced by transcoders, other parts frozen or live depending on the level. Level 2 ("linear") is what attribution graphs show.</>
  ),
  dlogp: (
    <>The numbers are the change in GPT-2's <b>log-probability</b> of the predicted word, in nats. 0 means no change. −1 means the word becomes about 2.7× less likely, +1 about 2.7× more likely; −0.1 is a 10% drop.</>
  ),
  ladder: (
    <>
      <b>Each row is a different way of predicting the same edit.</b>
      <span className="mt-1 block"><b>Graph today</b>: the attribution graph as drawn now. LayerNorm and attention are frozen at their unedited values.</span>
      <span className="mt-1 block"><b>Direct edges only</b>: just the clicked feature's own edge, nothing downstream.</span>
      <span className="mt-1 block"><b>Replacement</b>, <b>+ live LayerNorm</b>, <b>+ live attention</b>: the same twin with one more part of the real model switched back on each time.</span>
      <span className="mt-1 block"><b>Graph with the fix</b>: the proposed fix. Still a linear graph, but it keeps how LayerNorm and attention react to the edit.</span>
      <span className="mt-1 block">The vertical line is what GPT-2 actually does.</span>
    </>
  ),
  trust: (
    <>The <b>trust score</b> estimates how likely it is that the graph's prediction for this edit is close to what GPT-2 really does. It is a small model trained on 8,064 real edits and uses only the unedited run. It separates good from bad predictions well (AUROC 0.84 on edits that matter, where 0.5 is guessing and 1 is perfect). A very large prediction from the graph is a warning sign by itself.</>
  ),
  generation: (
    <>Inside GPT-2, every layer already holds a guess for the next word (the <b>logit lens</b>). A token is shaded by the first layer from which that guess is already the word GPT-2 finally writes. Light: decided early. Dark: decided only in the last layers.</>
  ),
  error: (
    <>The part of each MLP's output the transcoder does not reconstruct. The twin carries it as a fixed "error" term, which is computation it cannot explain.</>
  ),
  metrics: (
    <>
      <span className="block"><b>Rank correlation ρ</b>: whether the edits predicted to matter most are the ones that really do (1 = same order, 0 = no relation).</span>
      <span className="mt-1 block"><b>Median cosine</b>: whether the predicted change across all words points the same way as the real one (1 = same direction).</span>
      <span className="mt-1 block"><b>Sign agreement</b>: share of edits where the prediction gets the direction (up or down) right, among edits that move the answer noticeably. 50% is a coin flip.</span>
      <span className="mt-1 block"><b>Catastrophic misses</b>: predictions off by more than 1 nat (lower is better).</span>
      <span className="mt-1 block">For the first three, higher is better.</span>
    </>
  ),
  auroc: (
    <><b>AUROC</b> measures how well a score separates right from wrong predictions: 0.5 is guessing, 1 is perfect. It is the chance that a random correct prediction gets a higher score than a random wrong one.</>
  ),
  logitLens: (
    <>The <b>logit lens</b> asks what GPT-2 would predict if it stopped after a given layer. Each row decodes the model's internal state after one more layer.</>
  ),
  dla: (
    <><b>Direct logit attribution</b>: how much each attention head, by itself, pushes the answer up (blue) or down (red), in logits.</>
  ),
  kl: (
    <><b>KL divergence</b> measures how much the whole next-word distribution changed. 0 means unchanged; larger means a bigger change.</>
  ),
  ioi: (
    <>In "When Mary and John went to the store, John gave a drink to", the right answer is the name that appears once, Mary (the <b>indirect object, IO</b>). The repeated name, John, is the <b>subject, S</b>. <b>IO − S</b> is how much the model prefers Mary over John, in logits.</>
  ),
};
