"""Quantities read off a circuit-tracer ``Graph``, without running the model.

Node order in ``graph.adjacency_matrix`` (rows = targets, columns = sources) is
``[selected features, error nodes, token nodes, logit nodes]``. An edge into a feature or logit is
the source's contribution to the target's pre-activation (or demeaned logit): the source's
activation times the per-unit direct effect, with attention patterns, normalisation scales and
feature gates fixed as the graph was built.
"""

from __future__ import annotations

import torch


def _n_features(graph) -> int:
    return len(graph.selected_features)


def _logit_row(graph, target: int) -> int:
    n_logits = len(graph.logit_targets)
    if not 0 <= target < n_logits:
        raise IndexError(f"target {target} out of range for {n_logits} logit nodes")
    return graph.adjacency_matrix.shape[0] - n_logits + target


def feature_nodes(graph) -> tuple[torch.Tensor, torch.Tensor]:
    """``(features, activations)`` of the graph's feature nodes, in adjacency order:
    ``features[i] = (layer, pos, feature_idx)`` and ``activations[i]`` its clean activation."""
    sel = graph.selected_features
    return graph.active_features[sel], graph.activation_values[sel]


def node_influence(graph) -> torch.Tensor:
    """Influence of each feature node on the logit nodes, exactly as ``prune_graph`` ranks nodes
    (normalised |adjacency|, weighted by ``logit_probabilities``)."""
    from circuit_tracer.graph import compute_node_influence

    a = graph.adjacency_matrix
    w = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
    w[-len(graph.logit_targets) :] = graph.logit_probabilities.to(a.dtype)
    return compute_node_influence(a, w)[: _n_features(graph)]


def direct_contributions(graph, target: int = 0) -> torch.Tensor:
    """Each feature's direct edge to logit ``target``: activation × per-unit direct effect."""
    return graph.adjacency_matrix[_logit_row(graph, target), : _n_features(graph)].clone()


def total_effects(graph, target: int = 0, max_iter: int = 1000) -> torch.Tensor:
    """Signed total effect of one unit of each feature's activation on logit ``target``.

    Sums every path through feature nodes in the graph's linear model. Error and token nodes
    are sources only, so they are held fixed. Changing feature ``i``'s activation by ``Δa``
    is predicted to move the logit by ``Δa * total_effects(graph)[i]``. Features absent from a
    pruned or capped graph contribute nothing.
    """
    n = _n_features(graph)
    a = graph.adjacency_matrix.to(torch.float64)
    _, acts = feature_nodes(graph)
    acts = acts.to(torch.float64).to(a.device)
    if (acts == 0).any():
        raise ValueError("a feature node has zero activation; per-unit effects are undefined")
    per_unit = a[:n, :n] / acts[None, :]  # [target feature, source feature]
    current = a[_logit_row(graph, target), :n] / acts
    total = current.clone()
    for _ in range(max_iter):
        current = current @ per_unit
        if not current.any():
            return total.to(torch.float32)
        total += current
    raise RuntimeError(f"path sums did not terminate in {max_iter} steps (graph not acyclic?)")
