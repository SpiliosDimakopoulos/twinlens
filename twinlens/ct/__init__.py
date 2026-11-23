"""Checks of circuit-tracer attribution graphs against real interventions (DECISIONS.md D10).

Everything here reads a circuit-tracer ``Graph`` and drives a circuit-tracer ``ReplacementModel``,
so it works for any model and transcoder set circuit-tracer supports. circuit-tracer itself is
an optional dependency (``pip install twinlens[ct]``); :mod:`twinlens.ct.graph_math` needs only
torch.
"""

from twinlens.ct.graph_math import (
    direct_contributions,
    feature_nodes,
    node_influence,
    total_effects,
)

__all__ = ["direct_contributions", "feature_nodes", "node_influence", "total_effects"]
