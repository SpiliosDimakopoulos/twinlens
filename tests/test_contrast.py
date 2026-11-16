"""Second target of plan task 2.2: change of the logit difference answer - contrast."""

import torch

from twinlens.fidelity.engine import _effect_metrics


def test_logit_difference_change():
    g = torch.Generator().manual_seed(0)
    clean, out = torch.randn(11, generator=g), torch.randn(3, 11, generator=g)
    m = _effect_metrics(clean, out, target=4, contrast=7)
    want = (out[:, 4] - out[:, 7]) - (clean[4] - clean[7])
    assert torch.allclose(m["dld"], want)
    assert torch.allclose(_effect_metrics(clean, clean[None], 4, 7)["dld"], torch.zeros(1))


def test_no_contrast_keeps_old_columns():
    clean, out = torch.zeros(5), torch.ones(2, 5)
    assert set(_effect_metrics(clean, out, 0)) == {"dlogp", "kl", "top1", "dlogits"}
