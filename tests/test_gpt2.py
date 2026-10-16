import pytest
import torch

from twinlens.core.weights import CACHE_DIR


@pytest.mark.slow
def test_matches_huggingface(model):
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", cache_dir=CACHE_DIR / "hub").eval()
    toks = model.to_tokens(["The Eiffel Tower is in the city of", "When Mary and John went to the store, the"])
    with torch.no_grad():
        ref = hf(toks).logits
    assert (model(toks) - ref).abs().max() < 1e-3
    del hf


def test_cache_shapes(model):
    toks = model.to_tokens("Hello world, this is a test")
    _, cache = model.run_with_cache(toks)
    b, p = toks.shape
    assert cache["blocks.3.attn.hook_pattern"].shape == (b, 12, p, p)
    assert cache["blocks.3.attn.hook_z"].shape == (b, p, 12, 64)
    assert cache["blocks.3.mlp.hook_post"].shape == (b, p, 3072)
    pat = cache["blocks.3.attn.hook_pattern"]
    torch.testing.assert_close(pat.sum(-1), torch.ones_like(pat.sum(-1)))
    assert torch.all(pat.triu(1) == 0), "attention must be causal"


def test_hook_result_sums_to_attn_out(model):
    toks = model.to_tokens("The cat sat on the mat")
    _, cache = model.run_with_cache(toks, ["blocks.5.attn.hook_result", "blocks.5.hook_attn_out"])
    b_O = model.blocks[5]["b_O"]
    torch.testing.assert_close(
        cache["blocks.5.attn.hook_result"].sum(2) + b_O, cache["blocks.5.hook_attn_out"], atol=1e-4, rtol=1e-4
    )


def test_noop_hook_is_identity_and_hooks_are_removed(model):
    toks = model.to_tokens("A noop hook must not change anything")
    base = model(toks)
    with model.hooks([("blocks.7.hook_mlp_out", lambda x, n: x)]):
        torch.testing.assert_close(model(toks), base)
    assert not model._hooks


def test_zero_ablation_changes_output(model):
    toks = model.to_tokens("The capital of France is")
    base = model(toks)
    with model.hooks([("blocks.0.hook_mlp_out", lambda x, n: torch.zeros_like(x))]):
        assert (model(toks) - base).abs().max() > 1.0
