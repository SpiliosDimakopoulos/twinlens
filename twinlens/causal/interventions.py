"""JSON-serialisable interventions on the real model.

Each spec turns into a list of ``(hook_name, fn)`` pairs for ``GPT2.hooks``. Specs are pydantic
models so the API and the frontend can send them as-is.
"""

from __future__ import annotations

from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field

from twinlens.core.gpt2 import GPT2, HookFn

Hooks = list[tuple[str, HookFn]]


def _pos_index(positions: list[int] | None) -> slice | list[int]:
    return slice(None) if positions is None else positions


class HeadAblation(BaseModel):
    kind: Literal["head"] = "head"
    layer: int
    head: int
    mode: Literal["zero", "mean"] = "zero"
    scale: float = 0.0  # 0 = full ablation, 1 = no-op, >1 = amplify
    positions: list[int] | None = None

    def hooks(self, model: GPT2, mean_cache: dict[str, torch.Tensor] | None = None) -> Hooks:
        name = f"blocks.{self.layer}.attn.hook_z"
        pos = _pos_index(self.positions)

        def fn(z: torch.Tensor, _: str) -> torch.Tensor:
            z = z.clone()
            ref = torch.zeros_like(z[:, pos, self.head])
            if self.mode == "mean":
                if mean_cache is None or name not in mean_cache:
                    raise ValueError("mean ablation needs a mean_cache with hook_z")
                ref = mean_cache[name][..., self.head, :].expand_as(z[:, pos, self.head])
            z[:, pos, self.head] = ref + self.scale * (z[:, pos, self.head] - ref)
            return z

        return [(name, fn)]


class NeuronEdit(BaseModel):
    kind: Literal["neuron"] = "neuron"
    layer: int
    neuron: int
    value: float | None = None  # set to a value...
    scale: float = 0.0  # ...or multiply (0 = zero-ablate)
    positions: list[int] | None = None

    def hooks(self, model: GPT2, mean_cache: dict[str, torch.Tensor] | None = None) -> Hooks:
        pos = _pos_index(self.positions)

        def fn(post: torch.Tensor, _: str) -> torch.Tensor:
            post = post.clone()
            if self.value is not None:
                post[:, pos, self.neuron] = self.value
            else:
                post[:, pos, self.neuron] *= self.scale
            return post

        return [(f"blocks.{self.layer}.mlp.hook_post", fn)]


class Steering(BaseModel):
    """Activation addition (Turner et al., 2023): add ``coeff * (h(plus) - h(minus))``."""

    kind: Literal["steer"] = "steer"
    layer: int
    plus: str
    minus: str
    coeff: float = 5.0

    def vector(self, model: GPT2) -> torch.Tensor:
        name = f"blocks.{self.layer}.hook_resid_pre"
        acts = []
        for text in (self.plus, self.minus):
            _, cache = model.run_with_cache(model.to_tokens(text), [name])
            acts.append(cache[name][0])
        n = min(a.shape[0] for a in acts)
        return acts[0][:n] - acts[1][:n]  # [n_pos, d_model], aligned from the BOS token

    def hooks(self, model: GPT2, mean_cache: dict[str, torch.Tensor] | None = None) -> Hooks:
        vec = self.vector(model) * self.coeff

        def fn(resid: torch.Tensor, _: str) -> torch.Tensor:
            # Only the prompt's first positions are steered (as in ActAdd), and only on the
            # full forward pass (positions beyond the vector are left alone).
            n = min(vec.shape[0], resid.shape[1])
            resid = resid.clone()
            resid[:, :n] += vec[:n]
            return resid

        return [(f"blocks.{self.layer}.hook_resid_pre", fn)]


class DirectionEdit(BaseModel):
    """Add ``delta * direction`` to a hook (used for transcoder-feature edits on the real model)."""

    kind: Literal["direction"] = "direction"
    hook: str
    direction: list[float]
    delta: float
    positions: list[int] | None = None

    def hooks(self, model: GPT2, mean_cache: dict[str, torch.Tensor] | None = None) -> Hooks:
        d = torch.tensor(self.direction)
        pos = _pos_index(self.positions)

        def fn(x: torch.Tensor, _: str) -> torch.Tensor:
            x = x.clone()
            x[:, pos] += self.delta * d
            return x

        return [(self.hook, fn)]


Intervention = Annotated[
    HeadAblation | NeuronEdit | Steering | DirectionEdit, Field(discriminator="kind")
]


def build_hooks(
    model: GPT2,
    interventions: list[Intervention],
    mean_cache: dict[str, torch.Tensor] | None = None,
) -> Hooks:
    hooks: Hooks = []
    for spec in interventions:
        hooks += spec.hooks(model, mean_cache)
    return hooks


def mean_cache_for(model: GPT2, tokens: torch.Tensor, names: list[str]) -> dict[str, torch.Tensor]:
    """Mean activation over the batch (per position) for mean ablation."""
    _, cache = model.run_with_cache(tokens, names)
    return {k: v.mean(0, keepdim=True) for k, v in cache.items()}
