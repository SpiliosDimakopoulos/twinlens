"""Request models. Field limits are part of the HTTP contract."""

from __future__ import annotations

from pydantic import BaseModel, Field

from twinlens.causal.interventions import Intervention

MAX_CHARS = 1000  # rejects huge strings before they are tokenised
MAX_INTERVENTIONS = 16
N_TWIN_LAYERS = 12  # GPT-2 small
# Transcoder width per layer (gpt2small-transcoders). Fixed by the checkpoint; kept here as a
# constant because the loaded Transcoder only exposes it as an instance property (needs weights).
D_SAE = 24576

PromptField = Field(min_length=1, max_length=MAX_CHARS)


class RunRequest(BaseModel):
    prompt: str = PromptField
    target: str | None = Field(None, max_length=50)  # token to attribute; default: prediction


class InterveneRequest(BaseModel):
    prompt: str = PromptField
    interventions: list[Intervention] = Field(min_length=1, max_length=MAX_INTERVENTIONS)


class WhatIfRequest(BaseModel):
    prompt: str = PromptField
    layer: int = Field(ge=0, lt=N_TWIN_LAYERS)
    pos: int = Field(ge=0)
    feature: int = Field(ge=0, lt=D_SAE)
    value: float = Field(0.0, ge=-1e4, le=1e4, allow_inf_nan=False)


class PatchRequest(BaseModel):
    clean: str = PromptField
    corrupt: str = PromptField
    correct: str = Field(min_length=1, max_length=50)
    wrong: str = Field(min_length=1, max_length=50)
