"""Prompt datasets for the reproduction suite and the fidelity sweep."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from twinlens.core.gpt2 import GPT2

# --------------------------------------------------------------------------- IOI
# Templates and names follow Wang et al. (2022), "Interpretability in the Wild".
IOI_TEMPLATES = [
    "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give it to",
    "After {B} and {A} went to the {PLACE}, {B} gave a {OBJECT} to",
    "While {B} and {A} were working at the {PLACE}, {B} gave a {OBJECT} to",
    "Then, {B} and {A} had a long argument, and afterwards {B} said to",
    "Friends {B} and {A} found a {OBJECT} at the {PLACE}. {B} gave it to",
]
NAMES = [
    "Mary",
    "John",
    "Tom",
    "James",
    "Dan",
    "Sid",
    "Martin",
    "Amy",
    "Alice",
    "Bob",
    "Kate",
    "Sarah",
    "Paul",
    "Mark",
    "Lisa",
    "Anna",
    "David",
    "Peter",
    "Laura",
    "Michael",
    "Emma",
    "Jack",
    "Ben",
    "Steve",
    "Rachel",
    "Chris",
    "Jessica",
    "Eric",
    "George",
    "Andrew",
]
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "station", "house"]
OBJECTS = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack"]


@dataclass
class IOIBatch:
    prompts: list[str]
    tokens: torch.Tensor  # [batch, pos]
    io: torch.Tensor  # indirect-object token ids (correct answer)
    s: torch.Tensor  # subject token ids
    corrupt_tokens: torch.Tensor  # same templates with three unrelated names ("ABC" distribution)
    template: str


def _single_token(model: GPT2, words: list[str]) -> list[str]:
    return [w for w in words if len(model.tokenizer.encode(" " + w, prepend_bos=False)) == 1]


def ioi_batch(model: GPT2, n: int = 32, template_idx: int = 0, seed: int = 0) -> IOIBatch:
    rng = random.Random(seed)
    names = _single_token(model, NAMES)
    places = _single_token(model, PLACES)
    objects = _single_token(model, OBJECTS)
    template = IOI_TEMPLATES[template_idx]
    prompts, corrupt, io, s = [], [], [], []
    for _ in range(n):
        a, b, c, d = rng.sample(names, 4)
        fill = {"PLACE": rng.choice(places), "OBJECT": rng.choice(objects)}
        prompts.append(template.format(A=a, B=b, **fill))
        # ABC corruption: every name replaced, so no IO/S information remains
        corrupt.append(template.format(A=c, B=d, **fill))
        io.append(model.to_single_token(" " + a))
        s.append(model.to_single_token(" " + b))
    return IOIBatch(
        prompts,
        model.to_tokens(prompts),
        torch.tensor(io),
        torch.tensor(s),
        model.to_tokens(corrupt),
        template,
    )


# ---------------------------------------------------------------------- induction
def repeated_random_tokens(
    model: GPT2, n_batch: int = 8, seq_len: int = 50, seed: int = 0
) -> torch.Tensor:
    """[batch, 1 + 2*seq_len]: BOS + random tokens repeated twice (Olsson et al., 2022)."""
    gen = torch.Generator().manual_seed(seed)
    rand = torch.randint(1000, 20000, (n_batch, seq_len), generator=gen)
    bos = torch.full((n_batch, 1), model.tokenizer.bos_id)
    return torch.cat([bos, rand, rand], dim=1)


# ------------------------------------------------------------------ factual recall
# (prompt, subject, answer). Filtered at runtime to the facts GPT-2 small gets right.
FACTS = [
    ("The Space Needle is located in the city of", "The Space Needle", " Seattle"),
    ("The Colosseum is located in the city of", "The Colosseum", " Rome"),
    ("The Golden Gate Bridge is located in the city of", "The Golden Gate Bridge", " San"),
    ("Michael Jordan plays the sport of", "Michael Jordan", " basketball"),
    ("Lionel Messi plays the sport of", "Lionel Messi", " soccer"),
    ("The mother tongue of Vladimir Putin is", "Vladimir Putin", " Russian"),
    ("The Big Ben clock tower is in the city of", "The Big Ben clock tower", " London"),
    ("Barack Obama was born in the state of", "Barack Obama", " Hawaii"),
    ("Toyota is a car company headquartered in the country of", "Toyota", " Japan"),
    ("The Louvre museum is located in the city of", "The Louvre museum", " Paris"),
    ("Steve Jobs was the co-founder of the company", "Steve Jobs", " Apple"),
    ("The Taj Mahal is located in the country of", "The Taj Mahal", " India"),
    ("Serena Williams plays the sport of", "Serena Williams", " tennis"),
    ("The official language of Brazil is", "Brazil", " Portuguese"),
    ("Tom Brady plays the sport of", "Tom Brady", " football"),
    ("The Kremlin is located in the city of", "The Kremlin", " Moscow"),
    ("Mount Fuji is located in the country of", "Mount Fuji", " Japan"),
    ("The Sydney Opera House is located in the country of", "The Sydney Opera House", " Australia"),
]


@dataclass
class Fact:
    prompt: str
    tokens: torch.Tensor  # [1, pos]
    subject_positions: list[int]
    target: int
    p_target: float


def known_facts(model: GPT2, min_prob: float = 0.1) -> list[Fact]:
    out = []
    for prompt, subject, answer in FACTS:
        tokens = model.to_tokens(prompt)
        target = model.tokenizer.encode(answer, prepend_bos=False)[0]
        probs = model(tokens)[0, -1].softmax(-1)
        if probs.argmax().item() != target or probs[target] < min_prob:
            continue
        subj_ids = model.tokenizer.encode(
            (" " if prompt.index(subject) else "") + subject, prepend_bos=False
        )
        ids = tokens[0].tolist()
        starts = [i for i in range(len(ids)) if ids[i : i + len(subj_ids)] == subj_ids]
        if not starts:
            continue
        positions = list(range(starts[0], starts[0] + len(subj_ids)))
        out.append(Fact(prompt, tokens, positions, target, probs[target].item()))
    return out
