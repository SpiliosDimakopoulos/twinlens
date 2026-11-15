"""Build the external (held-out) prompt sets of plan task 2.1 (DECISIONS.md D9).

Sources, pinned to a revision:
  ioi      mib-bench/ioi (MIT), test split: Wang et al. (2023) templates as released by MIB
  sva      nyu-mll/blimp (CC BY 4.0): six subject-verb agreement paradigms, cut before the verb
  factual  evandez/relations (MIT): LRE factual relations (Hernandez et al., 2024), first template
  greater_than  generated here from the template of Hanna et al. (2023); GPT-2 only, because
           Gemma's tokenizer splits numbers into single digits

Every candidate has an answer token and a contrast token (a wrong answer of the same kind).
A candidate is kept when both are single tokens (the first token of the word, for sva and
factual), they differ, and the model's top-1 next token is the answer. Kept candidates are
shuffled with seed 0 and the first ``--per-family`` per family are written to
``experiments/prompts_external_<model>.json`` with their provenance. Runs the model, so it goes
to a server like the sweeps.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from common import ROOT
from huggingface_hub import hf_hub_download

IOI = ("mib-bench/ioi", "e5f3468f3af4c0883be35cd3bced8c711c95d286")
BLIMP = ("nyu-mll/blimp", "877fba0801ffb7cbd8c39c1ff314a46f053f6036")
LRE = ("evandez/relations", "1b9ec3cf2b8368e42bde7e80fcef384312a6ec07")
SVA_PARADIGMS = (
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
)
LRE_RELATIONS = (
    "city_in_country",
    "company_hq",
    "country_capital_city",
    "country_currency",
    "country_language",
    "food_from_country",
    "landmark_in_country",
    "person_native_language",
    "person_occupation",
    "person_plays_instrument",
    "person_plays_pro_sport",
    "product_by_company",
    "star_constellation",
)
SEED = 0


def _ioi() -> list[dict]:
    path = hf_hub_download(IOI[0], "test.json", repo_type="dataset", revision=IOI[1])
    text = Path(path).read_text(encoding="utf-8").strip()
    rows = json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines()]
    out = []
    for r in rows:
        ans = r["choices"][r["answerKey"]]
        con = r["choices"][1 - r["answerKey"]]
        out.append(
            {
                "prompt": r["prompt"],
                "answer": " " + ans,
                "contrast": " " + con,
                "source": f"{IOI[0]}@{IOI[1][:7]} test",
            }
        )
    return out


def _sva() -> list[dict]:
    out = []
    for p in SVA_PARADIGMS:
        f = hf_hub_download(
            BLIMP[0], f"{p}/train-00000-of-00001.parquet", repo_type="dataset", revision=BLIMP[1]
        )
        for r in pd.read_parquet(f).itertuples():
            if not r.one_prefix_method:
                continue
            good, bad = r.sentence_good.split(" "), r.sentence_bad.split(" ")
            if len(good) != len(bad):
                continue
            diff = [i for i, (a, b) in enumerate(zip(good, bad, strict=True)) if a != b]
            if len(diff) != 1 or diff[0] == 0:
                continue
            i = diff[0]
            out.append(
                {
                    "prompt": " ".join(good[:i]),
                    "answer": " " + good[i],
                    "contrast": " " + bad[i],
                    "source": f"{BLIMP[0]}@{BLIMP[1][:7]} {p}",
                }
            )
    return out


def _factual() -> list[dict]:
    out = []
    for rel in LRE_RELATIONS:
        url = f"https://raw.githubusercontent.com/{LRE[0]}/{LRE[1]}/data/factual/{rel}.json"
        with urllib.request.urlopen(url, timeout=60) as fh:
            d = json.load(fh)
        tmpl = d["prompt_templates"][0]
        objs = sorted({s["object"] for s in d["samples"]})
        rng = random.Random(f"{SEED}-{rel}")
        for s in d["samples"]:
            others = [o for o in objs if o != s["object"]]
            if not others:
                continue
            out.append(
                {
                    "prompt": tmpl.format(s["subject"]),
                    "answer": " " + s["object"],
                    "contrast": " " + rng.choice(others),
                    "source": f"{LRE[0]}@{LRE[1][:7]} {rel}",
                }
            )
    return out


def _greater_than() -> list[dict]:
    # Hanna et al. (2023): "The <noun> lasted from the year 17YY to the year 17"; the contrast
    # is the year YY - 1, which a correct model must rank below a later year.
    nouns = ("war", "dynasty", "expedition", "occupation", "reign", "conflict", "drought")
    out = []
    for noun in nouns:
        for yy in range(2, 99):
            out.append(
                {
                    "prompt": f"The {noun} lasted from the year 17{yy:02d} to the year 17",
                    "answer": None,
                    "contrast": f"{yy - 1:02d}",
                    "yy": yy,
                    "source": "generated, template of Hanna et al. (2023)",
                }
            )
    return out


def _first_token(model, s: str) -> int:
    return model.tokenizer.encode(s, prepend_bos=False)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gpt2", "gemma2"], required=True)
    ap.add_argument("--per-family", type=int, default=60)
    args = ap.parse_args()
    if args.model == "gpt2":
        from twinlens.core.gpt2 import GPT2

        model = GPT2.load("gpt2")
    else:
        from twinlens.core.gemma2 import Gemma2

        model = Gemma2.load()
    families = {"ioi": _ioi, "sva": _sva, "factual": _factual}
    if args.model == "gpt2":
        families["greater_than"] = _greater_than
    result: dict = {"model": args.model, "seed": SEED, "decisions": "D9", "families": {}}
    for fam, fn in families.items():
        cands = fn()
        random.Random(f"{SEED}-{fam}").shuffle(cands)
        kept, seen = [], set()
        for c in cands:
            if len(kept) >= args.per_family:
                break
            if c["prompt"] in seen:
                continue
            seen.add(c["prompt"])
            with torch.no_grad():
                logits = model.forward(model.to_tokens(c["prompt"]), last_only=True)[0, -1]
            top = int(logits.argmax())
            if fam == "greater_than":
                word = model.tokenizer.decode([top])
                ids = model.tokenizer.encode(c["contrast"], prepend_bos=False)
                if not (word.isdigit() and len(word) == 2 and int(word) > c["yy"]) or len(ids) != 1:
                    continue
                c["answer"], aid, cid = word, top, ids[0]
            else:
                a_ids = model.tokenizer.encode(c["answer"], prepend_bos=False)
                aid, cid = a_ids[0], _first_token(model, c["contrast"])
                if aid == cid or top != aid:
                    continue
            kept.append({**c, "answer_id": aid, "contrast_id": cid})
        result["families"][fam] = {
            "n_candidates": len(cands),
            "n_tested": len(seen),
            "prompts": kept,
        }
        print(
            f"{fam:13s} kept {len(kept)} of {len(seen)} tested ({len(cands)} candidates)",
            flush=True,
        )
    path = ROOT / "experiments" / f"prompts_external_{args.model}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
