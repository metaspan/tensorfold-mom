#!/usr/bin/env python
"""D1 (pre-registered BACKLOG 2026-11-14): train-set accuracy on the student.

Score the student on corpus rows from the GATE MISS categories
(arithmetic / unit conversion / spelling-reverse / grammar). If it fails on
rows it TRAINED on, the limit is capacity/optimisation, not distribution —
which deprioritises ladder steps 5b/5c.

Two metrics per sampled row:
  - greedy_match: student greedy completion contains the corpus completion's
    final-answer span (last non-empty line) normalized
  - tf_loss: teacher-forced mean CE of the student on the corpus completion
    (the quantity training minimised) — aggregate, per category.

Usage:
  systemd-run --user --scope -p MemoryMax=60G python d1_trainset.py \
      --model /home/derek/models/tensorfold-mom/qwen36-nested-16l-t3 \
      --data results/distill/rollouts-10k-clean.jsonl
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_train import load_rows

CATS = {
    "arith": re.compile(r"\b(plus|minus|times|divided|multiplied|what is \d|\d\s*[-+*/]\s*\d)\b", re.I),
    "units": re.compile(r"\b(convert|kilomet|mile|celsius|fahrenheit|kilogram|pound|metre|meter)\b", re.I),
    "letters": re.compile(r"\b(reverse|spell|letters? in|upside)\b", re.I),
    "grammar": re.compile(r"\b(grammar|correct the|fix the)\b", re.I),
    "prime": re.compile(r"\bprime number\b", re.I),
}


def norm(s: str) -> str:
    return "".join(ch for ch in s.casefold() if ch.isalnum() or ch == " ").strip()


def final_span(comp: str) -> str:
    lines = [l for l in comp.strip().splitlines() if l.strip()]
    return lines[-1] if lines else comp.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = load_rows(args.data)

    buckets = {k: [] for k in CATS}
    for msgs, comp in rows:
        q = msgs[-1]["content"] if msgs else ""
        for k, rx in CATS.items():
            if rx.search(q):
                buckets[k].append((msgs, comp, q))
                break
    rng = random.Random(args.seed)
    picked, seen = [], set()
    for k, lst in buckets.items():
        rng.shuffle(lst)
        take = lst[: max(4, args.n // max(1, sum(1 for v in buckets.values() if v)))]
        for r in take:
            if id(r) not in seen:
                seen.add(id(r))
                picked.append((k, *r))
    picked = picked[: args.n]
    print({k: len(v) for k, v in buckets.items()}, flush=True)
    print(f"sampled {len(picked)} rows", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    stats = {}
    for i, (cat, msgs, comp, q) in enumerate(picked):
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)
        enc = tok(prompt + comp, return_tensors="pt", truncation=True,
                  max_length=512).to(model.device)
        p_enc = tok(prompt, return_tensors="pt", truncation=True,
                    max_length=512)
        p_len = p_enc["input_ids"].shape[1]
        with torch.no_grad():
            logits = model(**enc).logits[0]
            labels = enc["input_ids"][0].clone()
            labels[:p_len] = -100
            shift_l = logits[:-1].float()
            shift_y = labels[1:]
            mask = shift_y != -100
            ce = torch.nn.functional.cross_entropy(
                shift_l[mask], shift_y[mask], reduction="mean")
            out = model.generate(**p_enc.to(model.device),
                                 max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        got = tok.decode(out[0][p_len:], skip_special_tokens=True)
        match = norm(final_span(comp))[:60] in norm(got) or (
            norm(got) and norm(final_span(comp)) and
            norm(got) in norm(final_span(comp)))
        st = stats.setdefault(cat, {"n": 0, "match": 0, "ce": 0.0,
                                    "misses": []})
        st["n"] += 1
        st["match"] += int(match)
        st["ce"] += float(ce)
        if not match and len(st["misses"]) < 3:
            st["misses"].append({"q": q[:70], "gold_tail": final_span(comp)[:60],
                                 "got": got[:80]})
        if (i + 1) % 25 == 0:
            print(f"row {i+1}/{len(picked)}", flush=True)

    print(json.dumps({k: {"n": v["n"],
                          "greedy_match": round(v["match"] / max(1, v["n"]), 3),
                          "mean_tf_ce": round(v["ce"] / max(1, v["n"]), 3),
                          "sample_misses": v["misses"]}
                      for k, v in stats.items()}, indent=1))
    if args.out:
        json.dump(stats, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
