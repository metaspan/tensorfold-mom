#!/usr/bin/env python
"""D2 (pre-registered BACKLOG 2026-11-14): early-exit baseline at layer 10.

Truncate the teacher (identical weights to the student's frozen prefix for
layers 0-9) to 10 layers and keep the heavy model's final norm + lm_head.
Score the 12 gate items greedily. This bounds how much of the ANSWER
information is present at the split point: items the truncated model gets
right cost the tail nothing extra; items it misses need tail computation.

Uses build_teacher (peak-safe GPU-direct loader). Judging identical to
teacher_gate.py (norm-containment, template-leak check).

Usage:
  systemd-run --user --scope -p MemoryMax=100G python d2_early_exit.py \
      --teacher /home/derek/models/tensorfold-mom/qwen36-35b-a3b-fp8-deq
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_kd_prep import build_teacher
from distill_train import STUDENT


def norm(s: str) -> str:
    return "".join(ch for ch in s.casefold() if ch.isalnum() or ch == " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--qa", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gate_qa.json"))
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer

    qa = json.load(open(args.qa))
    if isinstance(qa, dict):
        qa = qa["items"]
    tok = AutoTokenizer.from_pretrained(STUDENT)

    teacher, dev = build_teacher(args.teacher)
    keep = list(teacher.model.layers[: args.layers])
    teacher.model.layers = nn.ModuleList(keep)
    teacher.config.num_hidden_layers = args.layers
    print(f"early-exit at layer {args.layers}: "
          f"{sum(p.numel() for p in teacher.parameters())/1e9:.1f}B params left",
          flush=True)

    results = []
    for i, item in enumerate(qa):
        msgs = [{"role": "user", "content": item["q"]}]
        try:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
        ids = tok(prompt, return_tensors="pt").to(dev)
        n_in = ids["input_ids"].shape[1]
        with torch.no_grad():
            out = teacher.generate(**ids, max_new_tokens=args.max_new,
                                   do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        got = tok.decode(out[0][n_in:], skip_special_tokens=True)
        ok = (any(norm(a) in norm(got) for a in item["a"])
              and not any(t in got.casefold()
                          for t in ("<|im_start|", "<|im_end|", "<think>")))
        results.append({"id": item.get("id", i), "q": item["q"],
                        "reply": got[:160], "pass": bool(ok)})
        print(f"  [{i}] {'OK ' if ok else 'MISS'} {item['q']!r} -> "
              f"{got[:80]!r}", flush=True)

    correct = sum(r["pass"] for r in results)
    print(f"D2 EARLY-EXIT@{args.layers}: {correct}/12 (bounds answer info at "
          f"the split point)", flush=True)
    if args.out:
        json.dump({"layers": args.layers, "pass": correct, "total": len(qa),
                   "results": results}, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
