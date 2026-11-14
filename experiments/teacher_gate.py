#!/usr/bin/env python
"""Scored teacher-gate: the 12-item coherence gate on the KD teacher itself.

Pre-registered calibration (BACKLOG 2026-11-14, review adjudication): the
dequantised bf16 teacher must score >= 11/12 greedy on gate_qa.json before
student results are interpreted. Below that, the loader fidelity or the
dequant path is suspect, not the student. Doubles as the perplexity-review
ask ("teacher top-1 agreement on the 12 gate items") and corrects the
two-day review's un-scored 'parent answers all 12' claim.

Judging is identical to distill_gate.gate_hf (same items, greedy, thinking
off, containment judging, template-leak checks) — only the load path
differs: build_teacher's peak-safe GPU-direct loader (from_pretrained OOMs
the 35B on this unified-RAM box).

Usage:
  systemd-run --user --scope -p MemoryMax=100G python teacher_gate.py \
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
    ap.add_argument("--qa", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gate_qa.json"))
    ap.add_argument("--pass-min", type=int, default=11,
                    help="calibration bar (pre-registered 11/12)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import torch

    qa = json.load(open(args.qa))
    if isinstance(qa, dict):
        qa = qa["items"]
    tok = AutoTokenizer.from_pretrained(STUDENT)

    teacher, dev = build_teacher(args.teacher)

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
            out = teacher.generate(**ids, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        got = tok.decode(out[0][n_in:], skip_special_tokens=True)
        ok = (any(norm(a) in norm(got) for a in item["a"])
              and not got.startswith("<error")
              and not any(t in got.casefold()
                          for t in ("<|im_start|", "<|im_end|", "<think>")))
        results.append({"id": item.get("id", i), "q": item["q"],
                        "reply": got[:160], "pass": bool(ok)})
        print(f"  [{i}] {'OK ' if ok else 'MISS'} {item['q']!r} -> "
              f"{got[:80]!r}", flush=True)

    correct = sum(r["pass"] for r in results)
    verdict = correct >= args.pass_min
    print(f"TEACHER-GATE {os.path.basename(args.teacher.rstrip('/'))}: "
          f"{correct}/12 (bar {args.pass_min}) -> "
          f"{'CALIBRATION OK' if verdict else 'CALIBRATION FAIL'}", flush=True)
    if args.out:
        json.dump({"teacher": args.teacher, "pass": correct, "total": len(qa),
                   "pass_min": args.pass_min, "results": results},
                  open(args.out, "w"), indent=1)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
