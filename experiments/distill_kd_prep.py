#!/usr/bin/env python
"""Offline teacher top-k logits for distillation (logit-KD prep).

For each corpus row (prompt, completion), run ONE teacher forward over the
same tokenized sequence distill_train.py uses and save the teacher's top-K
next-token logits + token ids per position. The trainer then adds a KL term
against these — no engine logprobs needed, and teacher/student never share
memory (each pass fits RAM alone).

Teacher: official Qwen/Qwen3.6-35B-A3B-FP8 in the HF cache — the same base
the on-disk NVFP4 collector teacher was quantized from (config identity
verified: qwen3_5_moe_text, hidden 2048, 40 layers, 256 experts).

Tokenization is deliberately duplicated VERBATIM from distill_train.ce_on so
positions align 1:1 (the trainer slices with its own p_len; we save full T).

Output: kd-<corpus-stem>.pt — dict rows: {"idx": LongTensor[n, k],
"val": BFloat16Tensor[n, k], "len": int} in a torch.save list, n = T_i.

Usage:
  systemd-run --user --scope -p MemoryMax=100G python distill_kd_prep.py \
      --data results/distill/rollouts-10k-clean.jsonl \
      --out results/distill/kd-10k.pt --topk 8
"""
import argparse
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_train import STUDENT, load_rows  # same tokenizer + row loader

TEACHER_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/*")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.teacher is None:
        args.teacher = sorted(glob.glob(TEACHER_GLOB))[-1]

    tok = AutoTokenizer.from_pretrained(STUDENT)
    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    print(f"teacher: {args.teacher}\nrows: {len(rows)}", flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=False)
    teacher.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    out = []
    with torch.no_grad():
        for i, (msgs, comp) in enumerate(rows):
            # verbatim from distill_train.ce_on
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
            full = prompt + comp + tok.eos_token
            enc = tok(full, return_tensors="pt", truncation=True,
                      max_length=args.max_len)
            input_ids = enc["input_ids"]
            n = input_ids.shape[1]
            teacher.to(dev)
            logits = teacher(input_ids=input_ids.to(dev)).logits[0]  # [n, V]
            # position t logits predict token t+1; keep n rows, trainer aligns
            k = min(args.topk, logits.shape[-1])
            vals, idxs = torch.topk(logits.float(), k, dim=-1)
            out.append({"idx": idxs.to(torch.int32).cpu(),
                        "val": vals.to(torch.bfloat16).cpu(), "len": n})
            if i == 0:
                print("first row ok:", idxs.shape, flush=True)
            if (i + 1) % 200 == 0:
                teacher.to("cpu")
                torch.cuda.empty_cache()
                print(f"row {i+1}/{len(rows)}", flush=True)
                teacher.to(dev)
        teacher.to("cpu")
    torch.save(out, args.out)
    print(f"wrote {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
