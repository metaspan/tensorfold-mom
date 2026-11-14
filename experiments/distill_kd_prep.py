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

# --- page-cache discipline BEFORE transformers/safetensors import ------------
# GB10 is unified-RAM: the 67 GB of mmap'd shard pages from the device_map
# load are charged to the cgroup ALONGSIDE the 67 GB of (pinned) GPU weights,
# and the cgroup OOM-killer fires near full weight materialization. Wrap
# safetensors readers to posix_fadvise(DONTNEED) each shard after it is fully
# read so clean file pages leave the cgroup immediately. transformers binds
# `safe_open` via from-import at its own import time, so this patch must be
# installed first.
import safetensors
import safetensors.torch

_orig_safe_open = safetensors.safe_open


def _fadvise_dontneed(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


class _FadviseSafeOpen:
    def __init__(self, filename, framework="pt", device="cpu", **kw):
        self._path = str(filename)
        self._cm = _orig_safe_open(filename, framework=framework,
                                   device=device, **kw)
        self._keys = list(self._cm.keys())
        self._read = set()
        try:
            self._fd = os.open(self._path, os.O_RDONLY)
        except OSError:
            self._fd = None

    def _evict_after(self, key):
        # shard files are read in weight_map order: once we hand out the last
        # tensor of a shard, all its pages are read-once garbage
        self._read.add(key)
        if self._fd is not None and set(self._keys) <= self._read:
            try:
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    def get_tensor(self, key):
        t = self._cm.get_tensor(key)
        self._evict_after(key)
        return t

    def get_slice(self, key):
        # lazy slice: eviction cannot track reads through it; pre-evicting is
        # unsafe, so just track (behavior identical to unpatched)
        return self._cm.get_slice(key)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        r = self._cm.__exit__(*a)
        if self._fd is not None:
            try:
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        return r

    def __getattr__(self, n):
        return getattr(self._cm, n)


safetensors.safe_open = _FadviseSafeOpen

_orig_load_file = safetensors.torch.load_file


def _load_file_fadvise(path, *a, **k):
    d = _orig_load_file(path, *a, **k)
    _fadvise_dontneed(str(path))
    return d


safetensors.torch.load_file = _load_file_fadvise
# -----------------------------------------------------------------------------

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
        args.teacher, dtype=torch.bfloat16, trust_remote_code=False,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        # GB10 unified-RAM: mmap'd shard pages (67 GB) are charged to the
        # cgroup alongside pinned GPU weights and OOM-kill the load at ~97%.
        # disable_mmap reads each shard fully into anon memory instead
        # (~1.7 GB transient); page cache never accumulates.
        disable_mmap=True)
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
            logits = teacher(input_ids=input_ids.to(dev)).logits[0]  # [n, V]
            # position t logits predict token t+1; keep n rows, trainer aligns
            k = min(args.topk, logits.shape[-1])
            vals, idxs = torch.topk(logits.float(), k, dim=-1)
            out.append({"idx": idxs.to(torch.int32).cpu(),
                        "val": vals.to(torch.bfloat16).cpu(), "len": n})
            if i == 0:
                print("first row ok:", idxs.shape, flush=True)
            if (i + 1) % 500 == 0:
                print(f"row {i+1}/{len(rows)}", flush=True)
    torch.save(out, args.out)
    print(f"wrote {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
