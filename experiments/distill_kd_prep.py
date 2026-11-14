#!/usr/bin/env python
"""Offline teacher top-k logits for distillation (logit-KD prep).

For each corpus row (prompt, completion), run ONE teacher forward over the
same tokenized sequence distill_train.py uses and save the teacher's top-K
next-token logits + token ids per position. The trainer then adds a KL term
against these — no engine logprobs needed, and teacher/student never share
memory (each pass fits RAM alone).

Teacher: official Qwen/Qwen3.6-35B-A3B-FP8 in the HF cache — the same base
the on-disk NVFP4 collector teacher was quantized from (config identity
verified: qwen3_5_moe_text, hidden 2048, 40 layers, 256 experts). The HF
finegrained-fp8 triton path fails on this stack, so the default is the bf16
dequantized copy of the identical snapshot (dequant_fp8_teacher.py).

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


def build_teacher(teacher_dir: str):
    """Peak-memory-safe load for GB10 unified-RAM (121 GB host == GPU).
    from_pretrained's MoE custom weight conversion holds the unfused
    checkpoint (67 GB) AND the fused destination simultaneously — sampler
    showed anon 104 GB before the system OOM killer fired. Bypass it:
      1) instantiate from config directly on the target device (a CPU-side
         model would sit at the 67 GB high-water mark even after layer
         .to(cuda) — torch's CPU caching allocator does not return freed
         blocks to the OS; measured rss 51 GB mid-load -> system OOM)
      2) copy checkpoint tensors in per-shard (1.6 GB transient; fadvise
         wrapper above keeps page cache from accumulating)
      3) per-expert checkpoint tensors slice-copy into the fused GPU params
         (gate_up_proj rows [0:I]=gate,[I:2I]=up; down_proj direct)
    Combined peak ~70 GB. Fully under our control: no HF loader transients.
    Shared with teacher_gate.py."""
    from safetensors.torch import load_file as _st_load_file
    from transformers import AutoConfig
    import re as _re
    import gc

    cfg = AutoConfig.from_pretrained(teacher_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(dev):
        teacher = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    teacher.eval()

    sd_model = dict(teacher.named_parameters())
    # Fused-expert targets: transformers stores experts as
    #   <...>layers.L.mlp.experts.gate_up_proj [E, 2I, H]  (rows [0:I]=gate,
    #      [I:2I]=up — model chunks(2,-1) into gate,up)
    #   <...>layers.L.mlp.experts.down_proj    [E, H, I]
    # while the checkpoint stores per-expert tensors. Slice-copy each expert
    # straight into the (GPU-resident) fused parameter: ~2 MB transient.
    expert_params = {}  # L -> {"gate_up": Param, "down": Param}
    for name, prm in sd_model.items():
        m = _re.search(r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",
                       name)
        if m:
            L = int(m.group(1))
            expert_params.setdefault(L, {})
            expert_params[L]["gate_up" if m.group(2) == "gate_up_proj"
                            else "down"] = prm
    exp_key = _re.compile(
        r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
    ckpt_dir = teacher_dir
    shards = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors"))
    loaded, missing = set(), []
    with torch.no_grad():
        for fn in shards:
            sd = _st_load_file(os.path.join(ckpt_dir, fn))
            for k, v in sd.items():
                m = exp_key.search(k)
                if m and expert_params:
                    L, E, which = int(m.group(1)), int(m.group(2)), m.group(3)
                    ep = expert_params.get(L)
                    if ep is None:
                        missing.append(k)
                        continue
                    if which in ("gate", "up"):
                        I = v.shape[0]
                        tgt = ep["gate_up"].data[E]
                        off = 0 if which == "gate" else I
                        tgt[off:off + I].copy_(v.to(tgt.device))
                    else:
                        ep["down"].data[E].copy_(v.to(ep["down"].device))
                    loaded.add(k)
                    continue
                tgt = sd_model.get(k)
                if tgt is None:
                    # tolerate common prefix drift between ckpt and model
                    for cand in (k.replace("model.language_model.", "model."),
                                 "model." + k.split("model.", 1)[-1]
                                 if "model." in k else k):
                        tgt = sd_model.get(cand)
                        if tgt is not None:
                            break
                if tgt is None:
                    missing.append(k)
                    continue
                if tgt.shape != v.shape:
                    raise RuntimeError(f"shape mismatch {k}: ckpt {v.shape} "
                                       f"vs model {tuple(tgt.shape)}")
                tgt.data.copy_(v.to(tgt.device), non_blocking=True)
                loaded.add(k)
            del sd
            gc.collect()
    print(f"copied {len(loaded)} tensors; unmapped: {len(missing)}",
          flush=True)
    # MTP head tensors (no MTP module in base class) and vision-tower tensors
    # (text-only teacher forward) have no home and are unused — skip them;
    # anything else unmapped is fatal.
    def _skip(k):
        return k.startswith("mtp.") or k.startswith("model.visual.") or \
            k.startswith("visual.")
    skipped = [k for k in missing if _skip(k)]
    non_mtp = [k for k in missing if not _skip(k)]
    print(f"skipped mtp.*/visual tensors: {len(skipped)}", flush=True)
    if non_mtp:
        print("first unmapped keys:", non_mtp[:8], flush=True)
        raise RuntimeError("unmapped checkpoint keys — extend mapping")
    if dev == "cuda":
        for b in teacher.buffers():
            if b.device.type == "cpu" and b.numel() < 4_000_000:
                b.data = b.data.to("cuda")
    teacher.eval()
    return teacher, dev


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

    teacher, dev = build_teacher(args.teacher)

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
