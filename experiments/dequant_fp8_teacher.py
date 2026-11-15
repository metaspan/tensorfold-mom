#!/usr/bin/env python
"""Dequantize the official Qwen3.6-35B-A3B-FP8 snapshot to a plain-bf16
directory for HF teacher forwards when the finegrained-fp8 triton kernel is
unusable on this stack ("Unknown recipe" from triton_kernels layout utils).

Block-fp8 semantics (transformers finegrained_fp8): weight [out,in] e4m3,
weight_scale_inv [ceil(out/128), ceil(in/128)] bf16, one scale per 128x128
block. Output config.json has quantization_config removed.

Usage: python dequant_fp8_teacher.py --src <snapshot dir> --dst <dir>
"""
import argparse
import json
import math
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

BLOCK = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    scales = {}
    for fn in sorted(os.listdir(args.src)):
        if not fn.endswith(".safetensors"):
            continue
        tensors = load_file(os.path.join(args.src, fn))
        for k, v in tensors.items():
            if k.endswith("weight_scale_inv") or k.endswith("weight_scale"):
                scales[k] = v
        out = {}
        for k, v in tensors.items():
            if v.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                out[k] = v
                continue
            base = k[: -len(".weight")]
            sk = base + ".weight_scale_inv"
            if sk not in scales:
                sk = base + ".weight_scale"
            if sk not in scales:
                raise KeyError(f"no scale for {k}")
            sc = scales[sk].float()
            out_ = v.float()
            r, c = out_.shape
            big = sc.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
            out[k] = (out_ * big[:r, :c]).to(torch.bfloat16)
        # drop scale tensors: fully consumed above
        out = {k: v for k, v in out.items() if not (
            k.endswith("weight_scale_inv") or k.endswith("weight_scale"))}
        save_file(out, os.path.join(args.dst, fn),
                  metadata={"format": "pt"})
        print(f"{fn}: {len(out)} tensors dequantized", flush=True)

    for fn in os.listdir(args.src):
        if fn.endswith(".safetensors") or fn == "config.json":
            continue
        shutil.copy2(os.path.join(args.src, fn), os.path.join(args.dst, fn))
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    cfg.pop("quantization_config", None)
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        tc.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)
    print("config.json written (quantization_config removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
