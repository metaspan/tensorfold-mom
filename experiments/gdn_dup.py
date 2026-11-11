#!/usr/bin/env python3
"""GDN tail weight-tying export: train at 16 value-heads, export 32 by
pair duplication (docs/13 part 2).

Semantics (from the real parent checkpoint, gdn_header_probe.py):
  per-v-head tensors (16 -> 32):   A_log[32], dt_bias[32],
                                  in_proj_a/b.weight[32,2048]
  head-block tensors (rows x2):    in_proj_z.weight[4096,2048] (32x128),
                                  q-side of in_proj_qkv.weight[8192,2048],
                                  conv1d.weight[8192,1,4] (channels = qkv rows)
  column mirror of z:              out_proj.weight[2048,4096]
  UNCHANGED:                       norm.weight[128] (head-dim), k/v side of
                                  qkv + conv (16 heads, GQA 32/16 preserved)

GDN heads never mix channels (depthwise conv, per-head recurrence), so a
BLOCK layout — new head h pairs with h+16, identical rows — is consistent as
long as every per-head tensor uses the same convention. The tool ENFORCES
that: after duplication it asserts block halves are byte-equal and the
resulting shapes match the parent's layer-0 shapes exactly (hard-coded from
the header probe; a mismatch means the design drifted from the checkpoint).

fp8 tensors (in_proj_qkv/z/out = F8_E4M3 + scalar weight_scale/input_scale):
row/column duplication is pure byte copies; the scalar scales pass through.

Selftest: --selftest builds a synthetic 16-head layer, duplicates, and checks
shapes + pair equality + de-dup round-trip byte-for-byte.
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

# Expected exported (32-head) shapes, verbatim from gdn_header_probe on the
# parent checkpoint. Key = suffix under layers.<i>.linear_attn.
PARENT_SHAPES = {
    "A_log": [32],
    "dt_bias": [32],
    "in_proj_a.weight": [32, 2048],
    "in_proj_b.weight": [32, 2048],
    "in_proj_qkv.weight": [8192, 2048],
    "in_proj_qkv.weight_scale": [],
    "in_proj_qkv.input_scale": [],
    "in_proj_z.weight": [4096, 2048],
    "in_proj_z.weight_scale": [],
    "in_proj_z.input_scale": [],
    "out_proj.weight": [2048, 4096],
    "out_proj.weight_scale": [],
    "out_proj.input_scale": [],
    "conv1d.weight": [8192, 1, 4],
    "norm.weight": [128],
}
NP_DT = {"BF16": np.uint16, "F8_E4M3": np.uint8, "F32": np.float32}  # raw bytes


def dup_heads(arr: np.ndarray) -> np.ndarray:
    """[16, ...] -> [32, ...]: block layout, second half equals first."""
    if arr.shape[0] != 16:
        raise ValueError(f"expected 16 heads, got {arr.shape}")
    return np.concatenate([arr, arr], axis=0)


def qkv_q_side_dup(arr: np.ndarray, head_dim: int = 128, kv_heads: int = 16) -> np.ndarray:
    """16-head qkv rows [q:16*hd | k:16*hd | v:16*hd] -> [q:32*hd | k | v]."""
    qh = arr.shape[0] // (3 * head_dim) * head_dim if arr.shape[0] % (3 * head_dim) == 0 else None
    if arr.shape[0] != 3 * 16 * head_dim:
        raise ValueError(f"unexpected 16-head qkv rows {arr.shape[0]} (want {3*16*head_dim})")
    q = arr[: 16 * head_dim]
    kv = arr[16 * head_dim:]
    return np.concatenate([q, q, kv], axis=0)


def expand(gdn: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """16-head layer dict -> 32-head dict. Keys are suffixes under linear_attn."""
    out: dict[str, np.ndarray] = {}
    for name, a in gdn.items():
        if name in ("A_log", "dt_bias", "in_proj_a.weight", "in_proj_b.weight"):
            out[name] = dup_heads(a)
        elif name == "in_proj_z.weight":
            out[name] = dup_heads(a)                     # rows = 16*128 -> 32*128
        elif name == "in_proj_qkv.weight":
            out[name] = qkv_q_side_dup(a)
        elif name == "conv1d.weight":
            if a.shape[0] != 3 * 16 * 128:
                raise ValueError(f"conv1d channels {a.shape[0]} != 6144")
            q = a[: 16 * 128]
            kv = a[16 * 128:]
            out[name] = np.concatenate([q, q, kv], axis=0)
        elif name == "out_proj.weight":
            if a.shape[1] != 16 * 128:
                raise ValueError(f"out_proj cols {a.shape[1]} != 2048")
            out[name] = np.concatenate([a, a], axis=1)   # columns mirror z heads
        elif name.endswith("_scale"):
            out[name] = a.copy()                          # scalars pass through
        elif name == "norm.weight":
            out[name] = a.copy()                          # head-dim norm, unchanged
        else:
            raise ValueError(f"unknown gdn tensor {name}")
    return out


def assert_shapes(gdn32: dict[str, np.ndarray], where: str = "") -> None:
    for name, want in PARENT_SHAPES.items():
        got = list(gdn32[name].shape) if gdn32[name].ndim or want else []
        if got != want:
            raise AssertionError(f"{where}{name}: exported {got}, parent has {want}")
    # block-pair identity (the whole upcycling claim)
    assert np.array_equal(gdn32["A_log"][:16], gdn32["A_log"][16:])
    assert np.array_equal(gdn32["in_proj_z.weight"][:2048], gdn32["in_proj_z.weight"][2048:])
    assert np.array_equal(gdn32["out_proj.weight"][:, :2048], gdn32["out_proj.weight"][:, 2048:])
    assert np.array_equal(gdn32["in_proj_qkv.weight"][:2048], gdn32["in_proj_qkv.weight"][2048:4096])
    assert np.array_equal(gdn32["conv1d.weight"][:2048], gdn32["conv1d.weight"][2048:4096])


def selftest() -> int:
    rng = np.random.default_rng(7)
    g16 = {
        "A_log": rng.integers(0, 255, (16,)).astype(np.uint8),
        "dt_bias": rng.integers(0, 255, (16,)).astype(np.uint8),
        "in_proj_a.weight": rng.integers(0, 255, (16, 2048), dtype=np.uint8),
        "in_proj_b.weight": rng.integers(0, 255, (16, 2048), dtype=np.uint8),
        "in_proj_qkv.weight": rng.integers(0, 255, (6144, 2048), dtype=np.uint8),
        "in_proj_qkv.weight_scale": np.array(0.01, np.float32),
        "in_proj_qkv.input_scale": np.array(1.0, np.float32),
        "in_proj_z.weight": rng.integers(0, 255, (2048, 2048), dtype=np.uint8),
        "in_proj_z.weight_scale": np.array(0.02, np.float32),
        "in_proj_z.input_scale": np.array(1.0, np.float32),
        "out_proj.weight": rng.integers(0, 255, (2048, 2048), dtype=np.uint8),
        "out_proj.weight_scale": np.array(0.03, np.float32),
        "out_proj.input_scale": np.array(1.0, np.float32),
        "conv1d.weight": rng.integers(0, 255, (6144, 1, 4), dtype=np.uint8),
        "norm.weight": rng.integers(0, 255, (128,), dtype=np.uint16),
    }
    # raw-byte dtypes; shape is what the gate checks
    for k in ("norm.weight",):
        g16[k] = g16[k].astype(g16["A_log"].dtype)
    g32 = expand(g16)
    assert_shapes(g32, "selftest: ")
    # de-dup round trip: collapse second half away -> original bytes
    for name in ("A_log", "dt_bias", "in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight"):
        assert np.array_equal(g32[name][:16 if g32[name].ndim == 1 else 2048], g16[name])
    assert np.array_equal(g32["out_proj.weight"][:, :2048], g16["out_proj.weight"])
    q = g32["in_proj_qkv.weight"]
    assert np.array_equal(q[:2048], g16["in_proj_qkv.weight"][:2048])
    assert np.array_equal(q[4096:], g16["in_proj_qkv.weight"][2048:])
    print("selftest: OK (dup + shapes + pair identity + de-dup round trip)")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
