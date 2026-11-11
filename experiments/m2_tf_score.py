#!/usr/bin/env python3
"""Score the M2-TF dumps offline (bit-identity of shared attention layers).

Reads the two model dirs the teacher-forced probe already wrote and asserts,
per checkpoint: token_ids equal, and K/V rows at shared full-attention layers
BELOW the frozen prefix (3, 7) are bit-identical across engines. Rows are
stored f32 [kv_heads, seq, head_dim] by the dump hook, so identity is a plain
np.array_equal — the same bit pattern both engines wrote from bit-identical
frozen weights is the whole claim.

Run: /home/derek/.venv/bin/python m2_tf_score.py [dump_root]
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/home/derek/tensorfold-mom/src/tensorfold/experiments")
from s1_fit import load_model_dir

SHARED = [3, 7]
NESTED_DIR = "qwen36-nested-16l"
PARENT_DIR = "qwen36-35b-a3b-nvfp4"


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/derek/tensorfold-mom/experiments/results/m2-tf/kvdump"
    a = {d.ckpt: d for d in load_model_dir(os.path.join(root, NESTED_DIR))}
    b = {d.ckpt: d for d in load_model_dir(os.path.join(root, PARENT_DIR))}
    if not a or not b:
        print(f"missing dumps: nested={len(a)} parent={len(b)}")
        return 2
    common = sorted(set(a) & set(b))
    print(f"checkpoints: nested={sorted(a)} parent={sorted(b)} -> comparing {common}\n")

    failures = 0
    for k in common:
        da, db = a[k], b[k]
        if not np.array_equal(da.token_ids, db.token_ids):
            n = min(len(da.token_ids), len(db.token_ids))
            pos = next((i for i in range(n) if da.token_ids[i] != db.token_ids[i]), n)
            print(f"ckpt{k}: TOKEN MISMATCH at pos {pos} "
                  f"({len(da.token_ids)} vs {len(db.token_ids)} tok) "
                  f"nested {da.token_ids[max(0,pos-3):pos+4]} | parent {db.token_ids[max(0,pos-3):pos+4]}")
            failures += 1
            continue
        layers = sorted(set(da.layers) & set(db.layers) & set(SHARED))
        if not layers:
            print(f"ckpt{k}: no shared layers present in both (nested {da.layers} parent {db.layers})")
            failures += 1
            continue
        row_bad = 0
        for L in layers:
            for kind in ("K", "V"):
                ra, rb = da.kv.get((L, kind)), db.kv.get((L, kind))
                if ra is None or rb is None:
                    print(f"ckpt{k} L{L}{kind}: absent in one engine")
                    row_bad += 1
                    continue
                if ra.shape != rb.shape:
                    print(f"ckpt{k} L{L}{kind}: shape {ra.shape} vs {rb.shape}")
                    row_bad += 1
                    continue
                if not np.array_equal(ra, rb):
                    nd = int((ra != rb).sum())
                    md = float(np.abs(ra - rb).max())
                    print(f"ckpt{k} L{L}{kind}: {nd}/{ra.size} differ (max|d|={md:.4g})")
                    row_bad += 1
        tag = "IDENTICAL" if row_bad == 0 else f"{row_bad} ROW-IDENTITY FAILURES"
        print(f"ckpt{k}: {len(da.token_ids)} tok, layers {layers}, {tag}")
        failures += row_bad

    print(f"\nM2-TF score: {'PASS' if failures == 0 else f'FAIL ({failures})'} — "
          f"{len(common)} checkpoints, layers {SHARED}, K+V bit-identical required")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
