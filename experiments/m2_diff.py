#!/usr/bin/env python3
"""m2_diff.py — offline diff of M2 lane token streams from dump sidecars."""
import json
import sys

A = "/home/derek/tensorfold-mom/experiments/results/m2/kvdump/qwen36-nested-16l"
B = "/home/derek/tensorfold-mom/experiments/results/m2/kvdump/qwen36-35b-a3b-nvfp4"

for n in range(0, 9):
    try:
        ta = json.load(open(f"{A}/ckpt{n}.json"))["tokens"]
        tb = json.load(open(f"{B}/ckpt{n}.json"))["tokens"]
    except FileNotFoundError:
        continue
    if ta == tb:
        print(f"ckpt{n}: EQUAL ({len(ta)} tok)")
        continue
    print(f"ckpt{n}: nested {len(ta)} tok vs parent {len(tb)} tok")
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            lo = max(0, i - 4)
            print(f"  first diff pos {i}: nested {ta[lo:i+5]} | parent {tb[lo:i+5]}")
            break
    else:
        k = min(len(ta), len(tb))
        print(f"  equal prefix, length differs at {k}: nested tail {ta[k:k+10]} | parent tail {tb[k:k+10]}")
