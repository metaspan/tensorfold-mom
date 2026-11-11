#!/usr/bin/env python3
"""Dump GDN (linear_attn) + full-attn tensor names/shapes from the parent header."""
import json
import struct

import glob
for p in sorted(glob.glob("/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4/model-*.safetensors")):
    f = open(p, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    if not any(k.startswith("model.language_model.layers.0.linear_attn.") for k in h):
        continue
    print(f"== shard {p.split('/')[-1]} ==")
    print("== layer 0 linear_attn ==")
    for k, v in sorted(h.items()):
        if k.startswith("model.language_model.layers.0.linear_attn."):
            print(f"  {k.split('.',3)[3]:35s} {v['shape']} {v['dtype']}")
    print("== layer 0 non-linear_attn (norms/router) ==")
    for k, v in sorted(h.items()):
        if k.startswith("model.language_model.layers.0.") and ".linear_attn." not in k and ".mlp." not in k:
            print(f"  {k.split('.',3)[3]:35s} {v['shape']} {v['dtype']}")
    print("== layer 3 self_attn ==")
    for k, v in sorted(h.items()):
        if k.startswith("model.language_model.layers.3.self_attn."):
            print(f"  {k.split('.',3)[3]:35s} {v['shape']} {v['dtype']}")
