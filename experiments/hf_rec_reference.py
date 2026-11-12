#!/usr/bin/env python3
"""Third reference for the region-11 mystery: forward the first 179 tokens of the
M3 hop prompt through the bf16 nested checkpoint in HF eager and read the GDN
recurrent state of the 12th linear layer (model layer 14). Compare, per v-head,
against the two engine control files:
  /tmp/tf_mom_ctl_186_179_L40.bin  (parent engine's chunked save)
  /tmp/tf_mom_ctl_186_179_L16.bin  (nested engine's own cold save)
The engine values at heads 23-31 disagree between the two files
(parent ~1e-3, nested ~O(1)); eager HF says which one is arithmetically right.
"""
import sys, json
sys.path.insert(0, "/home/derek/tensorfold-mom/src/tensorfold/experiments")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from b5_cache_probe import TRIVIAL_TURNS, WORK_TURNS

STUDENT = "/home/derek/models/tensorfold-mom/qwen36-nested-16l-bf16"
REPLIES = "/home/derek/tensorfold-mom/experiments/results/m2-tf/parent_replies.json"
SYSTEM = "You are a concise assistant."
TURNS = 8
REGION = 25755648 // 12
RECB = 32 * 128 * 128 * 4
R11 = 11 * REGION

def engrec(path):
    with open(path, "rb") as f:
        f.seek(R11)
        b = f.read(RECB)
    return np.frombuffer(b, dtype=np.float32).reshape(32, 128, 128)

def main():
    tok = AutoTokenizer.from_pretrained(STUDENT)
    turns = (TRIVIAL_TURNS + WORK_TURNS)[:TURNS]
    replies = json.load(open(REPLIES))
    hist = [{"role": "system", "content": SYSTEM}]
    for u, a in zip(turns, replies):
        hist.append({"role": "user", "content": u})
        hist.append({"role": "assistant", "content": a})
    hist.append({"role": "user", "content": turns[-1]})
    ids = tok.apply_chat_template(hist, add_generation_prompt=True,
                                  enable_thinking=False, return_tensors="pt")
    n = ids.shape[1]
    print(f"prompt tokens: {n} (engine saw 186)")
    assert n == 186, f"tokenization mismatch: {n} != 186"
    ids = ids[:, :179].to("cuda")

    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = out.past_key_values
    states = []
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for ly in layers:
            rs = getattr(ly, "recurrent_state", None)
            if rs is not None:
                states.append(rs)
    else:
        rsd = getattr(cache, "recurrent_states", None) or getattr(cache, "slab_states", None)
        assert rsd is not None, f"unknown cache layout {type(cache)}"
        for e in rsd:
            if e is not None:
                states.append(e)
    print(f"linear states: {len(states)} (expect 12)")
    assert len(states) == 12
    s = states[11][0].float().cpu().numpy()  # [HV, ?, ?]
    print("hf state shape:", s.shape)

    P = engrec("/tmp/tf_mom_ctl_186_179_L40.bin")
    N = engrec("/tmp/tf_mom_ctl_186_179_L16.bin")
    def agree(a, b):
        if a.shape != b.shape:
            return -1.0
        num = np.abs(a - b).max()
        den = max(np.abs(a).max(), np.abs(b).max(), 1e-9)
        return float(num / den)
    for name, E in (("parent", P), ("nested", N)):
        for orient, H in (("direct", s), ("T", np.swapaxes(s, -1, -2))):
            # per-head worst relative error
            errs = [agree(H[h], E[h]) for h in range(32)]
            grp_low = max(errs[:23]); grp_hi = max(errs[23:])
            print(f"{name} {orient}: max-rel heads 0-22 {grp_low:.4f}, heads 23-31 {grp_hi:.4f}, "
                  f"all-heads {max(errs):.4f}")

if __name__ == "__main__":
    main()
