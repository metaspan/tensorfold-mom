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
from s1_fit import load_model_dir

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
    # Engine ground truth tokens: the M2-TF nested kvdump's last checkpoint is the
    # same teacher-forced 8-turn replay -- the exact 186-token hop prompt both engines saw.
    ck = load_model_dir("/home/derek/tensorfold-mom/experiments/results/m2-tf/kvdump/qwen36-nested-16l")
    last = sorted(ck, key=lambda c: c.ckpt)[-1] if not isinstance(ck, list) else ck[-1]
    tok_ids = list(getattr(last, "token_ids", None) if not isinstance(last, dict) else last["token_ids"])
    n = len(tok_ids)
    print(f"engine kept-state tokens: {n}, at={getattr(last, 'at', '?')}")
    assert n == 179 and getattr(last, "at", None) == 179, "dump is not the at=179 kept state"
    tok_ids = list(tok_ids)
    ids = torch.tensor([tok_ids[:179]], dtype=torch.long)
    # informational: HF template rendering of the same hist
    try:
        tok = AutoTokenizer.from_pretrained(STUDENT)
        hist = [{"role": "system", "content": SYSTEM}]
        tu = (TRIVIAL_TURNS + WORK_TURNS)[:TURNS]
        rp = json.load(open(REPLIES))
        for u, a in zip(tu, rp):
            hist.append({"role": "user", "content": u})
            hist.append({"role": "assistant", "content": a})
        hist.append({"role": "user", "content": tu[-1]})
        enc = tok.apply_chat_template(hist, add_generation_prompt=True,
                                      enable_thinking=False, return_dict=True)
        hf_ids = enc["input_ids"]
        print(f"HF template ids: {len(hf_ids)}; engine prefix match: "
              f"{sum(a == b for a, b in zip(hf_ids[:186], tok_ids))}/186")
    except Exception as e:
        print("HF template info skipped:", e)
    ids = ids.to("cuda")

    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.to("cuda")
    captured = {}
    def hook(mod, inp, out):
        captured["out"] = out
    tgt = None
    for nm, mod in model.named_modules():
        if ".layers.14." in nm and any(k in nm for k in ("linear_attn", "gated_delta", "gdn")):
            tgt = (nm, mod)
            break
    if tgt is None:  # fall back: the module whose forward returns a [B,H,128,128] tensor
        for nm, mod in model.named_modules():
            if ".layers.14." in nm and type(mod).__name__.lower().find("attn") >= 0:
                tgt = (nm, mod)
                break
    assert tgt is not None, "layer-14 GDN module not found"
    print("hooking", tgt[0], type(tgt[1]).__name__)
    h = tgt[1].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    h.remove()
    o = captured.get("out")
    cands = []
    if isinstance(o, torch.Tensor):
        cands = [o]
    elif isinstance(o, (tuple, list)):
        cands = [x for x in o if torch.is_tensor(x)]
    states = [x for x in cands if x.dim() == 4 and 128 in x.shape[-2:]]
    print("captured tensors:", [(tuple(x.shape), str(x.dtype)) for x in cands])
    assert states, f"no [B,H,128,128] recurrent state captured; got {[tuple(x.shape) for x in cands]}"
    s = states[0][0].float().cpu().numpy()
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
