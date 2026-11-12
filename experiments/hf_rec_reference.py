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
    return engrec_at(path, 11)

def engrec_at(path, reg):
    with open(path, "rb") as f:
        f.seek(reg * REGION)
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
    rec_captured = {}
    def run_fwd():
        from transformers.cache_utils import DynamicCache
        dc_init = DynamicCache.__init__
        orig_urs = DynamicCache.update_recurrent_state
        def patched_init(self, *a, **k):
            dc_init(self, *a, **k)
            def urs(*args, **kw):
                for x in list(args) + list(kw.values()):
                    if torch.is_tensor(x) and x.dim() == 4:
                        rec_captured.setdefault("states", []).append(x)
                return orig_urs(self, *args, **kw)
            self.update_recurrent_state = urs
        DynamicCache.__init__ = patched_init
        try:
            with torch.no_grad():
                return model(ids, use_cache=True)
        finally:
            DynamicCache.__init__ = dc_init
    out = run_fwd()
    h.remove()
    recs = rec_captured.get("states", [])
    print("recurrent updates:", [(tuple(x.shape), str(x.dtype)) for x in recs][:14])
    assert len(recs) == 12, f"expected 12 linear-layer state updates, got {len(recs)}"
    s = recs[-1][0].float().cpu().numpy()  # last linear layer = model layer 14
    print("hf state shape:", s.shape)
    np.save("/tmp/hf_l14_state.npy", s)

    # region sweep: compare EVERY captured linear state to its engine region.
    # Regions 0-10 are engine-agreeing ground truth: a sound reference pipeline must
    # correlate ~1 there; only then is the layer-14 comparison trustworthy.
    def corr(a, b):
        a = a.reshape(-1); b = b.reshape(-1)
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            return 1.0 if np.abs(a - b).max() < 1e-6 else 0.0
        return float(np.corrcoef(a, b)[0, 1])
    for name, path in (("parent", "/tmp/tf_mom_ctl_186_179_L40.bin"),
                       ("nested", "/tmp/tf_mom_ctl_186_179_L16.bin")):
        with open(path, "rb") as f:
            pass
        for orient in ("direct", "T"):
            line = []
            for reg in range(12):
                E = engrec_at(path, reg)
                H = recs[reg][0].float().cpu().numpy()
                if orient == "T":
                    H = np.swapaxes(H, -1, -2)
                cs = [corr(H[h], E[h]) for h in range(32)]
                line.append(f"r{reg}:{np.median(cs):+.2f}")
            print(f"{name} {orient}: " + " ".join(line))

if __name__ == "__main__":
    main()
