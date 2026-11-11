#!/usr/bin/env python3
"""CPU-cheap smoke for the distillation window (no weight loading, no forward).

Claims tested (the ones that would burn a GPU window):
  A. HF 5.8.1 resolves config -> causal-LM class (meta-device init, instant).
  B. The checkpoint's safetensors index names are what distill_train's freeze
     rule expects: no layers.<0..9> name can land in the trainable set, and
     the final-norm rule matches the ACTUAL param name (catches e.g. a
     'language_model.' prefix silently freezing the final norm).
Run 1 lesson: loading the 10 GiB MoE weights on CPU exceeds any sane timeout
(killed at 900 s) -- test names, not bytes; the GPU window itself proves load.
"""
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM

STUDENT = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"
FROZEN = 10
TAIL = range(FROZEN, 16)

cfg = AutoConfig.from_pretrained(STUDENT)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16,
                                             trust_remote_code=False)
print(f"A OK: {type(model).__name__}, meta params: {len(list(model.named_parameters()))}")

# checkpoint-side names (what from_pretrained will map onto the model).
# The nested checkpoint is a SINGLE model.safetensors (M0 wrote one shard; no
# index json) -- read names straight from the header's tensor map.
import struct
from safetensors import safe_open
with safe_open(f"{STUDENT}/model.safetensors", framework="pt") as f:
    ckpt_names = list(f.keys())
print(f"B: checkpoint tensors: {len(ckpt_names)}")

# distill_train's freeze rule, verbatim semantics
def trainable_rule(name: str) -> bool:
    keep = any(f"layers.{i}." in name for i in TAIL)
    return keep or "lm_head" in name or ("norm" in name and "layers" not in name)

leak = [n for n in ckpt_names if trainable_rule(n)
        and any(f"layers.{i}." in n for i in range(FROZEN))]
assert not leak, f"FROZEN-LAYER LEAK: {leak[:5]}"

tail_norm = [n for n in ckpt_names if n.endswith("norm.weight") and "layers." not in n]
assert any(trainable_rule(n) for n in tail_norm), f"final norm not trainable: {tail_norm[:5]}"
heads = [n for n in ckpt_names if "lm_head" in n]
assert heads and all(trainable_rule(n) for n in heads), f"lm_head issue: {heads[:5]}"
tail_missing = [n for n in ckpt_names if any(f"layers.{i}." in n for i in TAIL)
                and not trainable_rule(n)]
assert not tail_missing, f"tail tensors not trainable: {tail_missing[:5]}"
# mtp.*: NOT model params (HF drops them -- verified below), so the freeze
# loop can never train them. CONSEQUENCE (recorded, G2/G4): save_pretrained
# output loses mtp.* -> quality-gate serving is drafter-less (M1 proved that
# serves fine); deployment must re-merge mtp.* from the M0 checkpoint.
mtp_ckpt = [n for n in ckpt_names if n.startswith("mtp.")]
model_set = {n for n, _ in model.named_parameters()}
mtp_in_model = [n for n in mtp_ckpt if n in model_set]
assert not mtp_in_model, f"mtp unexpectedly loaded as params: {mtp_in_model[:5]}"
print(f"mtp.* tensors: {len(mtp_ckpt)} (dropped by HF -- re-merge at deploy)")

missing = [n for n in ckpt_names if n not in model_set and not n.startswith("mtp.")]
assert not missing, f"checkpoint tensors with no model param: {missing[:5]}"
print("SMOKE-PASS")
