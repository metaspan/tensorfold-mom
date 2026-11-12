#!/usr/bin/env python3
"""Student-only distillation v1: tail-only SFT on heavy rollouts (docs/13 part 3).

Builds the trainable student from the nested checkpoint
(qwen36-nested-16l): layers < frozen_prefix are FROZEN (requires_grad=False,
byte-identity to the parent is the shared-cache claim — never touch them),
tail layers + final norm + lm_head train. Quantized tensors (F8_E4M3 /
NVFP4 storage) are dequantized to bf16 on load for training; the frozen
prefix in the EXPORT file is always re-copied from the parent by
nested_extract, so only the tail needs a round trip.

Loss: token CE on the completion span only (prompt/labels=-100 masking),
chat-templated with the SAME template as the checkpoint (chat_template.jinja)
— thinking blocks in completions are trained, which is what makes lane-hops
familiar (spec data mix).

Run (GPU window, production down):
  /home/derek/.venv/bin/python distill_train.py \
      --data results/distill/rollouts.jsonl --out-dir /home/derek/models/tensorfold-mom/qwen36-nested-16l-t1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT = "/home/derek/models/tensorfold-mom/qwen36-nested-16l-bf16"  # materialized (nested_materialize.py): names fused, NVFP4 dequantized
FROZEN_LAYERS = 10  # layers 0..9 frozen (matches nested_extract default)


def load_rows(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        r = json.loads(line)
        if not r.get("completion"):
            continue
        msgs = []
        if r.get("history_completion"):
            msgs = [{"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["history_completion"]}]
            # hop rows store the follow-up prompt inside completion flow only;
            # for v1 treat the continuation target against last user turn:
            msgs.append({"role": "user", "content": "Reply with just OK plus the final answer again."})
        else:
            msgs = [{"role": "user", "content": r["prompt"]}]
        rows.append((msgs, r["completion"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16,
                                                 trust_remote_code=False)
    model.to(dev)

    # Freeze the shared-cache surface: everything below the tail.
    n_frozen = 0
    for name, prm in model.named_parameters():
        keep = False
        for tail_id in range(FROZEN_LAYERS, 16):
            if f"layers.{tail_id}." in name:
                keep = True
        if "lm_head" in name or "norm" in name and "layers" not in name:
            keep = True
        prm.requires_grad_(keep)
        if not keep:
            n_frozen += 1
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"params frozen: {n_frozen}, trainable: {len(trainable)} "
          f"({sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)/1e6:.0f}M)")
    if not trainable:
        print("nothing trainable — abort"); return 2

    rows = load_rows(args.data)
    print(f"rows: {len(rows)}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    eos_ids = {tok.eos_token_id}
    for eid in getattr(tok, "additional_special_tokens", []) or []:
        t = tok.convert_tokens_to_ids(eid)
        if isinstance(t, int) and t >= 0:
            eos_ids.add(t)

    for ep in range(args.epochs):
        tot_loss, n = 0.0, 0
        for msgs, comp in rows:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            full = prompt + comp + tok.eos_token
            enc = tok(full, return_tensors="pt", truncation=True, max_length=args.max_len)
            input_ids = enc["input_ids"].to(dev)
            labels = input_ids.clone()
            p_len = min(tok(prompt, add_special_tokens=False)["input_ids"].__len__(),
                        labels.shape[1])
            labels[:, :p_len] = -100
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            if not torch.isfinite(loss):
                print("non-finite loss; skipping row"); continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot_loss += loss.item(); n += 1
        print(f"epoch {ep+1}: mean loss {tot_loss/max(n,1):.4f} over {n} rows")

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)
    # record provenance so M3 can gate on the frozen digest staying unchanged
    meta = {"frozen_layers": FROZEN_LAYERS, "trainable": len(trainable),
            "data": os.path.abspath(args.data), "epochs": args.epochs, "lr": args.lr}
    json.dump(meta, open(os.path.join(args.out_dir, "DISTILL.json"), "w"), indent=1)
    print(f"student tail saved -> {args.out_dir}")
    print("NEXT: verify frozen layers still byte-match the parent via "
          "nested_extract.py --verify on the rebuilt dir before M3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
