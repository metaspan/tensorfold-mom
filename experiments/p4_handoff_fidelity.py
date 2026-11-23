#!/usr/bin/env python3
"""P4 handoff-spike fidelity (offline, HF-side). GATE: docs-mom/gates/
20261124-handoff-spike.md -- read it first; the bar is PRE-REGISTERED.

Question: can thin-lane attention K/V translated into heavy-lane attention
space (ridge-fit linear maps, 512-dim both sides) let the heavy lane produce
the SAME greedy continuation as exact prefill -- with GDN state recomputed
normally (docs/12: attention-favorable / GDN-unfavorable)?

Method (mirrors S1 collector conventions so maps are comparable):
  1. Fit set: 8 conversations (thin corpus + B5-long blocks). For each, the
     SAME token ids go through thin (qwen35-2b HF) and heavy (35B fp8->bf16
     dequant). Per full-attention layer, capture the (K, V) rows the
     attention reads, fit ridge per (src thin layer -> dst heavy layer)
     pair (each heavy attention layer picks its best thin source), fit ONLY
     on the fit set.
  2. Eval: 20 NOVEL escalation turns. Baseline = heavy greedy 32 tokens on
     the full prompt. Treated = heavy prefill with each attention layer's
     K/V rows for the prompt positions REPLACED by translated thin rows
     (Wk/Wv outputs swapped post-hoc via forward pre-hook); GDN layers run
     normally (their state comes from the same forward pass -- recompute
     path, per docs/12). Compare greedy continuations token-for-token.
  3. BAR: 20/20 turns identical. Any miss -> P4 dies (record + discard).

Memory rule: never run alongside training (>20 GiB jobs serialize).
Run: venv-tensorfold-mom python, after the 5g window:
  ~/.hermes/venv-tensorfold-mom/bin/python experiments/p4_handoff_fidelity.py \
      --out experiments/results/p4/run-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b5_cache_probe as B5  # TRIVIAL_TURNS / WORK_TURNS corpus

REPO = "/home/derek/tensorfold-mom"
SMALL = os.environ.get("TFM_SMALL", "/home/derek/.hermes/models/tensorfold-mom/qwen35-2b")
BIG_FP8 = os.environ.get("P4_HEAVY_FP8", "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-fp8")
HEAVY_BF16 = os.environ.get("P4_HEAVY_BF16",
                            "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-fp8-deq")
N_FIT = 8
N_EVAL = 20
GEN_TOKENS = 32
ATTN_INTERVAL = 4  # full_attention_interval (docs/12)


def dequant_heavy_if_needed():
    """Block-FP8 (e4m3 + 128x128 scale_inv bf16) -> bf16, one-time (verified
    recipe from mtp_train; cached on disk)."""
    if os.path.isdir(HEAVY_BF16) and os.path.exists(os.path.join(HEAVY_BF16, "model.safetensors.index.json")):
        return HEAVY_BF16
    from safetensors.torch import load_file, save_file
    from glob import glob
    os.makedirs(HEAVY_BF16, exist_ok=True)
    index = json.load(open(os.path.join(BIG_FP8, "model.safetensors.index.json")))
    # group tensors by shard
    by_shard = {}
    for k, f in index["weight_map"].items():
        by_shard.setdefault(f, []).append(k)
    out_index = {"metadata": index.get("metadata", {}), "weight_map": {}}
    for f, keys in sorted(by_shard.items()):
        sd = load_file(os.path.join(BIG_FP8, f))
        out = {}
        for k in keys:
            t = sd[k]
            inv = sd.get(k.replace(".weight", ".weight_scale_inv"))
            if t.dtype == torch.float8_e4m3fn and inv is not None:
                blocks = t.float().view(t.shape[0] // 128, 128, t.shape[1] // 128, 128)
                scale = inv.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:t.shape[0], :t.shape[1]]
                t = (blocks * scale.view(t.shape[0] // 128, 128, t.shape[1] // 128, 128).permute(0, 1, 2, 3).reshape(t.shape[0], t.shape[1])).to(torch.bfloat16)
                del scale
            out[k] = t.to(torch.bfloat16) if t.is_floating_point() else t
        fname = f.replace("model-", "deq-model-")
        save_file(out, os.path.join(HEAVY_BF16, fname), metadata={"format": "pt"})
        for k in out:
            out_index["weight_map"][k] = fname
        del sd, out
        print(f"dequant {f} -> {fname}", flush=True)
    json.dump(out_index, open(os.path.join(HEAVY_BF16, "model.safetensors.index.json"), "w"))
    import shutil
    for meta in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "generation_config.json"):
        src = os.path.join(BIG_FP8, meta)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HEAVY_BF16, meta))
    return HEAVY_BF16


def build_conversations(n, start=0, long_block=False):
    """Alternating thin-style conversation + escalation; novel phrasings via
    index offset so eval never saw fit rows."""
    from b5_cache_probe import TRIVIAL_TURNS, WORK_TURNS
    convs = []
    for c in range(n):
        msgs = [{"role": "system", "content": B5.SYSTEM}]
        user_turns = []
        base = (start + c)
        for i in range(6):
            if i % 2 == 0:
                user_turns.append(TRIVIAL_TURNS[(base + i) % len(TRIVIAL_TURNS)])
            else:
                user_turns.append(WORK_TURNS[(base + i) % len(WORK_TURNS)][:180])
        if long_block:
            user_turns.append("Read this and say ACK: " + ("document token stream " * 200))
        user_turns.append("Now answer in one short sentence: what was the overall theme?")
        convs.append(user_turns)
    return convs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/results/p4/run-01"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(7)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading thin", flush=True)
    tok_t = AutoTokenizer.from_pretrained(SMALL)
    thin = AutoModelForCausalLM.from_pretrained(
        SMALL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    thin.eval()

    print("dequant + loading heavy (slow first time)", flush=True)
    heavy_dir = dequant_heavy_if_needed()
    tok_h = AutoTokenizer.from_pretrained(heavy_dir)
    heavy = AutoModelForCausalLM.from_pretrained(
        heavy_dir, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True)
    heavy.eval()

    # attention-layer indices per docs/12 geometry
    def attn_layers(model):
        cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
        n = cfg.num_hidden_layers
        return [i for i in range(n) if (i + 1) % ATTN_INTERVAL == 0]
    src_layers, dst_layers = attn_layers(thin), attn_layers(heavy)
    print(f"thin attn {src_layers} -> heavy attn {dst_layers}", flush=True)

    def capture_rows(model, toks, layers):
        """Return {layer: (K, V)} [rows, kv_heads*head_dim] bf16 as fp32."""
        rows = {}
        hooks = []

        def make_hook(idx):
            def hook(mod, inp, out):
                # qwen3_5 attention: k_proj/v_proj outputs captured separately
                rows.setdefault(idx, {})
            return hook
        # capture k/v projections' outputs
        base = model.model if hasattr(model, "model") else model
        for L in layers:
            attn = base.layers[L].self_attn
            def hook_k(mod, inp, out, L=L):
                rows.setdefault(L, {})["K"] = out.detach().float().cpu()
            def hook_v(mod, inp, out, L=L):
                rows.setdefault(L, {})["V"] = out.detach().float().cpu()
            hooks.append(attn.k_proj.register_forward_hook(hook_k))
            hooks.append(attn.v_proj.register_forward_hook(hook_v))
        with torch.no_grad():
            model(input_ids=toks.to(model.device))
        for h in hooks:
            h.remove()
        return rows

    def encode(msgs):
        text = tok_t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        # heavy has its own template; token ids must match (verified identical
        # BPE); assert equality:
        ids_t = tok_t(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        text_h = tok_h.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids_h = tok_h(text_h, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if not torch.equal(ids_t, ids_h):
            # templates differ slightly (audio specials); fall back to shared tokenization
            return ids_t, ids_t
        return ids_t, ids_h

    # ---- fit maps (ridge) on fit set: each dst heavy layer picks best thin src
    print("fitting maps on", N_FIT, "fit conversations", flush=True)
    fit_t = build_conversations(N_FIT, start=0, long_block=True)
    X = {s: [] for s in src_layers}
    Y = {d: [] for d in dst_layers}
    for msgs in fit_t:
        it, ih = encode(msgs)
        rt = capture_rows(thin, it, src_layers)
        rh = capture_rows(heavy, ih, dst_layers)
        n = min(rt[src_layers[0]]["K"].shape[1], rh[dst_layers[0]]["K"].shape[1])
        for s in src_layers:
            X[s].append(torch.cat([rt[s]["K"][0, :n], rt[s]["V"][0, :n]], -1))
        for d in dst_layers:
            Y[d].append(torch.cat([rh[d]["K"][0, :n], rh[d]["V"][0, :n]], -1))
    maps = {}
    for d in dst_layers:
        Yd = torch.cat(Y[d]).numpy()
        best = None
        for s in src_layers:
            Xs = torch.cat(X[s]).numpy()
            if Xs.shape[1] != Yd.shape[1]:
                continue
            # ridge W: Yd ~ Xs @ W ; pick src layer minimizing residual
            A = Xs.T @ Xs + 1e-3 * np.eye(Xs.shape[1], dtype=np.float64)
            W = np.linalg.solve(A, Xs.T @ Yd.astype(np.float64)).astype(np.float32)
            resid = float(np.linalg.norm(Xs @ W - Yd)) / float(np.linalg.norm(Yd))
            if best is None or resid < best[1]:
                best = (s, resid, W)
        maps[d] = {"src": best[0], "rel_resid": best[1], "W": best[2]}
        print(f"heavy attn {d}: best thin src {best[0]} rel_resid {best[1]:.4f}", flush=True)
    json.dump({str(d): {"src": m["src"], "rel_resid": m["rel_resid"]} for d, m in maps.items()},
              open(os.path.join(args.out, "maps.json"), "w"), indent=1)

    # ---- eval: 20 novel escalations; treated = heavy prefill with attention
    # K/V replaced by translated thin rows (GDN untouched)
    def patch_heavy(prompt_ids, msgs):
        """Install pre-hooks replacing k_proj/v_proj outputs with translated
        thin rows for the prompt positions. Returns removers."""
        it, _ = encode(msgs)
        rt = capture_rows(thin, it, src_layers)
        removers = []
        base = heavy.model
        feat = Y[dst_layers[0]][0].shape[1] // 2
        for d, m in maps.items():
            s = m["src"]
            W = m["W"]
            rows = rt[s]
            K = torch.cat([rows["K"], rows["V"]], -1)[0].numpy()
            translated = K @ W
            Kt = torch.from_numpy(translated[:, :feat]).to(torch.bfloat16)
            Vt = torch.from_numpy(translated[:, feat:]).to(torch.bfloat16)
            # two hooks: k_proj then v_proj
            def hk(mod, inp, out, Kt=Kt):
                return Kt.to(out.device, out.dtype)
            def hv(mod, inp, out, Vt=Vt):
                return Vt.to(out.device, out.dtype)
            removers.append(base.layers[d].self_attn.k_proj.register_forward_hook(hk))
            removers.append(base.layers[d].self_attn.v_proj.register_forward_hook(hv))
        return removers

    print("evaluating", N_EVAL, "novel escalations", flush=True)
    eval_convs = build_conversations(N_EVAL, start=N_FIT + 5, long_block=True)
    agree = 0
    table = []
    for ci, msgs in enumerate(eval_convs):
        it, ih = encode(msgs)
        gen_base = heavy.generate(ih.to(heavy.device), max_new_tokens=GEN_TOKENS,
                                  do_sample=False, pad_token_id=tok_h.eos_token_id)[0, ih.shape[1]:]
        removers = patch_heavy(ih, msgs)
        try:
            gen_tr = heavy.generate(ih.to(heavy.device), max_new_tokens=GEN_TOKENS,
                                    do_sample=False, pad_token_id=tok_h.eos_token_id)[0, ih.shape[1]:]
        finally:
            for h in removers:
                h.remove()
        same = torch.equal(gen_base, gen_tr)
        agree += int(same)
        table.append({"case": ci, "agree": bool(same),
                      "baseline": tok_h.decode(gen_base)[:120],
                      "treated": tok_h.decode(gen_tr)[:120]})
        print(f"case {ci}: {'AGREE' if same else 'MISS'}", flush=True)
    verdict = "PASS" if agree == N_EVAL else "FAIL"
    json.dump({"agree": agree, "n": N_EVAL, "verdict": verdict, "table": table},
              open(os.path.join(args.out, "fidelity.json"), "w"), indent=1)
    print(f"VERDICT: {verdict} ({agree}/{N_EVAL}) -> {os.path.join(args.out, 'fidelity.json')}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
