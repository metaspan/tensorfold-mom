#!/usr/bin/env python3
"""Pre-registered coherence gate for a distilled student (docs/13 part 3, v1).

PASS criterion, FIXED BEFORE TRAINING RUNS (S1 lesson: no post-hoc thresholds):
  - 12 fixed QA items, disjoint from every distill_collect seed (verify by
    inspection before each new seed batch lands);
  - greedy decoding, thinking OFF (what the trivial lane serves in production);
  - normalized containment of any accepted answer string (casefold, strip
    punctuation/whitespace);
  - PASS = >= 10/12 correct, zero empty/errored replies, zero template-token
    leakage (no '<|im_start|', '<|im_end|', '<think>' in any reply).

The gate does NOT gate on adoption or KV bytes -- those are M3/M4's job;
this is the coherence deliverable only.

AMENDMENT 2026-11-13 (before any run, harness only -- items, decoding rules
and the >=10/12 bar are UNCHANGED): --engine hf added, and made the mode for
bf16 students. The native CUDA engine categorically refuses plain bf16 MoE
checkpoints (cuda_geometry.zig:290 `if (bf16 and moe) return
error.UnsupportedQwenQuantization`), so the t1 student (bf16 tail, 279
tensors) can never boot the native server; the native gate stays registered
for NVFP4 students (BL-30 ships the quantized export + trained MTP head).
Greedy, thinking off, containment judging, leakage checks are identical in
both modes.

Run (GPU window): /home/derek/.venv/bin/python distill_gate.py --engine hf \
    --model /home/derek/models/tensorfold-mom/qwen36-nested-16l-t1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
PORT = 8219
LOG = "/tmp/distill-gate.log"


def post(payload, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def norm(s: str) -> str:
    return re.sub(r"[^0-9a-z/]+", " ", s.casefold()).strip()


def gate_hf(qa, args) -> int:
    """The same 12 items, same greedy + thinking-off rules, generated with the
    HF stack (the native engine refuses bf16 MoE checkpoints outright)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    def reply(item):
        msgs = [{"role": "user", "content": item["q"]}]
        try:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        n_in = ids["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=48, do_sample=False)
        return tok.decode(out[0][n_in:], skip_special_tokens=True)

    return judge(qa, args, reply)


def judge(qa, args, reply) -> int:
    correct, fails, leaks = 0, [], []
    for i, item in enumerate(qa):
        try:
            got = reply(item)
        except Exception as e:
            got = f"<error {e}>"
        low = got.casefold()
        if any(t in low for t in ("<|im_start|", "<|im_end|", "<think>")):
            leaks.append((i, got[:60]))
        n = norm(got)
        if any(norm(a) in n for a in item["a"]) and not got.startswith("<error"):
            correct += 1
        else:
            fails.append((i, item["q"], got[:80]))
    for i, q, g in fails:
        print(f"  miss {i}: {q!r} -> {g!r}")
    for i, g in leaks:
        print(f"  LEAK {i}: {g!r}")
    verdict = correct >= args.pass_min and not leaks
    print(f"GATE {os.path.basename(args.model)}: {correct}/12 correct, "
          f"{len(leaks)} template-leak(s) -> {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="student checkpoint dir")
    ap.add_argument("--qa", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "gate_qa.json"))
    ap.add_argument("--pass-min", type=int, default=10, help="correct of 12 required")
    ap.add_argument("--engine", choices=("native", "hf"), default="native",
                    help="native boots tensorfold mom-serve (NVFP4 students); "
                         "hf generates with transformers (bf16 students the engine refuses)")
    args = ap.parse_args()

    qa = json.load(open(args.qa))
    assert len(qa) == 12, "gate is registered for 12 items"
    if not os.path.isdir(args.model):
        print(f"no such model dir: {args.model}")
        return 2

    if args.engine == "hf":
        return gate_hf(qa, args)
    log = open(LOG, "w")
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT),
                          "--model", args.model, "--name", "student", "--lane", "default"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=os.environ.copy())
    correct = 0
    fails = []
    leaks = []
    try:
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if json.loads(r.read().decode()).get("status") == "ok":
                        break
            except Exception:
                pass
            if p.poll() is not None:
                print("server died:", open(LOG).read()[-2000:])
                return 2
            time.sleep(2)
        else:
            print("health timeout:", open(LOG).read()[-2000:])
            return 2

        for i, item in enumerate(qa):
            try:
                r = post({"model": "student",
                          "messages": [{"role": "user", "content": item["q"]}],
                          "max_tokens": 48, "temperature": 0, "stream": False,
                          "chat_template_kwargs": {"enable_thinking": False}})
                got = r["choices"][0]["message"]["content"]
            except Exception as e:
                got = f"<error {e}>"
            low = got.casefold()
            if any(t in low for t in ("<|im_start|", "<|im_end|", "<think>")):
                leaks.append((i, got[:60]))
            n = norm(got)
            if any(norm(a) in n for a in item["a"]) and not got.startswith("<error"):
                correct += 1
            else:
                fails.append((i, item["q"], got[:80]))

        for i, q, g in fails:
            print(f"  miss {i}: {q!r} -> {g!r}")
        for i, g in leaks:
            print(f"  LEAK {i}: {g!r}")
        verdict = correct >= args.pass_min and not leaks
        print(f"GATE {os.path.basename(args.model)}: {correct}/12 correct, "
              f"{len(leaks)} template-leak(s) -> {'PASS' if verdict else 'FAIL'}")
        return 0 if verdict else 1
    finally:
        p.terminate()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    sys.exit(main())
