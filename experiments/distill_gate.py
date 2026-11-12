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

The gate does NOT gate on adoption or KV bytes -- those are M2/M3/M4's job;
this is the coherence deliverable only.

Run (GPU window): /home/derek/.venv/bin/python distill_gate.py \
    --model /home/derek/models/tensorfold-mom/qwen36-nested-16l-t2
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="student checkpoint dir")
    ap.add_argument("--qa", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "gate_qa.json"))
    ap.add_argument("--pass-min", type=int, default=10, help="correct of 12 required")
    args = ap.parse_args()

    qa = json.load(open(args.qa))
    assert len(qa) == 12, "gate is registered for 12 items"
    if not os.path.isdir(args.model):
        print(f"no such model dir: {args.model}")
        return 2

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
