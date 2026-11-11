#!/usr/bin/env python3
"""Distillation data collection (docs/13 part 3, v1 = SFT-on-teacher-rollouts).

Boots the heavy engine (mom-serve single lane), rolls out N prompts sampled
from a seed corpus with temperature, and writes JSONL rows:

  {"prompt": <str, chat-templated user turn(s)>,
   "completion": <str, heavy rollout INCLUDING <|im_start|> block when enabled>,
   "thinking": bool}

Lane-hop rows (the MoM-specific data mix): with probability --hop-rate a row
is additionally emitted with the *heavy-prefixed* structure — a heavy answer
that a later trivial request continues — by replaying the completion as a
prior assistant turn and rolling a short continuation. The thin student is
trained on chat-templated (prompt -> completion) with the thinking block
INSIDE the completion, which is what makes "thinking blocks in history"
familiar to it at lane-hop time.

v1 uses token CE on teacher text (SFT). v2 (needs logprobs in the engine —
/v1/models advertises logprobs:false today) swaps CE for KL to teacher
logits; the row format already carries everything v2 needs to add them.

Run (GPU window): /home/derek/.venv/bin/python distill_collect.py --out <jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
BIG = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
PORT = 8217
LOG = "/tmp/distill-collect.log"

# Seed prompts: the trivial-lane traffic the student must own. Mix of styles;
# replace/extend with real gateway logs for production distillation runs.
SEEDS = [
    "What is 2+2? Answer in one word.",
    "Convert 3 miles to kilometres, one decimal place.",
    "Is 'their' spelled right in: 'They went to their house'?",
    "Give me a one-line summary of what an API is.",
    "What day of the week is 2026-01-01?",
    "Extract the email from: 'Reach me at jane.d@corp.io please'.",
    "Translate 'good morning' to German.",
    "Which is bigger, 7/8 or 0.86? One word.",
    "List three prime numbers under 20.",
    "Fix the grammar: 'he dont like apples'.",
    "What does HTTP 429 mean, one sentence.",
    "Capital of Australia? One word.",
    "Round 3.14159 to two decimals.",
    "How many minutes are in 2.5 hours?",
    "Yes or no: can whales breathe underwater?",
    "Spell the word that means 'fear of spiders'.",
]


def post(payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=8, help="full passes over seeds")
    ap.add_argument("--hop-rate", type=float, default=0.3)
    ap.add_argument("--thinking", action="store_true", help="leave thinking on")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if os.path.exists(args.out):
        print(f"{args.out} exists; refusing to clobber")
        return 2
    env = os.environ.copy()
    env["TF_QWEN_REQUEST_STARTS"] = "1"
    log = open(LOG, "w")
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT),
                          "--model", BIG, "--name", "heavy", "--lane", "default"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    n_rows = 0
    try:
        deadline = time.time() + 900
        while time.time() < deadline and not health():
            if p.poll() is not None:
                print("server died:", open(LOG).read()[-1500:])
                return 2
            time.sleep(2)
        else:
            if not health():
                print("health timeout"); return 2

        tpl_kwargs = {} if args.thinking else {"enable_thinking": False}
        with open(args.out, "w") as f:
            for ep in range(args.passes):
                for s in SEEDS:
                    msgs = [{"role": "user", "content": s}]
                    try:
                        r = post({"model": "heavy", "messages": msgs, "max_tokens": 512,
                                  "temperature": 0.8, "stream": False,
                                  "chat_template_kwargs": tpl_kwargs})
                        comp = r["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"seed failed ({e}); skipping")
                        continue
                    row = {"prompt": s, "completion": comp, "thinking": bool(args.thinking)}
                    f.write(json.dumps(row) + "\n"); n_rows += 1
                    if rng.random() < args.hop_rate:
                        # lane-hop row: heavy answer sits in history; trivial
                        # follow-up completion from heavy is the target.
                        hop_msgs = msgs + [{"role": "assistant", "content": comp},
                                           {"role": "user",
                                            "content": "Thanks. Reply with just OK plus the final answer again."}]
                        try:
                            r2 = post({"model": "heavy", "messages": hop_msgs, "max_tokens": 64,
                                       "temperature": 0.5, "stream": False,
                                       "chat_template_kwargs": tpl_kwargs})
                            row2 = {"prompt": s, "history_completion": comp,
                                    "completion": r2["choices"][0]["message"]["content"],
                                    "lane_hop": True, "thinking": bool(args.thinking)}
                            f.write(json.dumps(row2) + "\n"); n_rows += 1
                        except Exception as e:
                            print(f"hop failed ({e})")
                print(f"epoch {ep+1}/{args.passes}: {n_rows} rows")
                f.flush()
    finally:
        p.terminate()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    print(f"wrote {n_rows} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
