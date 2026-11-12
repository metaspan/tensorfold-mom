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
    # arithmetic (short-answer)
    "What is 2+2? Answer in one word.",
    "What is 17+26? Answer with just the number.",
    "What is 144/12? One word.",
    "What is 9*8? Just the number.",
    "What is 100-37? Just the number.",
    "What is 15% of 200? Just the number.",
    "Is 91 prime? Yes or no.",
    "Which is bigger, 7/8 or 0.86? One word.",
    "What is the square root of 81? One word.",
    "Round 3.14159 to two decimals.",
    "What is 2^10? Just the number.",
    "Sum of 3, 17 and 5? Just the number.",
    # units & conversions
    "Convert 3 miles to kilometres, one decimal place.",
    "Convert 72 Fahrenheit to Celsius, rounded to one decimal.",
    "How many minutes are in 2.5 hours?",
    "Convert 5 kilograms to pounds, whole number.",
    "How many centimetres in 2 inches, one decimal?",
    "How many days are in a leap year? One word.",
    # spelling / grammar
    "Is 'their' spelled right in: 'They went to their house'?",
    "Fix the grammar: 'he dont like apples'.",
    "Spell the word that means 'fear of spiders'.",
    "Which is correct: 'a effect' or 'an effect'?",
    "Plural of 'octopus' in formal English?",
    "Is 'recieve' correct? Answer yes or no.",
    # extraction / formatting
    "Extract the email from: 'Reach me at jane.d@corp.io please'.",
    "Extract the URL from: 'See https://example.com/docs for more'.",
    "Return only the city from: 'Shipped from Berlin, DE on Tuesday'.",
    "Convert 'on=1 off=0' to a JSON object with keys on and off.",
    "List three prime numbers under 20.",
    "Capitalize each word of: 'the quick brown fox'.",
    # translation
    "Translate 'good morning' to German.",
    "Translate 'thank you' to Japanese.",
    "Translate 'where is the station' to French.",
    "How do you say 'cheers' in Spanish?",
    # dates / time
    "What day of the week is 2026-01-01?",
    "What year did World War 2 end? One word.",
    "How many days are there in March?",
    "Is 2100 a leap year? Yes or no.",
    # quick facts (yes/no, one word)
    "Yes or no: can whales breathe underwater?",
    "Capital of Australia? One word.",
    "What does HTTP 429 mean, one sentence.",
    "Give me a one-line summary of what an API is.",
    "Chemical symbol for sodium? One word.",
    "How many continents are there? One word.",
    "Who wrote 'Romeo and Juliet'? One name.",
    "Is the AND gate output 1 when both inputs are 1? Yes or no.",
    "What planet is known as the Red Planet? One word.",
    "Largest ocean? One word.",
    "Does water boil at 100C at sea level? Yes or no.",
    "Number of bits in a byte? One word.",
    "What language is this code in: print('hi')? One word.",
    "What does CPU stand for? One line.",
    "Define 'latency' in one sentence.",
    # tiny code / logic
    "Reverse the string 'hello'.",
    "In Python, is a list mutable? Yes or no.",
    "What is 10 in binary? Just the digits.",
    "One line of CSS to center text.",
    "Does git commit -m need a message? Yes or no.",
    "True or false: JSON keys must be quoted.",
    "What does SELECT * FROM t LIMIT 1 return, one line?",
    "Convert '3 apples and 4 apples' to just 'N apples'.",
    "Sort these: 3, 1, 2. Just the sequence.",
    # misc trivial-lane traffic
    "Suggest one color for a calm dashboard background.",
    "One sentence: what is a webhook?",
    "Give a two-word affirmative response.",
    "Repeat exactly: 'all systems go'.",
    "What is the IP range for 169.254.0.0/16 used for, one line?",
    "How many seconds in 5 minutes? Just the number.",
    "Singular of 'data' in strict usage?",
    "Is UTC a timezone or a time standard? One of the two.",
    "Which is further north: Lisbon or Athens? One word.",
    "How many sides does a hexagon have? One word.",
    "Opposite of 'verbose' one word?",
    "What does TTL mean in networking? One line.",
    "Should a health check return 503 when unhealthy? Yes or no.",
    "One word answer: is Python statically typed?",
    "Give me the answer to life, the universe, everything, as an integer.",
    "Convert 'yes' to a boolean lowercase.",
    "Is 0.1+0.2 exactly 0.3 in float32? Yes or no.",
    "What is the ASCII code of 'A'? Just the number.",
    "Which port does HTTPS use by default? Just the number.",
    "What does gzip do? One sentence.",
    "Name one HTTP idempotent method.",
    "Is SQLite a client-server database? Yes or no.",
    "What does 'LGTM' mean? One short phrase.",
    "How many bits is a u8? One word.",
    "Should logs to stderr be flushed on error? Yes or no.",
    "What is the time complexity of binary search? One phrase.",
    "In one word, is Rust garbage-collected?",
    "Does TCP guarantee order? Yes or no.",
    "What does chmod 755 do, one sentence.",
    "Is JSON a superset of YAML? Yes or no.",
    "One word: hotter or colder is the tropics vs tundra?",
    "Convert 0.5 to a percentage, one word.",
    "What is the MIME type for JSON? One line.",
    "Name one stable Linux package manager.",
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
    ap.add_argument("--thinking-rate", type=float, default=0.2,
                    help="fraction of rows rolled with thinking ON (spec data mix)")
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
        seen = set()
        with open(args.out, "w") as f:
            for ep in range(args.passes):
                for s in SEEDS:
                    msgs = [{"role": "user", "content": s}]
                    # spec data mix: a slice of rows carries the teacher's thinking
                    # block, captured separately (the trainer composes it into the
                    # assistant turn as <think>...</think>).
                    want_think = args.thinking or rng.random() < args.thinking_rate
                    kw = tpl_kwargs if args.thinking else (
                        {} if want_think else {"enable_thinking": False})
                    try:
                        r = post({"model": "heavy", "messages": msgs, "max_tokens": 512,
                                  "temperature": 0.8, "stream": False,
                                  "chat_template_kwargs": kw})
                        comp = r["choices"][0]["message"]["content"]
                        think = (r["choices"][0]["message"].get("reasoning_content")
                                 if want_think else None)
                    except Exception as e:
                        print(f"seed failed ({e}); skipping")
                        continue
                    key = (s, comp[:120])
                    if key in seen:
                        continue  # greedy-ish repeats add no signal; temperature noise does
                    seen.add(key)
                    row = {"prompt": s, "completion": comp, "thinking": bool(want_think)}
                    if think:
                        row["thinking_text"] = think
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
