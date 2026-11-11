#!/usr/bin/env python3
"""S1 collect corpus B: the GENERALIZATION set for the S1 gate.

Run 1 of the gate fitted on the B5 28-turn conversation and held out other
checkpoints OF THE SAME conversation. Because conversation prefixes are
nested and KV rows are causal, every held-out row already occurred verbatim
in training (verified: 519/519), so a top-1 of 1.0 there certifies row
retrievability, not generalization. This collector replays a DISJOINT 20-turn
conversation (topics, phrasings and lengths that share no user text with
B5) on both lanes, so a map fitted on B5 is scored on rows it has literally
never seen. s1_fit.py --train-corpus/--eval-corpus consumes both dump roots.

Window owner stops production first; same server recipe as s1_collect.py.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
SMALL = "/home/derek/models/tensorfold-mom/qwen35-2b"
BIG = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
PORT = 8213
DUMP = "/home/derek/tensorfold-mom/experiments/results/kvdump-b"

# Deliberately disjoint from b5_cache_probe's TRIVIAL_TURNS/WORK_TURNS (no
# arithmetic-chain, capitalise/translate/spell/capital-of-France trivial set,
# no merge-sort/TLS/rate-limiter/job-queue work set).
TRIVIAL_TURNS_B = [
    "What is 7 times 8?",
    "Give me the boiling point of water in Celsius.",
    "Name the three primary colours.",
    "How many days are in a leap year?",
    "What planet is known as the red planet?",
    "Convert 5 kilometres to miles, roughly.",
    "Who wrote Romeo and Juliet?",
    "What is the square root of 144?",
    "Is a whale a fish? One word.",
    "What language is 'danke' from?",
]
WORK_TURNS_B = [
    "Write a Python function that parses an .ics calendar file and returns "
    "the busiest week; explain your parsing choices and handle recurring events.",
    "Explain the difference between optimistic and pessimistic concurrency control "
    "in databases, then recommend one for a hotel booking system with reasons.",
    "Design a schema for a multi-tenant todo app with shared projects; cover "
    "indexes, soft deletes, and how you'd shard it past 100M rows.",
    "Walk through how a B+ tree handles a range scan after a page split, and "
    "what changes when the tree is stored on an SSD with random 4K reads.",
    "A service's p99 latency spiked after a deploy but p50 is unchanged. "
    "List the four most likely causes in order and how you'd confirm each.",
    "Compare TCP BBR and CUBIC on a lossy satellite link, then state which you "
    "would pick for bulk file sync and why.",
    "Write a SQL query that finds, per department, the employee with the third "
    "highest salary, breaking ties by hire date; explain the window functions.",
    "Describe how garbage collection in a generational heap decides what to "
    "promote, and what pathological allocation pattern would defeat it.",
    "Draft the API contract for a rate-limited webhooks service: auth, retry "
    "semantics, idempotency, and backpressure signals.",
    "Explain why float addition is non-associative, then show two money-handling "
    "strategies that avoid the resulting bugs.",
]
TURNS = len(TRIVIAL_TURNS_B) + len(WORK_TURNS_B)


def post(payload, timeout=300):
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
    if os.path.isdir(DUMP):
        subprocess.run(["rm", "-rf", DUMP], check=True)
    env = os.environ.copy()
    env["TF_QWEN_REQUEST_STARTS"] = "1"
    env["TF_MOM_KVDUMP"] = DUMP
    log = open("/tmp/s1-mom2.log", "w")
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT),
                          "--parallel", "2", "--no-sticky",
                          "--model", SMALL, "--name", "qwen3.5-2b", "--lane", "trivial", "--pin",
                          "--model", BIG, "--name", "qwen3.6-35b", "--lane", "default"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            if p.poll() is not None:
                print("server died:", open("/tmp/s1-mom2.log").read()[-1200:])
                return 2
            if health():
                break
            time.sleep(2)
        else:
            print("health timeout")
            return 2

        for lane_model in ("qwen3.5-2b", "qwen3.6-35b"):
            msgs = [{"role": "system", "content": "You are a helpful assistant."}]
            for i in range(TURNS):
                user = WORK_TURNS_B[i // 2][:280] if i % 2 == 1 else TRIVIAL_TURNS_B[i // 2]
                msgs.append({"role": "user", "content": user})
                r = post({"model": lane_model, "messages": msgs, "max_tokens": 16,
                          "temperature": 0,
                          "chat_template_kwargs": {"enable_thinking": False}})
                if not r.get("choices"):
                    print(f"lane {lane_model} turn {i}: empty reply")
                    return 1
                msgs.append({"role": "assistant",
                             "content": r["choices"][0]["message"]["content"]})
            print(f"lane {lane_model}: {TURNS} turns replayed")

        time.sleep(3)
        counts = {}
        for m in sorted(os.listdir(DUMP)) if os.path.isdir(DUMP) else []:
            d = os.path.join(DUMP, m)
            if os.path.isdir(d):
                counts[m] = len([f for f in os.listdir(d) if f.endswith(".bin")])
        print("dump counts:", json.dumps(counts))
        if len(counts) < 2 or min(counts.values()) < TURNS:
            print("S1 corpus-B collect: FAIL (missing dumps)")
            return 1
        print("S1 corpus-B collect: OK")
        return 0
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
