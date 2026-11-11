#!/usr/bin/env python3
"""S1 collect: paired KV dumps of the SAME prefixes from both MoM lanes.

Boots mom-serve with TF_MOM_KVDUMP + TF_QWEN_REQUEST_STARTS=1 and replays the
B5 28-turn conversation twice — once forcing qwen3.5-2b, once forcing
qwen3.6-35b — so every message-start checkpoint is dumped per lane with
identical token ids. s1_fit.py consumes the dump dir. Window owner stops
production first."""
import json
import os
import subprocess
import sys
import time
import urllib.request

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
SMALL = "/home/derek/models/tensorfold-mom/qwen35-2b"
BIG = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
PORT = 8212
DUMP = "/home/derek/tensorfold-mom/experiments/results/kvdump"

sys.path.insert(0, "/home/derek/tensorfold-mom/experiments")
from b5_cache_probe import TRIVIAL_TURNS, WORK_TURNS, TURNS  # shared corpus


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
    log = open("/tmp/s1-mom.log", "w")
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
                print("server died:", open("/tmp/s1-mom.log").read()[-1200:])
                return 2
            if health():
                break
            time.sleep(2)
        else:
            print("health timeout")
            return 2

        for lane_model in ("qwen3.5-2b", "qwen3.6-35b"):
            msgs = [{"role": "system", "content": "You are a concise assistant."}]
            for i in range(TURNS):
                user = WORK_TURNS[i // 2][:280] if i % 2 == 1 else TRIVIAL_TURNS[i]
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

        # let any queued dumps flush
        time.sleep(3)
        counts = {}
        for m in sorted(os.listdir(DUMP)) if os.path.isdir(DUMP) else []:
            d = os.path.join(DUMP, m)
            if os.path.isdir(d):
                counts[m] = len([f for f in os.listdir(d) if f.endswith(".bin")])
        print("dump counts:", json.dumps(counts))
        if len(counts) < 2 or min(counts.values()) < TURNS:
            print("S1 collect: FAIL (missing dumps)")
            return 1
        print("S1 collect: OK")
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
