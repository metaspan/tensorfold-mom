#!/usr/bin/env python3
"""G3 decode-resume gate (docs/13 M3): adoption must be observationally free.

Three phases on one mom-serve:
  A. baseline: replay N teacher-forced turns on the NESTED lane alone,
     recording its reply token streams (replies from parent_replies.json keep
     prompts identical across phases).
  B. adopt: replay the same turns on the PARENT lane (fills the parent's kept
     states), then restart the server and replay the same turns on the nested
     lane with TF_MOM_ADOPT=1 -- nested submits consult the parent's kept
     states and adopt (docs/13 v1; requires the mom-side wiring commit).
  C. compare: nested replies byte-for-byte equal across A and B. Any
     difference means the adopted bytes differ from computed ones -> the
     kvdump diff is the debugger.

Until the wiring lands, phase B without TF_MOM_ADOPT support just re-runs
the baseline (the probe then PASSes vacuously -- the server log line
'adopted' must appear in phase B for the gate to count; the probe refuses to
call it PASS without it).

Run (GPU window): /home/derek/.venv/bin/python m3_adoption_probe.py [turns]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "/home/derek/tensorfold-mom/experiments")
from b5_cache_probe import TRIVIAL_TURNS, WORK_TURNS

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
NESTED = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"
BIG = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
NESTED_NAME = "qwen36-nested"
BIG_NAME = "qwen3.6-35b"
PORT = 8218
OUT = "/home/derek/tensorfold-mom/experiments/results/m3"
REPLIES_FILE = "/home/derek/tensorfold-mom/experiments/results/m2-tf/parent_replies.json"
LOG = "/tmp/m3-adopt-mom.log"
SYSTEM = "You are a concise assistant."


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


def boot(extra_env: dict):
    log = open(LOG, "w")
    env = os.environ.copy()
    env["TF_QWEN_REQUEST_STARTS"] = "1"
    env.update(extra_env)
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT),
                          "--parallel", "2", "--no-sticky",
                          "--model", NESTED, "--name", NESTED_NAME, "--lane", "trivial", "--pin",
                          "--model", BIG, "--name", BIG_NAME, "--lane", "default"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    deadline = time.time() + 900
    while time.time() < deadline:
        if p.poll() is not None:
            print("server died:", open(LOG).read()[-1500:])
            return None
        if health():
            return p
        time.sleep(2)
    print("health timeout:", open(LOG).read()[-1500:])
    p.terminate()
    return None


def kill(p):
    if p is None:
        return
    p.terminate()
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()


def replay(name: str, user_turns, replies) -> list[str]:
    """Teacher-forced replay on one lane; returns the lane's OWN replies."""
    out: list[str] = []
    hist: list[dict] = [{"role": "system", "content": SYSTEM}]
    for i, c in enumerate(user_turns):
        hist.append({"role": "user", "content": c})
        r = post({"model": name, "messages": hist, "max_tokens": 24, "temperature": 0,
                  "stream": False, "chat_template_kwargs": {"enable_thinking": False}})
        got = r["choices"][0]["message"]["content"]
        out.append(got)
        hist.append({"role": "assistant", "content": replies[i]})  # teacher-forced
    return out


def main() -> int:
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    user_turns = (TRIVIAL_TURNS + WORK_TURNS)[:turns]
    replies = json.load(open(REPLIES_FILE))
    if len(replies) < len(user_turns):
        print("need parent_replies.json from the M2-TF run"); return 2
    os.makedirs(OUT, exist_ok=True)

    # A. baseline nested
    p = boot({})
    if p is None:
        return 2
    try:
        base = replay(NESTED_NAME, user_turns, replies)
    finally:
        kill(p)
    time.sleep(4)

    # B. ONE boot with adoption on: the parent fills kept states, then the nested lane
    # answers ONLY the last turn's prompt -- the shape adoption exists for (a hop:
    # long prompt, cold for the nested store, warm in the parent's sink). Replaying
    # every turn self-hits the nested store and adoption can never fire (run 1:
    # hits 7, misses 1 -- the lone cold miss was turn 1, too short vs the deep sink).
    p = boot({"TF_MOM_ADOPT": "1"})
    if p is None:
        return 2
    try:
        _ = replay(BIG_NAME, user_turns, replies)  # parent prefills the whole conversation
        hist = [{"role": "system", "content": SYSTEM}]
        for i, c in enumerate(user_turns):
            hist.append({"role": "user", "content": c})
            if i + 1 < len(user_turns):
                hist.append({"role": "assistant", "content": replies[i]})
        r = post({"model": NESTED_NAME, "messages": hist, "max_tokens": 24, "temperature": 0,
                  "stream": False, "chat_template_kwargs": {"enable_thinking": False}})
        adopt = [r["choices"][0]["message"]["content"]]
    finally:
        kill(p)
    # Only the real thing counts: 'adopted the paired engine's kept state' from adoptMiss.
    # (The parent's 'adopt sink at ...' save lines would otherwise make the gate vacuous.)
    adopted_lines = [l for l in open(LOG) if "adopted the paired" in l.lower()]
    json.dump({"baseline": base, "adopted": adopt, "log_adopt_lines": adopted_lines},
              open(f"{OUT}/g3.json", "w"), indent=1)

    ok = adopt[0] == base[-1]
    if not adopted_lines:
        print("G3: NOT RUN -- server logged no 'adopt' lines (wiring absent?); "
              "refusing to call this PASS")
        return 3
    print(f"G3 decode-resume gate: {'PASS' if ok else 'FAIL'} "
          f"(hop turn vs baseline; adopt log lines: {len(adopted_lines)})")
    if not ok:
        print(f"  baseline: {base[-1]!r}\n  adopted : {adopt[0]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
