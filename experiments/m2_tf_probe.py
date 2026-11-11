#!/usr/bin/env python3
"""M2 identity gate, teacher-forced variant.

The plain M2 probe (m2_identity_probe.py) replays each lane's OWN replies;
once the degenerate pre-distillation nested lane pollutes its history with
gibberish, the lanes are no longer fed the same tokens and token-mismatch is
a probe artifact, not an identity failure. This variant teacher-forces BOTH
lanes with the same fixed history (the parent lane's real replies, captured
once into results/m2-tf/parent_replies.json), so both prompt strings are
byte-identical at every turn. Any token mismatch here is a real template/
vocab fault.

Compares K and V rows at shared full-attention layers BELOW the frozen
prefix (layers 3, 7) for bit-identity across checkpoints. Tail layers (11,
15) are deliberately excluded.

Launch pattern mirrors m2_identity_probe.py (verified working on device):
tensorfold-native mom-serve --model <dir> --name <id> --lane <lane> [--pin].
KVDump writer names subdirs by checkpoint model-dir basename.

Run (GPU window): /home/derek/.venv/bin/python m2_tf_probe.py [turns]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "/home/derek/tensorfold-mom/experiments")
sys.path.insert(0, "/home/derek/tensorfold-mom/src/tensorfold/experiments")
import numpy as np
from b5_cache_probe import TRIVIAL_TURNS, WORK_TURNS  # shared corpus
import s1_fit

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
NESTED = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"
BIG = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
NESTED_NAME = "qwen36-nested"
BIG_NAME = "qwen3.6-35b"
PORT = 8216
DUMP = "/home/derek/tensorfold-mom/experiments/results/m2-tf/kvdump"
OUT = "/home/derek/tensorfold-mom/experiments/results/m2-tf"
REPLIES_FILE = f"{OUT}/parent_replies.json"
LOG = "/tmp/m2-tf-mom.log"
SHARED = [3, 7]
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


def chat(model: str, msgs: list[dict]) -> str | None:
    try:
        r = post({"model": model, "messages": msgs, "max_tokens": 24, "temperature": 0,
                  "stream": False, "chat_template_kwargs": {"enable_thinking": False}})
        return r["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"chat {model} failed: {e}")
        return None


def boot(args: list[str], env_extra: dict):
    log = open(LOG, "w")
    env = os.environ.copy()
    env.update(env_extra)
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT)] + args,
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


def main() -> int:
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    user_turns = (TRIVIAL_TURNS + WORK_TURNS)[:turns]

    # Phase 0: capture the parent lane's REAL replies once (single-engine boot),
    # so the teacher-forced history is natural 35B text, identical for both lanes.
    if not os.path.isfile(REPLIES_FILE):
        os.makedirs(OUT, exist_ok=True)
        p = boot(["--model", BIG, "--name", BIG_NAME, "--lane", "default"],
                 {"TF_QWEN_REQUEST_STARTS": "1"})
        if p is None:
            return 2
        try:
            cap: list[dict] = [{"role": "system", "content": SYSTEM}]
            reps: list[str] = []
            for i, c in enumerate(user_turns):
                cap.append({"role": "user", "content": c})
                r = chat(BIG_NAME, cap)
                if r is None:
                    r = c[:60]  # last-resort filler (logged)
                    print(f"capture turn {i+1} failed; filler used")
                reps.append(r)
                cap.append({"role": "assistant", "content": r})
        finally:
            kill(p)
        json.dump(reps, open(REPLIES_FILE, "w"), indent=1)
        print(f"captured {len(reps)} parent replies -> {REPLIES_FILE}")
        time.sleep(5)

    replies = json.load(open(REPLIES_FILE))
    if len(replies) < len(user_turns):
        print(f"{REPLIES_FILE} has {len(replies)} replies, need {len(user_turns)}")
        return 2

    if os.path.isdir(DUMP):
        subprocess.run(["rm", "-rf", DUMP], check=True)
    p = boot(["--parallel", "2", "--no-sticky",
              "--model", NESTED, "--name", NESTED_NAME, "--lane", "trivial", "--pin",
              "--model", BIG, "--name", BIG_NAME, "--lane", "default"],
             {"TF_QWEN_REQUEST_STARTS": "1", "TF_MOM_KVDUMP": DUMP})
    if p is None:
        return 2
    try:
        for name in (NESTED_NAME, BIG_NAME):
            hist: list[dict] = [{"role": "system", "content": SYSTEM}]
            for i, c in enumerate(user_turns):
                hist.append({"role": "user", "content": c})
                if chat(name, hist) is None:
                    print(f"lane {name} turn {i+1} request failed")
                    return 1
                hist.append({"role": "assistant", "content": replies[i]})
            print(f"lane {name}: {len(user_turns)} teacher-forced turns replayed")
        time.sleep(3)
    finally:
        kill(p)

    failures = 0
    for k in range(len(user_turns) + 1):
        try:
            da = s1_fit.load_dump(os.path.join(DUMP, "qwen36-nested-16l", f"ckpt{k}.bin"),
                                  os.path.join(DUMP, "qwen36-nested-16l", f"ckpt{k}.json"))
            db = s1_fit.load_dump(os.path.join(DUMP, "qwen36-35b-a3b-nvfp4", f"ckpt{k}.bin"),
                                  os.path.join(DUMP, "qwen36-35b-a3b-nvfp4", f"ckpt{k}.json"))
        except FileNotFoundError as e:
            print(f"ckpt{k}: missing dump ({e})")
            failures += 1
            continue
        if da.tokens != db.tokens:
            for pos, (x, y) in enumerate(zip(da.tokens, db.tokens)):
                if x != y:
                    lo = max(0, pos - 4)
                    print(f"ckpt{k}: TOKEN MISMATCH at pos {pos}: "
                          f"nested {da.tokens[lo:pos+5]} | parent {db.tokens[lo:pos+5]}")
                    break
            else:
                print(f"ckpt{k}: token LENGTH differs {len(da.tokens)} vs {len(db.tokens)}")
            failures += 1
            continue
        layers = sorted(set(da.layers) & set(db.layers) & set(SHARED))
        if not layers:
            print(f"ckpt{k}: no shared layers present")
            failures += 1
            continue
        row_bad = 0
        for L in layers:
            for kind in ("K", "V"):
                ra, rb = da.rows.get((L, kind)), db.rows.get((L, kind))
                if ra is None or rb is None or ra.shape != rb.shape:
                    print(f"ckpt{k} L{L}{kind}: shape/presence mismatch")
                    row_bad += 1
                    continue
                if not np.array_equal(ra, rb):
                    nd = int((ra != rb).sum())
                    print(f"ckpt{k} L{L}{kind}: {nd}/{ra.size} values differ "
                          f"(max |d|={np.abs(ra.astype(np.float32)-rb.astype(np.float32)).max():.4g})")
                    row_bad += 1
        print(f"ckpt{k}: {len(da.tokens)} tok, layers {layers}, "
              f"{'IDENTICAL' if row_bad == 0 else f'{row_bad} ROW-IDENTITY FAILURES'}")
        failures += row_bad

    print(f"\nM2-TF identity gate: {'PASS' if failures == 0 else f'FAIL ({failures})'} — "
          f"{len(user_turns)+1} checkpoints, layers {SHARED}, K+V bit-identical required")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
