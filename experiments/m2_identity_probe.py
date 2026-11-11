#!/usr/bin/env python3
"""M2 identity gate (docs/13): bit-identity of shared-prefix KV rows.

One mom-serve process, two lanes: nested-16l (frozen 35B prefix, trivial,
pinned) and the 35B parent (default). TF_MOM_KVDUMP + TF_QWEN_REQUEST_STARTS=1
so every message-start checkpoint is dumped per lane. The nested lane carries
the parent's chat template + vocab (extract copied them), so both lanes
tokenize the replayed conversation IDENTICALLY — token ids must match per
checkpoint, then the K/V rows of the shared attention layers (3, 7) must be
BIT-identical across engines. Anything else breaks the weight-identity premise
and blocks everything downstream (M3 adoption, M4 zero-copy).

Usage: python3 m2_identity_probe.py [turns]     (default 8)
Exit: 0 PASS / 1 gate FAIL / 2 infra FAIL. Window owner stops production.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
NESTED = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"
BIG = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
PORT = 8215
DUMP = "/home/derek/tensorfold-mom/experiments/results/kvdump-m2"
LOG = "/tmp/m2-nested-mom.log"

sys.path.insert(0, "/home/derek/tensorfold-mom/experiments")
sys.path.insert(0, "/home/derek/tensorfold-mom/src/tensorfold/experiments")
from b5_cache_probe import TRIVIAL_TURNS, WORK_TURNS, TURNS  # shared corpus
import numpy as np
import s1_fit

NESTED_NAME = "qwen36-nested"
BIG_NAME = "qwen3.6-35b"
SHARED_MAX = 9  # frozen prefix layers 0..9; attention members: 3, 7


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


def replay(model, turns):
    msgs = [{"role": "system", "content": "You are a concise assistant."}]
    for i in range(turns):
        user = WORK_TURNS[i // 2][:280] if i % 2 == 1 else TRIVIAL_TURNS[i // 2]
        msgs.append({"role": "user", "content": user})
        r = post({"model": model, "messages": msgs, "max_tokens": 16, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}})
        if not r.get("choices"):
            raise RuntimeError(f"{model} turn {i}: empty reply")
        msgs.append({"role": "assistant",
                     "content": r["choices"][0]["message"]["content"]})
    print(f"lane {model}: {turns} turns replayed")


def main() -> int:
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if os.path.isdir(DUMP):
        subprocess.run(["rm", "-rf", DUMP], check=True)
    env = os.environ.copy()
    env["TF_QWEN_REQUEST_STARTS"] = "1"
    env["TF_MOM_KVDUMP"] = DUMP
    log = open(LOG, "w")
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT),
                          "--parallel", "2", "--no-sticky",
                          "--model", NESTED, "--name", NESTED_NAME, "--lane", "trivial", "--pin",
                          "--model", BIG, "--name", BIG_NAME, "--lane", "default"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    try:
        deadline = time.time() + 900
        while time.time() < deadline:
            if p.poll() is not None:
                print("server died:", open(LOG).read()[-1500:])
                return 2
            if health():
                break
            time.sleep(2)
        else:
            print("health timeout:", open(LOG).read()[-1500:])
            return 2

        for m in (NESTED_NAME, BIG_NAME):
            replay(m, turns)
        time.sleep(3)

        nd = {d.ckpt: d for d in s1_fit.load_model_dir(os.path.join(DUMP, "qwen36-nested-16l"))}
        hd = {d.ckpt: d for d in s1_fit.load_model_dir(os.path.join(DUMP, "qwen36-35b-a3b-nvfp4"))}
        common = sorted(set(nd) & set(hd))
        if len(common) < turns:
            print(f"dump counts low: nested {sorted(nd)} heavy {sorted(hd)} (turns {turns})")
            return 2
        fails = 0
        for c in common:
            n, h = nd[c], hd[c]
            if not np.array_equal(n.token_ids, h.token_ids):
                print(f"ckpt{c}: TOKEN MISMATCH (template/vocab drift — fix serving config, "
                      f"not the gate)")
                fails += 1
                continue
            layers = sorted(set(n.layers) & set(h.layers))
            if any(l > SHARED_MAX for l in n.layers):
                print(f"ckpt{c}: nested dumps layers above the frozen prefix "
                      f"({n.layers}) — tail rows must never enter the gate; comparing shared subset only")
            for l in [x for x in layers if x <= SHARED_MAX]:
                for w in ("K", "V"):
                    if not np.array_equal(n.feats(l, w), h.feats(l, w)):
                        a, b = n.feats(l, w), h.feats(l, w)
                        bad = int((a != b).any(axis=1).sum())
                        print(f"ckpt{c} layer{l} {w}: NOT BIT-IDENTICAL ({bad}/{a.shape[0]} rows differ)")
                        fails += 1
            print(f"ckpt{c}: {len(n.token_ids)} tok, shared attn layers "
                  f"{[x for x in layers if x <= SHARED_MAX]} checked")
        shared = [l for l in sorted(set(nd[common[0]].layers) & set(hd[common[0]].layers)) if l <= SHARED_MAX]
        if not shared:
            print("M2: FAIL — no shared attention layers (3,7) found in nested dumps")
            return 1
        print(f"M2 identity gate: {'PASS' if fails == 0 else f'FAIL ({fails})'} — "
              f"{len(common)} checkpoints, layers {shared}, K+V bit-identical required")
        return 0 if fails == 0 else 1
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
