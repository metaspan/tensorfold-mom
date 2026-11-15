#!/usr/bin/env python3
"""BL-6T tuned thin-decode probe — scenario D_tuned, pre-registered in
docs-mom/gates/20261113-bl6t-tuned-decode.md BEFORE any run. Single arm, exactly one
difference from bl6_gate A_stock: the boot line (--parallel 4 --context 8192,
TF_MOM_TRIVIAL_PRIORITY=1). Thresholds are fixed: mean tok_s >= 154.0, every variant
>= 138.0. Run ONCE whatever the outcome; no config fishing afterwards.
"""
import json
import os
import subprocess
import time
import urllib.request

import m3_adoption_probe as P

MAXTOK = int(os.environ.get("BL6_MAXTOK", "96"))
VARIANTS = int(os.environ.get("BL6_VARIANTS", "3"))
PREFIX = int(os.environ.get("BL6_PREFIX", "4"))
MEAN_BAR = 154.0
GUARD_BAR = 138.0
OUT_DIR = os.environ.get("BL6T_OUT", "/home/derek/tensorfold-mom/experiments/results/bl6t")
os.makedirs(OUT_DIR, exist_ok=True)
EXTRA_ENV = {"TF_QWEN_KV_PAGES": "64", "TF_QWEN_KV_PAGES_ALL": "1", "TF_MOM_TRIVIAL_PRIORITY": "1"}


def boot_tuned(truncate=True):
    log = "/tmp/bl6t-tuned.log"
    env = os.environ.copy()
    env["TF_QWEN_REQUEST_STARTS"] = "1"
    env.update(EXTRA_ENV)
    P.LOG = log
    argv = [P.BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(P.PORT),
            "--parallel", "4", "--context", "8192", "--no-sticky",
            "--model", P.NESTED, "--name", P.NESTED_NAME, "--lane", "trivial", "--pin",
            "--model", P.BIG, "--name", P.BIG_NAME, "--lane", "default"]
    with open(log, "w" if truncate else "a") as lg:
        p = subprocess.Popen(argv, stdout=lg, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    deadline = time.time() + 900
    while time.time() < deadline:
        if p.poll() is not None:
            print("server died:", open(log, errors="replace").read()[-1500:])
            return None
        if P.health():
            return p
        time.sleep(2)
    print("health timeout")
    p.terminate()
    return None


def post_stream(payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{P.PORT}/v1/chat/completions",
        data=json.dumps({**payload, "stream": True, "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            d = json.loads(body)
            if d.get("usage"):
                tokens = d["usage"].get("completion_tokens", tokens)
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return (ttft if ttft is not None else total), total, tokens


def main() -> int:
    if not os.path.exists(P.REPLIES_FILE):
        print("need parent_replies.json at", P.REPLIES_FILE)
        return 2
    replies = json.load(open(P.REPLIES_FILE))
    user_turns = (P.TRIVIAL_TURNS + P.WORK_TURNS)[:PREFIX + 1]
    p = boot_tuned()
    if p is None:
        print("ARM D_tuned boot FAILED")
        return 2
    rows = []
    try:
        for v in range(VARIANTS):
            sys_prompt = P.SYSTEM + f"\n[bl6t variant {v}]"
            hist = [{"role": "system", "content": sys_prompt}]
            for c in user_turns[:PREFIX]:
                hist.append({"role": "user", "content": c})
                r = P.post({"model": P.BIG_NAME, "messages": hist, "max_tokens": 24,
                            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
                hist.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
            hist.append({"role": "user", "content": user_turns[PREFIX]})
            ttft, total, toks = post_stream(
                {"model": P.NESTED_NAME, "messages": hist, "max_tokens": MAXTOK,
                 "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
            dec = total - ttft
            rows.append({"variant": v, "ttft_s": round(ttft, 4), "decode_s": round(dec, 4),
                         "tokens": toks, "tok_s": round(toks / dec, 1) if dec > 0.05 and toks else None})
            print(json.dumps(rows[-1]), flush=True)
    finally:
        P.kill(p)
    vals = [r["tok_s"] for r in rows if r["tok_s"]]
    mean = sum(vals) / max(1, len(vals))
    ok_mean = mean >= MEAN_BAR
    ok_guard = bool(vals) and min(vals) >= GUARD_BAR
    verdict = "GO" if (ok_mean and ok_guard) else "NO-GO"
    print(f"\nBL6T tuned decode: mean {mean:.1f} tok/s (bar >= {MEAN_BAR}), min {min(vals) if vals else 0} (guard >= {GUARD_BAR})")
    print(f"BL6T verdict: {verdict}")
    out = {"scenario": "D_tuned", "boot": {"parallel": 4, "context": 8192,
                                           "env": EXTRA_ENV},
           "rows": rows, "mean_tok_s": round(mean, 1), "min_tok_s": min(vals) if vals else 0,
           "bars": {"mean": MEAN_BAR, "guard": GUARD_BAR}, "verdict": verdict}
    path = os.path.join(OUT_DIR, f"bl6t-{time.strftime('%Y%m%d-%H%M')}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
