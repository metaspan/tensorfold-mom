#!/usr/bin/env python3
"""B5 cache-effectiveness probe (recovered from docs/09).

Single mom-serve, both lanes resident, --no-sticky free re-routing,
greedy, thinking off. Each turn sends the full history as the prompt;
the client streams so TTFT is time-to-first-content-delta, and asks the
server for exact usage token counts (stream_options.include_usage).
The conversation is replayed twice: pass 1 cold, pass 2 warm. Per-turn
prefill reuse is read from the engine's own `prompt cache: N tokens,
resumed at R, kept K` line.

Usage: python3 b5_cache_probe.py [turns=28] [--arm A|B]

Arms (spike-2 rerun, per the shared-cache plan P0.2):
  A (repro)  : stock lanes, default env            -> resumed at 0, kept 0
  B (opt-in) : TF_QWEN_REQUEST_STARTS=1            -> same-lane reuse

Env overrides: TFM_BIN, TFM_SMALL, TFM_BIG, TFM_SMALL_NAME, TFM_BIG_NAME,
               TFM_PORT, TFM_OUT, TFM_LOG, TFM_CONTEXT
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BIN = os.environ.get("TFM_BIN",
                     "/home/derek/tensorfold-mom/zig-out/native/bin/tensorfold-native")
SMALL = os.environ.get("TFM_SMALL",
                       "/home/derek/.hermes/models/tensorfold-mom/qwen35-2b")
BIG = os.environ.get("TFM_BIG",
                     "/home/derek/.hermes/models/tensorfold-mom/qwen36-35b-a3b-nvfp4")
SMALL_NAME = os.environ.get("TFM_SMALL_NAME", "qwen3.5-2b")
BIG_NAME = os.environ.get("TFM_BIG_NAME", "qwen3.6-35b")
PORT = int(os.environ.get("TFM_PORT", "8218"))
OUT = os.environ.get("TFM_OUT",
                     "/home/derek/tensorfold-mom/experiments/results/b5")
LOG = os.environ.get("TFM_LOG", "/tmp/b5-mom.log")
CONTEXT = os.environ.get("TFM_CONTEXT", "4096")
SYSTEM = "You are a concise assistant."

# 28-turn alternating conversation: 14 trivial (-> small lane) +
# 14 work (-> big lane). Even index = trivial, odd index = work.
TRIVIAL_TURNS = [
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
    "What is the capital of Japan?",
    "How many continents are there?",
    "What is 2 to the power of 10?",
    "Which gas do plants absorb from the atmosphere?",
]
WORK_TURNS = [
    "Write a Python function that parses an .ics calendar file and returns "
    "the busiest week; explain your parsing choices and handle recurring events.",
    "Explain the difference between optimistic and pessimistic concurrency "
    "control in databases, then recommend one for a hotel booking system.",
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
    "Implement an LRU cache in Python with O(1) get and put, including eviction.",
    "What is the CAP theorem, and what does it imply for a distributed key-value "
    "store under a network partition?",
    "Write a recursive descent parser for a small arithmetic grammar with +, *, "
    "and parentheses; show the AST for '2 + 3 * 4'.",
    "Explain the difference between a process and a thread, and when you'd use "
    "each for a CPU-bound workload.",
]
TURNS = len(TRIVIAL_TURNS) + len(WORK_TURNS)  # 28


def post_stream(payload, timeout=300):
    """Returns (ttft_s, total_s, completion_tokens, content)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({**payload, "stream": True,
                         "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    pieces = []
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
                pc = (ch.get("delta") or {}).get("content")
                if pc and ttft is None:
                    ttft = time.perf_counter() - t0
                if pc:
                    pieces.append(pc)
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    return ttft, total, tokens, "".join(pieces)


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
            names = {m.get("id") for m in json.loads(r.read().decode()).get("data", [])}
        return SMALL_NAME in names and BIG_NAME in names
    except Exception:
        return False


def boot(extra_env=None, log_path=LOG, truncate=True):
    logf = open(log_path, "w" if truncate else "a")
    env = os.environ.copy()
    env["TF_QWEN_REQUEST_STARTS"] = "1"  # arm B default; arm A caller deletes it
    if extra_env:
        env.update(extra_env)
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT),
                          "--parallel", "1", "--no-sticky", "--context", CONTEXT,
                          "--model", SMALL, "--name", SMALL_NAME, "--lane", "trivial", "--pin",
                          "--model", BIG, "--name", BIG_NAME, "--lane", "default"],
                         stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    deadline = time.time() + 900
    while time.time() < deadline:
        if p.poll() is not None:
            print("server died:", open(log_path, errors="replace").read()[-1500:])
            return None
        if health():
            return p
        time.sleep(2)
    print("health timeout:", open(log_path, errors="replace").read()[-1500:])
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


def cache_lines(path):
    """Return list of dicts {prompt, resumed_at, kept} parsed from log."""
    out = []
    pat = re.compile(r"prompt cache:\s*(\d+)\s*tokens,\s*resumed at\s*(\d+),\s*kept\s*(\d+)")
    for line in open(path, errors="replace"):
        if "prompt cache:" not in line:
            continue
        m = pat.search(line)
        if m:
            out.append({"prompt": int(m.group(1)),
                        "resumed_at": int(m.group(2)),
                        "kept": int(m.group(3))})
    return out


def replay_pass(user_turns, pass_label, base_dir, log_path, start_line=0):
    """Teacher-forced replay of one full pass; returns per-turn rows.

    Reads cache lines from log_path[start_line:] and pairs one line per turn.
    """
    rows = []
    hist = [{"role": "system", "content": SYSTEM}]
    os.makedirs(base_dir, exist_ok=True)
    turn_log = open(os.path.join(base_dir, f"{pass_label}.jsonl"), "w")
    for i, c in enumerate(user_turns):
        lane = BIG_NAME if i % 2 == 1 else SMALL_NAME
        hist.append({"role": "user", "content": c})
        ttft, total, toks, content = post_stream(
            {"model": lane, "messages": hist, "max_tokens": 24, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False}})
        hist.append({"role": "assistant", "content": content})
        rows.append({"turn": i, "lane": lane, "ttft_s": round(ttft, 4),
                     "e2e_s": round(total, 4), "completion_tokens": toks,
                     "prompt_tokens": None, "content": content})
        turn_log.write(json.dumps(rows[-1]) + "\n")
    turn_log.close()
    lines = cache_lines(log_path)[start_line:]
    for r, cl in zip(rows, lines):
        r["prompt_tokens"] = cl["prompt"]
        r["resumed_at"] = cl["resumed_at"]
        r["kept"] = cl["kept"]
    return rows, len(lines)


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    s = sorted(sorted_vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (k - f) * (s[c] - s[f]))


def analyze(rows, label):
    """Per-lane aggregate: reuse %, prefill-saved %, ttft p50."""
    out = {}
    for lane in (SMALL_NAME, BIG_NAME):
        lr = [r for r in rows if r["lane"] == lane]
        # skip turn 0 (cold by construction) for reuse %
        measurable = [r for r in lr if r["turn"] > 0]
        reuse_turns = [r for r in measurable if r.get("resumed_at", 0) > 0]
        reused_tokens = sum(r.get("resumed_at", 0) for r in measurable)
        total_tokens = sum(r.get("prompt_tokens", 0) or 0 for r in measurable)
        ttfts = [r["ttft_s"] for r in lr]
        out[lane] = {
            "n": len(lr),
            "reuse_turns": len(reuse_turns),
            "reuse_turns_measurable": len(measurable),
            "reuse_pct": round(100.0 * len(reuse_turns) / max(len(measurable), 1), 1),
            "prefill_saved_pct": round(100.0 * reused_tokens / max(total_tokens, 1), 1),
            "ttft_p50_ms": round(percentile(ttfts, 50) * 1000, 1),
            "ttft_p95_ms": round(percentile(ttfts, 95) * 1000, 1),
        }
    return out


def main() -> int:
    args = sys.argv[1:]
    turns = TURNS
    arm = "B"
    i = 0
    while i < len(args):
        if args[i] == "--arm" and i + 1 < len(args):
            arm = args[i + 1]
            i += 2
        else:
            try:
                turns = int(args[i])
            except ValueError:
                pass
            i += 1
    if turns > TURNS:
        turns = TURNS
    if arm not in ("A", "B"):
        print("arm must be A or B"); return 2

    user_turns = [WORK_TURNS[i // 2][:280] if i % 2 == 1 else TRIVIAL_TURNS[i // 2]
                  for i in range(turns)]
    os.makedirs(OUT, exist_ok=True)

    env_extra = {} if arm == "B" else {"TF_QWEN_REQUEST_STARTS": ""}

    # Single process, both lanes resident: cold pass then warm pass (docs/09).
    mom_log = os.path.join(OUT, f"mom-arm{arm}.log")
    p = boot(extra_env=env_extra, log_path=mom_log)
    if p is None:
        return 2
    try:
        cold, n_cold = replay_pass(user_turns, f"cold-{arm}", OUT, mom_log)
        warm, _ = replay_pass(user_turns, f"warm-{arm}", OUT, mom_log, start_line=n_cold)
    finally:
        kill(p)

    cold_agg = analyze(cold, "cold")
    warm_agg = analyze(warm, "warm")

    # Gate (arm B pre-registered bar): same-lane warm-pass reuse >= 85% cold
    # and >= 99% warm. Arm A: reuse ~0% on both.
    report = {
        "arm": arm, "turns": turns, "context": CONTEXT,
        "cold": cold, "warm": warm,
        "agg": {"cold": cold_agg, "warm": warm_agg},
    }
    # Verdict
    if arm == "B":
        small_warm = warm_agg[SMALL_NAME]
        big_warm = warm_agg[BIG_NAME]
        pass_reuse = (small_warm["reuse_pct"] >= 99.0 and big_warm["reuse_pct"] >= 99.0)
        report["verdict"] = "PASS" if pass_reuse else "FAIL"
        report["bar"] = "warm-pass same-lane reuse >= 99% on both lanes"
    else:
        small_cold = cold_agg[SMALL_NAME]
        big_cold = cold_agg[BIG_NAME]
        report["verdict"] = "PASS" if (small_cold["reuse_pct"] == 0.0 and
                                       big_cold["reuse_pct"] == 0.0) else "FAIL"
        report["bar"] = "cold pass reuse == 0% on both lanes (no opt-in)"

    path = os.path.join(OUT, f"b5-arm{arm}-{time.strftime('%Y%m%d-%H%M')}.json")
    json.dump(report, open(path, "w"), indent=1)
    print(json.dumps(report["agg"], indent=1))
    print("VERDICT:", report["verdict"], "|", report["bar"])
    print("wrote", path)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
