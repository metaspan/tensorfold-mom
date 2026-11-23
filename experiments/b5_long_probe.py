#!/usr/bin/env python3
"""B5-long probe (P1): size the escalation re-prefill prize.

Three-turn conversation per case:
  1. Turn 1: short trivial -> thin lane.
  2. Turn 2: long block of L tokens -> thin lane (absorbs the unseen block).
  3. Turn 3: forced to the heavy lane.

Measure the heavy lane's prefill of the delta in turn 3 (ttft + prompt tokens
from the engine's `prompt cache:` log line).

Sweep: L in {512, 2k, 4k, 16k, 32k}, 5 blocks per L. 32k is best-effort
(this GB10 build crashes the server at 32k context with two lanes; sweep
continues at smaller L).

Config: arm B (TF_QWEN_REQUEST_STARTS=1).

Pre-registered bar: p90 heavy delta-prefill < 500 ms AND < 15% of the
escalation turn's e2e time -> close cross-model translation for the MVP
(P4 not started).

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
import urllib.error
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
                     "/home/derek/tensorfold-mom/experiments/results/b5-long")
LOG = os.environ.get("TFM_LOG", "/tmp/b5-long-mom.log")
CONTEXT = os.environ.get("TFM_CONTEXT", "16384")
SYSTEM = "You are a concise assistant."

LS = [int(x) for x in os.environ.get("TFM_LS", "512,2048,4096,16384,32768").split(",")]
BLOCKS_PER_L = 5

BASE_TEXTS = [
    ("Write a design document for a distributed key-value store. Cover the "
     "replication protocol, consensus algorithm, failure detector, read repair, "
     "anti-entropy, hinted handoff, read consistency levels, write consistency "
     "levels, quorum arithmetic, sloppy quorum, read repair vs anti-entropy, "
     "membership, gossip, hinted handoff, read repair, entropy, vector clocks, "
     "dotted version vectors, last-write-wins, causal consistency, session "
     "guarantees, monotonic reads, monotonic writes, read your writes, "
     "writes follow reads, partition tolerance, availability, CAP tradeoffs, "
     "PACELC, hinted handoff, read repair, anti-entropy, Merkle trees, "
     "compaction, memtable, SSTable, bloom filter, block cache, write-ahead "
     "log, fsync, group commit, durable, atomic, isolated, consistent, "
     "serializable, snapshot isolation, serializability, two-phase commit, "
     "three-phase commit, Paxos, Raft, Zab, leader election, log replication, "
     "snapshot, install snapshot, dynamic membership, joint consensus. "),
    ("Implement and explain a B+ tree storage engine. Cover node structure, "
     "internal vs leaf nodes, split, merge, redistribution, range scan, point "
     "query, insert, delete, underflow, overflow, fill factor, page size, "
     "buffer pool, LRU, clock replacement, dirty page, write ahead log, "
     "redo, undo, checkpoint, recovery, ARIES, lock coupling, crabbing, "
     "index latch, B-link tree, concurrency, logging, logical vs physical "
     "logging, page checksum, torn page, write ahead logging, group commit, "
     "fsync, direct I/O, asynchronous I/O, io_uring, mmap, page cache, "
     "buffer cache, read ahead, prefetch, sequential scan, index scan, "
     "index-only scan, heap scan, tuple, slot, page, header, free space map, "
     "visibility map, vacuum, autovacuum, transaction id, multiversion "
     "concurrency control, snapshot isolation, serializable snapshot isolation, "
     "predicate lock, index lock, tuple lock, deadlock, wait-for graph, "
     "timeout, abort, restart. "),
    ("Describe the internet protocol stack from the physical layer to the "
     "application layer. Ethernet, MAC address, ARP, ND, IPv4, IPv6, subnetting, "
     "CIDR, longest prefix match, routing, forwarding, BGP, OSPF, IS-IS, RIP, "
     "distance vector, link state, Dijkstra, Bellman-Ford, path vector, "
     "route aggregation, route flap, dampening, TCP, UDP, QUIC, SCTP, "
     "three-way handshake, slow start, congestion avoidance, fast retransmit, "
     "fast recovery, cubic, BBR, Reno, NewReno, selective repeat, go-back-N, "
     "sliding window, flow control, receive window, congestion window, "
     "bandwidth delay product, head-of-line blocking, multiplexing, stream, "
     "datagram, ICMP, IGMP, DHCP, DNS, recursive resolver, iterative resolver, "
     "cache, TTL, authoritative, zone transfer, HTTP/1.1, HTTP/2, HTTP/3, "
     "TLS, handshake, certificate, CA, key exchange, AES, ChaCha20, "
     "Poly1305, RSA, ECDSA, Diffie-Hellman, elliptic curve, post-quantum. "),
    ("Explain how a relational database executes a SQL query. Parser, lexer, "
     "grammar, abstract syntax tree, semantic analysis, binder, resolver, "
     "catalog, metadata, statistics, histogram, cardinality estimation, "
     "selectivity, join order, dynamic programming, exhaustive search, "
     "heuristic, greedy, hash join, merge join, nested loop join, index "
     "nested loop join, sort merge join, aggregation, hash aggregate, stream "
     "aggregate, sort, external merge sort, top-N, limit, offset, window "
     "function, range, rows, partition by, order by, frame, aggregate, "
     "projection, filter, pushdown, predicate pushdown, projection pushdown, "
     "materialization, pipelining, row mode, batch mode, vectorized execution, "
     "SIMD, columnar, row store, OLTP, OLAP, HTAP, ACID, BASE, transaction, "
     "begin, commit, rollback, savepoint, isolation level, read uncommitted, "
     "read committed, repeatable read, serializable, snapshot, phantom, "
     "non-repeatable read, dirty read, dirty write, write skew, read skew, "
     "lost update, serialization anomaly, write skew anomaly. "),
    ("Describe how a modern optimizing compiler turns source code into machine "
     "code. Lexer, parser, AST, semantic analysis, type checking, type "
     "inference, Hindley-Milner, unification, middle end, intermediate "
     "representation, three-address code, SSA, static single assignment, "
     "dominance, dominator tree, dominance frontier, phi function, "
     "optimization pass, constant folding, constant propagation, copy "
     "propagation, dead code elimination, dead store elimination, common "
     "subexpression expression, value numbering, GVN, SCCVN, loop invariant "
     "code motion, induction variable, strength reduction, loop unrolling, "
     "loop vectorization, loop interchange, loop tiling, loop fusion, loop "
     "fission, loop distribution, software pipelining, modulo scheduling, "
     "register allocation, graph coloring, Chaitin, Briggs, linear scan, "
     "spilling, coalescing, live range splitting, instruction selection, "
     "instruction scheduling, instruction emission, peephole optimization, "
     "dead code, unreachable code, branch prediction, return address, stack "
     "frame, calling convention, ABI, calling convention, x86-64, AArch64, "
     "RISC-V, calling convention, register argument, stack argument, return "
     "value, frame pointer, stack pointer, link register, program counter. "),
]


def make_block(target_tokens: int, base: str, max_tokens=None) -> str:
    # Empirical on this tokenizer family: ~1.75 tokens/word for this text.
    if max_tokens is not None and target_tokens > max_tokens:
        target_tokens = max_tokens
    words_needed = max(1, int(target_tokens / 1.75))
    words = base.split()
    full = (words * (words_needed // len(words) + 1))[:words_needed]
    return " ".join(full)


def post_stream(payload, timeout=300):
    """Returns (ttft_s, total_s, completion_tokens, content). Raises on HTTP error."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({**payload, "stream": True,
                         "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    completion = 0
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
                completion = d["usage"].get("completion_tokens", completion)
            for ch in d.get("choices", []):
                pc = (ch.get("delta") or {}).get("content")
                if pc and ttft is None:
                    ttft = time.perf_counter() - t0
                if pc:
                    pieces.append(pc)
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    return ttft, total, completion, "".join(pieces)


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
    env["TF_QWEN_REQUEST_STARTS"] = "1"
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
    out = []
    pat = re.compile(r"prompt cache:\s*(\d+)\s*tokens,\s*resumed at\s*(\d+),\s*kept\s*(\d+)")
    for line in open(path, errors="replace"):
        if "prompt cache:" in line:
            m = pat.search(line)
            if m:
                out.append({"prompt": int(m.group(1)),
                            "resumed_at": int(m.group(2)),
                            "kept": int(m.group(3))})
    return out


def percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (k - f) * (s[c] - s[f]))


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    p = boot()
    if p is None:
        return 2
    results = []
    line_base = 0
    try:
        for L in LS:
            for b in range(BLOCKS_PER_L):
                base = BASE_TEXTS[b % len(BASE_TEXTS)]
                # Leave context headroom for the template + turn 1/3 overhead.
                block = make_block(L, base, max_tokens=int(CONTEXT) - 160)
                short = "What is the capital of France?"
                case = {"L": L, "block": b}
                try:
                    hist = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": short}]
                    t1_ttft, t1_total, t1_tok, t1_content = post_stream(
                        {"model": SMALL_NAME, "messages": hist, "max_tokens": 16,
                         "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}})
                    hist.append({"role": "assistant", "content": t1_content})

                    hist.append({"role": "user", "content": block})
                    t2_ttft, t2_total, t2_tok, t2_content = post_stream(
                        {"model": SMALL_NAME, "messages": hist, "max_tokens": 16,
                         "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}})
                    hist.append({"role": "assistant", "content": t2_content})

                    hist.append({"role": "user",
                                 "content": "Summarize the above in one sentence."})
                    t3_ttft, t3_total, t3_tok, t3_content = post_stream(
                        {"model": BIG_NAME, "messages": hist, "max_tokens": 24,
                         "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}})

                    # Pair turns with cache log lines (one per turn, in order).
                    # Server log is buffered; poll until 3 new lines appear.
                    wanted = line_base + 3
                    cl = []
                    for _ in range(100):
                        all_lines = cache_lines(LOG)
                        if len(all_lines) >= wanted:
                            cl = all_lines[line_base:wanted]
                            break
                        time.sleep(0.2)
                    if len(cl) < 3:
                        case.update({"status": "error: cache lines missing"})
                        results.append(case)
                        continue
                    line_base = wanted
                    t1_cl, t2_cl, t3_cl = cl
                    case.update({
                        "thin_t1_prompt_tokens": t1_cl["prompt"],
                        "thin_t1_resumed_at": t1_cl["resumed_at"],
                        "thin_t2_prompt_tokens": t2_cl["prompt"],
                        "thin_t2_resumed_at": t2_cl["resumed_at"],
                        "thin_t2_ttft_ms": round(t2_ttft * 1000, 1),
                        "thin_t2_prefill_saved_pct": round(
                            100.0 * t2_cl["resumed_at"] / max(t2_cl["prompt"], 1), 1),
                        "heavy_t3_delta_tokens": t3_cl["prompt"],
                        "heavy_t3_resumed_at": t3_cl["resumed_at"],
                        "heavy_t3_kept": t3_cl["kept"],
                        "heavy_t3_ttft_ms": round(t3_ttft * 1000, 1),
                        "heavy_t3_e2e_ms": round(t3_total * 1000, 1),
                        "status": "ok",
                    })
                except (urllib.error.HTTPError, urllib.error.URLError,
                        ConnectionError, OSError) as e:
                    case.update({"status": f"error: {e}"})
                results.append(case)
                print(f"L={L} block={b}: {case.get('status','ok')} "
                      f"thin_t2={case.get('thin_t2_prompt_tokens')}tok "
                      f"{case.get('thin_t2_ttft_ms')}ms | "
                      f"heavy_t3_delta={case.get('heavy_t3_delta_tokens')}tok "
                      f"ttft={case.get('heavy_t3_ttft_ms')}ms e2e={case.get('heavy_t3_e2e_ms')}ms",
                      flush=True)
    finally:
        kill(p)

    ok = [r for r in results if r.get("status") == "ok"]
    all_ttft = [r["heavy_t3_ttft_ms"] for r in ok]
    all_delta = [r["heavy_t3_delta_tokens"] for r in ok]
    agg = {
        "n_cases": len(results), "n_ok": len(ok),
        "context": CONTEXT,
        "heavy_delta_prefill_p50_ms": round(percentile(all_ttft, 50), 1),
        "heavy_delta_prefill_p90_ms": round(percentile(all_ttft, 90), 1),
        "heavy_delta_prefill_p99_ms": round(percentile(all_ttft, 99), 1),
        "heavy_delta_tokens_p50": round(percentile(all_delta, 50), 0),
        "heavy_delta_tokens_p90": round(percentile(all_delta, 90), 0),
        "e2e_p50_ms": round(percentile([r["heavy_t3_e2e_ms"] for r in ok], 50), 1),
    }
    per_L = {}
    for L in LS:
        rows = [r for r in ok if r["L"] == L]
        per_L[L] = {
            "cases": len(rows),
            "p90_ttft_ms": round(percentile([r["heavy_t3_ttft_ms"] for r in rows], 90), 1),
            "p90_delta_tokens": round(percentile([r["heavy_t3_delta_tokens"] for r in rows], 90), 0),
            "thin_t2_prefill_saved_p50_pct": round(
                percentile([r["thin_t2_prefill_saved_pct"] for r in rows], 50), 1),
        }

    p90 = agg["heavy_delta_prefill_p90_ms"]
    p90_e2e = agg["e2e_p50_ms"] * 0.15
    bar_ok = (p90 < 500) and (p90 < p90_e2e)
    agg["bar_p90_under_500ms"] = p90 < 500
    agg["bar_p90_under_15pct_e2e"] = p90 < p90_e2e
    agg["bar_15pct_e2e_ms"] = round(p90_e2e, 1)
    agg["verdict"] = ("PASS (close translation for MVP)" if bar_ok
                      else "P4-eligible (bar not met)")

    report = {"agg": agg, "per_L": per_L, "results": results}
    path = os.path.join(OUT, f"b5-long-{time.strftime('%Y%m%d-%H%M')}.json")
    json.dump(report, open(path, "w"), indent=1)
    print(json.dumps(agg, indent=1))
    print("per_L:", json.dumps(per_L))
    print("VERDICT:", agg["verdict"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
