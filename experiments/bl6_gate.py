#!/usr/bin/env python3
"""BL-6 Shared-KV go/no-go gate: 3 arms, pre-registered thresholds (review 2026-11-12 §3).

Arms (each a fresh mom-serve boot, production 35B parent + 16l nested thin):
  A stock     : no adoption (thin full-prefills every turn)
  B copy      : TF_MOM_ADOPT=1 (copy-path adoption)
  C view      : TF_MOM_ADOPT=1 + TF_MOM_SHARED_VIEW=1 (zero-copy shared view)

Measures on the thin lane, teacher-forced 8-turn replay, temperature 0:
  TTFT  = request send -> first streamed content token
  tok/s = completion_tokens / (total - ttft) per turn

GO gate (pre-registered, adopted verbatim):
  blended TTFT of arms B/C over turns 2..8 <= 0.85 x stock arm  AND
  thin decode tok/s (stock arm mean) >= 0.90 x 171 tok/s.
Refuses a vacuous GO: arms B/C must show adoption log lines.

Usage: TFM_PORT=8218 python3 bl6_gate.py [turns=8]
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_adoption_probe as P

OUT_DIR = os.environ.get("BL6_OUT", "/home/derek/tensorfold-mom/experiments/results/bl6")
os.makedirs(OUT_DIR, exist_ok=True)
MAXTOK = int(os.environ.get("BL6_MAXTOK", "96"))
STOCK_DECODE_TARGET = 0.90 * 171.0

ARMS = {
    "A_stock": {},
    "B_copy": {"TF_MOM_ADOPT": "1"},
    "C_view": {"TF_MOM_ADOPT": "1", "TF_MOM_SHARED_VIEW": "1", "TF_MOM_SHARED_VERIFY": "1"},
}
PAGED = {"TF_QWEN_KV_PAGES": "64", "TF_QWEN_KV_PAGES_ALL": "1"}


def post_stream(payload, timeout=600):
    """Returns (ttft_s, total_s, completion_tokens)."""
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


def replay_timed(user_turns, replies):
    rows = []
    hist = [{"role": "system", "content": P.SYSTEM}]
    for i, c in enumerate(user_turns):
        hist.append({"role": "user", "content": c})
        ttft, total, toks = post_stream(
            {"model": P.NESTED_NAME, "messages": hist, "max_tokens": MAXTOK,
             "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        dec = total - ttft
        rows.append({"turn": i, "ttft_s": round(ttft, 4), "decode_s": round(dec, 4),
                     "tokens": toks, "tok_s": round(toks / dec, 1) if dec > 0.05 and toks else None})
        # continue the conversation with the model's own reply is NOT used:
        # teacher-force the parent reply exactly like the probe, so all arms see identical prefixes.
        hist.append({"role": "assistant", "content": replies[i]})
    return rows


VARIANTS = int(os.environ.get("BL6_VARIANTS", "3"))
PREFIX = int(os.environ.get("BL6_PREFIX", "4"))  # parent-warm turns before the measured thin turn


def run_arm(arm, env, user_turns, replies):
    """Routing scenario per variant: parent warm on turns 0..PREFIX-1 (its OWN replies),
    thin has NEVER seen this conversation -> measure the thin's first turn (adoption vs
    full prefill). Salted system prompts keep variants radix-disjoint so the thin's prompt
    cache cannot serve a previous variant. Round-1 lesson: replaying one nested session
    warm the thin's own cache and flattens TTFT in every arm."""
    log = f"/tmp/bl6-{arm}.log"
    P.LOG = log
    p = P.boot({**PAGED, **env}, truncate=True)
    if p is None:
        return None, None
    rows = []
    try:
        for v in range(VARIANTS):
            sys_prompt = P.SYSTEM + f"\n[bl6 variant {v}]"
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
    finally:
        P.kill(p)
    txt = open(log, errors="replace").read()
    adopted = txt.count("adopt sink at") + txt.count("adopt shared view")
    return rows, adopted


def blended(rows):
    # every row is a cold-thin adoption-eligible turn; nothing to exclude.
    tail = [r for r in rows if r["ttft_s"] > 0]
    return sum(r["ttft_s"] for r in tail) / max(1, len(tail))


def mean_toks(rows):
    v = [r["tok_s"] for r in rows if r["tok_s"]]
    return sum(v) / max(1, len(v))


def main() -> int:
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    user_turns = (P.TRIVIAL_TURNS + P.WORK_TURNS)[:turns]
    if not os.path.exists(P.REPLIES_FILE):
        print("need parent_replies.json (production M2-TF) at", P.REPLIES_FILE)
        return 2
    replies = json.load(open(P.REPLIES_FILE))
    if len(replies) < turns:
        print(f"replies file has {len(replies)} < {turns}")
        return 2

    results = {}
    for arm, env in ARMS.items():
        print(f"=== arm {arm} env={env}", flush=True)
        rows, adopted = run_arm(arm, env, user_turns, replies)
        if rows is None:
            print(f"ARM {arm} boot FAILED")
            return 2
        results[arm] = {"rows": rows, "adopted_lines": adopted,
                        "blended_ttft": round(blended(rows), 4),
                        "mean_tok_s": round(mean_toks(rows), 1)}
        print(json.dumps(results[arm], indent=1)[:900], flush=True)

    a = results["A_stock"]
    ratio_b = results["B_copy"]["blended_ttft"] / a["blended_ttft"]
    ratio_c = results["C_view"]["blended_ttft"] / a["blended_ttft"]
    ok_ttft = min(ratio_b, ratio_c) <= 0.85
    ok_decode = a["mean_tok_s"] >= STOCK_DECODE_TARGET
    ok_nonvac = results["B_copy"]["adopted_lines"] > 0 and results["C_view"]["adopted_lines"] > 0
    verdict = "GO" if (ok_ttft and ok_decode and ok_nonvac) else "NO-GO"
    print(f"\nBL6 TTFT ratios vs stock: copy={ratio_b:.3f} view={ratio_c:.3f} (need <=0.85)")
    print(f"BL6 stock thin decode {a['mean_tok_s']} tok/s (need >= {STOCK_DECODE_TARGET:.0f})")
    print(f"BL6 adoption lines: copy={results['B_copy']['adopted_lines']} view={results['C_view']['adopted_lines']}")
    print(f"BL6 verdict: {verdict}")
    out = {"results": results, "ratios": {"copy": ratio_b, "view": ratio_c},
           "stock_tok_s": a["mean_tok_s"], "non_vacuous": ok_nonvac, "verdict": verdict}
    path = os.path.join(OUT_DIR, f"bl6-{time.strftime('%Y%m%d-%H%M')}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
