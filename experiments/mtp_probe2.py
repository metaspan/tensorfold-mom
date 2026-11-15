#!/usr/bin/env python3
"""MTP trained-head probe v2: tok/s + per-request done stats.

Sends chat_template_kwargs enable_thinking:false (thin lane defaults
thinking ON; confounds tok/s). Prints per-request tok/s and the aggregate;
the serve log's done lines (rounds=/accepted=X/Y) are parsed separately.
"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8219
MODEL = sys.argv[2] if len(sys.argv) > 2 else "mtp-t3"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12

PROMPTS = [
    "List three prime numbers between 10 and 20.",
    "What is 12 times 9? Answer with the number only.",
    "In one sentence, why is the sky blue?",
    "Name the three largest planets in our solar system.",
    "What is 45 plus 27? Just the number.",
    "Translate 'good morning' to French.",
    "How many minutes are in 2.5 hours?",
    "Write a two-line rhyme about the sea.",
]


def post_stream(payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({**payload, "stream": True,
                         "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    ttft = None
    tokens = 0
    text = ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            d = json.loads(body)
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece or ""
            if d.get("usage"):
                tokens = d["usage"].get("completion_tokens", tokens)
    return ttft or 0.0, time.time() - t0, tokens, text


def main():
    base = {"model": MODEL,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0}
    post_stream({**base, "messages": [{"role": "user", "content": "Hi"}],
                 "max_tokens": 8})  # warmup
    rows = []
    for i in range(N):
        p = PROMPTS[i % len(PROMPTS)]
        ttft, total, toks, text = post_stream(
            {**base, "messages": [{"role": "user", "content": p}],
             "max_tokens": 128})
        dec = total - ttft
        r = {"tok_s": round(toks / dec, 1) if dec > 0.05 else None,
             "tokens": toks, "ttft": round(ttft, 3)}
        rows.append(r)
        print(json.dumps(r), "|", text[:70].replace("\n", " "), flush=True)
    vals = [r["tok_s"] for r in rows if r["tok_s"]]
    vals_s = sorted(vals)
    med = vals_s[len(vals_s) // 2] if vals_s else None
    print(json.dumps({"mean_tok_s": round(sum(vals) / len(vals), 1) if vals else None,
                      "median_tok_s": med,
                      "min_tok_s": min(vals) if vals else None,
                      "n": len(vals)}))


if __name__ == "__main__":
    main()
