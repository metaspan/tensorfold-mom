#!/usr/bin/env python3
"""BL-21(b) probe: thin decode with the parent's own MTP head as drafter.
Same measurement as BL-6 (tok/s = completion_tokens / (total - ttft)), plus a
coherence spot-check of one reply (drafting must not corrupt output)."""
import json, sys, time, urllib.request

PORT = 8219
PROMPTS = [
    "List three prime numbers between 10 and 20.",
    "What is 12 times 9? Answer with the number only.",
    "In one sentence, why is the sky blue?",
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
    # warmup (JIT/graph capture, MTP calibration chains)
    post_stream({"model": "nested-mtp", "messages": [{"role": "user", "content": "Hi"}],
                 "max_tokens": 8, "temperature": 0})
    rows = []
    for p in PROMPTS:
        ttft, total, toks, text = post_stream(
            {"model": "nested-mtp", "messages": [{"role": "user", "content": p}],
             "max_tokens": 96, "temperature": 0})
        dec = total - ttft
        rows.append({"tok_s": round(toks / dec, 1) if dec > 0.05 else None,
                     "tokens": toks, "ttft": round(ttft, 3)})
        print(rows[-1], "|", text[:120].replace("\n", " "))
    vals = [r["tok_s"] for r in rows if r["tok_s"]]
    mean = round(sum(vals) / len(vals), 1) if vals else None
    print(json.dumps({"mean_tok_s": mean, "rows": rows}))


if __name__ == "__main__":
    main()
