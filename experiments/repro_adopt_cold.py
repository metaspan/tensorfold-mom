#!/usr/bin/env python3
"""Repro: adopted cold-thin completion returns empty on the 35B pair.
Boot the view arm, warm the parent, then hit the thin cold turn three ways:
non-streaming, streaming, and non-streaming again (fresh prefix variant)."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_adoption_probe as P


def post_json(payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{P.PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    turns = (P.TRIVIAL_TURNS + P.WORK_TURNS)[:8]
    env = {"TF_MOM_ADOPT": "1", "TF_MOM_SHARED_VIEW": "1", "TF_MOM_SHARED_VERIFY": "1", "TF_MOM_ADOPT_HOST": "1",
           "TF_QWEN_KV_PAGES": "64", "TF_QWEN_KV_PAGES_ALL": "1"}
    P.LOG = "/tmp/adopt-cold-repro.log"
    p = P.boot(env, truncate=True)
    if p is None:
        return 2
    try:
        for v in (0, 10):  # salted: each number = its own cold conversation
            hist = [{"role": "system", "content": P.SYSTEM + f"\n[repro {v}]"}]
            for c in turns[:4]:
                hist.append({"role": "user", "content": c})
                r = post_json({"model": P.BIG_NAME, "messages": hist, "max_tokens": 24,
                               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
                hist.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
            hist.append({"role": "user", "content": turns[4]})
            base = {"model": P.NESTED_NAME, "messages": hist, "max_tokens": 48, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            mode = "nonstream" if v == 0 else "stream"
            if mode == "nonstream":
                r = post_json(dict(base))
                ch = r["choices"][0]
                print(f"v{v} {mode}: finish={ch.get('finish_reason')} "
                      f"len={len(ch['message']['content'])} usage={r.get('usage')} "
                      f"content={ch['message']['content'][:80]!r}")
            else:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{P.PORT}/v1/chat/completions",
                    data=json.dumps({**base, "stream": True,
                                     "stream_options": {"include_usage": True}}).encode(),
                    headers={"Content-Type": "application/json"})
                pieces, usage, finish = [], None, None
                with urllib.request.urlopen(req, timeout=600) as r2:
                    for line in r2:
                        if not line.startswith(b"data: "):
                            continue
                        body = line[6:].strip()
                        if body == b"[DONE]":
                            break
                        d = json.loads(body)
                        if d.get("usage"):
                            usage = d["usage"]
                        for ch2 in d.get("choices", []):
                            if ch2.get("finish_reason"):
                                finish = ch2["finish_reason"]
                            pc = (ch2.get("delta") or {}).get("content")
                            if pc:
                                pieces.append(pc)
                print(f"v{v} {mode}: finish={finish} len={sum(len(x) for x in pieces)} usage={usage} "
                      f"content={''.join(pieces)[:80]!r}")
    finally:
        P.kill(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
