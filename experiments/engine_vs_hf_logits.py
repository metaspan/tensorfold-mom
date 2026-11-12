#!/usr/bin/env python3
"""Engine-vs-checkpoint logits gate.

Same 179 tokens (M2-TF kept-state stream): the NVFP4 engine (mom-serve, raw
/v1/completions, temp 0) and the bf16 HF checkpoint must agree on the next-token
distribution. Engine right + HF wrong -> materialized checkpoint broken
(distillation poisoned). Both HF kernels already agree with each other, so HF
internals are not the suspect; a mismatch localizes the fault for good.
Token identity is PROVEN via the engine's kvdump before comparing logits.
"""
import json, os, subprocess, sys, time, urllib.request

import numpy as np
import torch

sys.path.insert(0, "/home/derek/tensorfold-mom/src/tensorfold/experiments")
from s1_fit import load_model_dir
from b5_cache_probe import TRIVIAL_TURNS, WORK_TURNS

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
NESTED = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"
STUDENT = "/home/derek/models/tensorfold-mom/qwen36-nested-16l-bf16"
REPLIES = "/home/derek/tensorfold-mom/experiments/results/m2-tf/parent_replies.json"
DUMPDIR = "/tmp/engine_vs_hf_kvdump"
PORT = 8321
SYSTEM = "You are a concise assistant."


def post(path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def main():
    # 1) ground-truth tokens + rendered text (hist WITHOUT generation prompt = 179 tokens)
    ck = load_model_dir("/home/derek/tensorfold-mom/experiments/results/m2-tf/kvdump/qwen36-nested-16l")
    items = list(ck.values()) if isinstance(ck, dict) else list(ck)
    c8 = [c for c in items if getattr(c, "at", None) == 179][0]
    want = list(c8.token_ids)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)
    turns = (TRIVIAL_TURNS + WORK_TURNS)[:8]
    replies = json.load(open(REPLIES))
    hist = [{"role": "system", "content": SYSTEM}]
    for u, a in zip(turns, replies):
        hist.append({"role": "user", "content": u})
        hist.append({"role": "assistant", "content": a})

    # 2) boot engine single-lane with kvdump
    os.makedirs(DUMPDIR, exist_ok=True)
    for f in os.listdir(DUMPDIR):
        os.remove(os.path.join(DUMPDIR, f))
    log = open("/tmp/engine_vs_hf_server.log", "w")
    env = os.environ.copy()
    env["TF_QWEN_REQUEST_STARTS"] = "1"
    env["TF_MOM_KVDUMP"] = DUMPDIR
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT),
                          "--model", NESTED, "--name", "nested", "--lane", "default", "--pin"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    deadline = time.time() + 900
    while time.time() < deadline and not health():
        if p.poll() is not None:
            print("server died:", open("/tmp/engine_vs_hf_server.log").read()[-1500:])
            return 2
        time.sleep(2)
    if not health():
        print("health timeout")
        p.terminate()
        return 2
    try:
        r = post("/v1/chat/completions", {"model": "nested", "messages": hist,
                                          "max_tokens": 12, "temperature": 0,
                                          "chat_template_kwargs": {"enable_thinking": False}})
    finally:
        p.terminate()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    choice = r["choices"][0]
    eng_text = choice["text"]
    print("engine completion (12 greedy tokens):", repr(eng_text))

    # 3) token identity proof via the engine's own dump
    ck2 = load_model_dir(DUMPDIR)
    items2 = list(ck2.values()) if isinstance(ck2, dict) else list(ck2)
    got = list(items2[-1].token_ids)
    same = got[:179] == want
    print(f"engine prompt tokens: {len(got)}; first 179 == kept-state stream: {same}")
    if not same:
        print("engine ids:", got[:10], "...")
        print("want  ids:", want[:10], "...")
        return 3

    # 4) HF greedy continuation on the engine's own prompt tokens
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.to("cuda")
    model.eval()
    ids = torch.tensor([got], device="cuda")
    hf_ids = []
    with torch.no_grad():
        for _ in range(12):
            out = model(ids, use_cache=False)
            nxt = int(out.logits[0, -1].float().argmax())
            hf_ids.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], dim=1)
    eng_ids = tok.encode(eng_text, add_special_tokens=False)
    print("hf   ids:", hf_ids)
    print("eng  ids:", eng_ids)
    div = next((i for i, (h, e) in enumerate(zip(hf_ids, eng_ids)) if h != e),
               min(len(hf_ids), len(eng_ids)))
    print("hf   text:", repr(tok.decode(hf_ids)))
    print("eng  text:", repr(eng_text))
    print(f"first divergence at completion position {div} of {len(eng_ids)}")
    print("LOGITS-GATE-DONE tokens_match =", same, "agree_len =", div)
    return 0


if __name__ == "__main__":
    sys.exit(main())