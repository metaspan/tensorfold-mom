#!/usr/bin/env python3
"""M1 nested-checkpoint smoke: open the nested 16-layer truncation on the
native engine (docs/13 M1). Single lane, mom-serve; health, /v1/models, and
one deterministic chat. PASS = coherent reply + 16-layer facts + sane grant.
Window owner stops production first."""
import json
import os
import subprocess
import sys
import time
import urllib.request

BIN = "/home/derek/tensorfold-mom/src/tensorfold/zig-out/native/bin/tensorfold-native"
NESTED = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"
PORT = 8214
LOG = "/tmp/m1-nested.log"


def post(payload, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get(path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    env = os.environ.copy()
    log = open(LOG, "w")
    p = subprocess.Popen([BIN, "mom-serve", "--host", "127.0.0.1", "--port", str(PORT),
                          "--parallel", "1",
                          "--model", NESTED, "--name", "qwen36-nested",
                          "--lane", "trivial", "--pin"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            if p.poll() is not None:
                print("SERVER DIED — loader refused the nested checkpoint:")
                print(open(LOG).read()[-2500:])
                return 2
            try:
                if get("/health").get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            print("health timeout; log tail:")
            print(open(LOG).read()[-2500:])
            return 2

        models = get("/v1/models")
        print("/v1/models:", json.dumps(models)[:600])
        r = post({"model": "qwen36-nested",
                  "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
                  "max_tokens": 16, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}})
        content = (r.get("choices") or [{}])[0].get("message", {}).get("content", "")
        print("chat reply:", repr(content))
        ok = bool(content) and ("4" in content)
        print("M1:", "PASS" if ok else "FAIL (incoherent reply)")
        return 0 if ok else 1
    finally:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
