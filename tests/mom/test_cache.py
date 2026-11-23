#!/usr/bin/env python3
"""MoM MVP cache correctness regression (P2, 23 Nov 2026).

Adopted MVP config (docs-mom/docs/12 addendum): BOTH lanes request-starts
prompt-cache mode (TF_QWEN_REQUEST_STARTS=1); exact re-prefill on lane
switch; no cross-checkpoint shared cache (S1 run 3 NO-GO).

Checks (device run needed for 2-4):
  1. tokenizer consistency: identical BPE vocab + merges across the two
     checkpoints; no added-token id conflicts (shared-prefix correctness
     precondition).
  2. boot registers the request-starts cache mode.
  3. same-lane warm-pass reuse: both lanes reuse their own prefix
     (resumed_at > 0 on repeated turns; prefill-saved >= 90% warm).
  4. cross-lane switch = full prefill: the first request that moves a
     conversation to the OTHER lane reports resumed at 0 (no phantom reuse).

Run on device (GB10): python3 tests/mom/test_cache.py
Exits non-zero on any failure. Uses the boot/post helpers from
experiments/b5_cache_probe.py; port via TFM_PORT (default 8219 to stay off
the probe band).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "experiments"))
os.environ.setdefault("TFM_PORT", "8219")
os.environ.setdefault("TFM_LOG", "/tmp/mom-cache-test.log")
os.environ.setdefault("TFM_OUT", "/tmp/mom-cache-test-results")

import b5_cache_probe as P  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def test_tokenizer_consistency():
    """BPE identity precondition (no device needed)."""
    for fn in ("tokenizer.json",):
        a = json.load(open(os.path.join(P.SMALL, fn)))
        b = json.load(open(os.path.join(P.BIG, fn)))
        check("vocab identical", a["model"]["vocab"] == b["model"]["vocab"],
              f"{len(a['model']['vocab'])} tokens")
        # Merges are stored inconsistently across the checkpoints (list of
        # pairs vs pre-joined strings); normalize before comparing.
        def norm(ms):
            return [m if isinstance(m, str) else m[0] + " " + m[1] for m in ms]
        check("merges identical (normalized)",
              norm(a["model"]["merges"]) == norm(b["model"]["merges"]),
              f"{len(a['model']['merges'])} merges")
        at_a = {t["content"]: t["id"] for t in a.get("added_tokens", [])}
        at_b = {t["content"]: t["id"] for t in b.get("added_tokens", [])}
        conflicts = {k: (at_a[k], at_b[k]) for k in at_a.keys() & at_b.keys()
                     if at_a[k] != at_b[k]}
        check("no added-token id conflicts", not conflicts, str(conflicts) if conflicts else "")


def main() -> int:
    print("== 1. tokenizer consistency (offline) ==")
    test_tokenizer_consistency()

    print("== 2. boot with MVP env ==")
    p = P.boot(truncate=True)
    if p is None:
        print("FAIL: boot"); return 1
    try:
        log = open(P.LOG, errors="replace").read()
        check("request-starts cache mode registered",
              "registered overrides: TF_QWEN_REQUEST_STARTS=1" in log)
        check("both lanes loaded",
              "routing over 2 lanes" in log
              and "loaded (pinned)" in log)

        # ---- same-lane warm reuse + cross-lane full prefill ----
        # 6-turn alternating conversation, replayed cold then warm (same
        # process), forcing each lane to see every other turn.
        turns = [P.TRIVIAL_TURNS[i // 2] if i % 2 == 0 else P.WORK_TURNS[i // 2][:200]
                 for i in range(8)]
        cold, n = P.replay_pass(turns, "cold-regression", P.OUT, P.LOG)
        warm, _ = P.replay_pass(turns, "warm-regression", P.OUT, P.LOG, start_line=n)

        print("== 3. same-lane warm reuse ==")
        agg = P.analyze(warm, "warm")
        for lane in (P.SMALL_NAME, P.BIG_NAME):
            check(f"warm prefill-saved >= 90% ({lane})",
                  agg[lane]["prefill_saved_pct"] >= 90.0,
                  f"{agg[lane]['prefill_saved_pct']}%")
            check(f"warm reuse turns ({lane})",
                  agg[lane]["reuse_turns"] >= agg[lane]["reuse_turns_measurable"] - 1,
                  f"{agg[lane]['reuse_turns']}/{agg[lane]['reuse_turns_measurable']}")

        print("== 4. cross-lane switch = full prefill ==")
        # Cold pass: each lane's FIRST turn must be a full prefill
        # (resumed at 0) even though the conversation exists on the other
        # lane -- no phantom cross-checkpoint reuse.
        first_small = next(r for r in cold if r["lane"] == P.SMALL_NAME)
        first_big = next(r for r in cold if r["lane"] == P.BIG_NAME)
        check("first small-lane turn full prefill",
              first_small.get("resumed_at") == 0, f"resumed at {first_small.get('resumed_at')}")
        check("first big-lane turn full prefill (post-switch)",
              first_big.get("resumed_at") == 0, f"resumed at {first_big.get('resumed_at')}")
    finally:
        P.kill(p)

    if failures:
        print(f"\nRESULT: FAIL ({len(failures)} checks): {failures}")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
