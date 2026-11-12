# B0 baseline — pristine upstream `14dea6c6` on spark-2d3b (GB10, 2026-11-10)

Environment: Zig 0.17.0 (~/.local/bin/zig, exact `.zig-version` match), Python 3.12 via
`uv run --no-project --with pytest --with pyyaml`, PYTHONPATH=web. Production
container on :8000 RUNNING during the run (relevant only if a test truly needs the GPU).
Logs: `notes/b0-zig-test.log`, `notes/b0-zig-golden.log`.

## Results

| Suite | Result |
|---|---|
| `zig build test` | exits 0, but 5 test binaries print `failed command:` — KNOWN-RED below |
| `zig build test-golden` | 322/335 equal, **0 unexpected differences** (13 known) — PASS |
| `python3 -m pytest` | 462 passed, 10 skipped, 521 subtests passed; 1 fail — KNOWN-RED |

## KNOWN-RED list (pre-fork; identical failures post-change are NOT regressions)

1. `test-cuda-host` ×3 binaries: GPU-path warnings `GpuFailed: every admitted stream
   fails`, `packed decision failed (DeviceLost)`, `TF_GLM_L2PF: layer 0's site n reads
   0x10000000` — host tests exercising injected-failure paths on this GB10/CUDA host.
2. `test-env-defaults-host` ×1: same TF_GLM_L2PF prefetch warning.
3. DEEPSEEK_CHAT_GOLDENS (in a `test-cuda-host` binary): 3 committed-hash mismatches
   over 234 encoding.py cases — environment: `TF_TOKENIZER_DIRS` unset (skipped
   markers logged), so golden text differs. CI sets it.
4. pytest `tools/zig/zig_cache_restat_test.py::RestatTests::test_only_a_real_change_misses`
   — exceeds 60s pytest-timeout on this host (zig-cache re-stat is disk-bound).
   Run the file alone with a higher timeout if it ever matters.

Note the quirk: `zig build test` returns 0 despite failed commands — rely on the
`failed command:` count (expect exactly 5) rather than the exit code alone.

UPDATE (same day): per-step counts SEED-VARY — pristine `zig build test-cuda-host` alone
showed 8 failed commands (seed 0x6e4a87a9) where the full-suite run showed fewer; a fork
run showed 4. The GPU-path injected-failure host tests are seed-sensitive. Triage rule:
compare per-step counts pristine-vs-fork in the SAME session, and grep for NEW test names
(the baseline KNOWN-RED list above), not raw counts.

## Commands (reproducible)

```
cd /home/derek/tensorfold-mom/src/tensorfold
export PATH=/home/derek/.local/bin:$PATH
zig build test > ../notes/b0-zig-test.log 2>&1          # expect exit 0, 5x failed command
zig build test-golden                                     # expect 0 unexpected differences
PYTHONPATH=web uv run --no-project --with pytest --with pyyaml \
  python3 -m pytest -q --timeout=60                       # expect 462+ passed, only restat fail
```

Anything NEW red after B1+ = fork regression — bisect before proceeding.
