# MoM experiment harness

Every gate in the diary (`../docs-mom/README.md`) is reproducible from this
directory. Convention: probes print `VERDICT`-style lines and exit 0 only on
PASS; device windows run under `systemd-run --user --scope -p MemoryMax=100G`;
the `/home/derek/.venv/bin/python` interpreter is required (numpy); probes
that import `s1_fit`/`s1_align` rely on this exact directory being on
`sys.path`.

## Production tooling (not just probes)

| File | What it does |
|---|---|
| `nested_extract.py` | The ONLY tool that lays bytes on our NVFP4 checkpoints (mergekit would silently requantize). Byte surgery with safetensors header arithmetic, per-layer config-table truncation, MTP retention, head duplication. Ships `--verify`: sha256-compare every copied tensor against the parent. |
| `nested_materialize.py` | NVFP4 → bf16 materializer for HF-side training (fix of 2026-11-12: multiplies the `{name}_scale` fp8 weight scales — without it, weights inflated ~3×10³). |

## Track A / B (single-process MoM)

| Gate | Script | Criterion | Run |
|---|---|---|---|
| Front-door smoke | `smoke_frontdoor.py` | 10/10 routed responses | `python smoke_frontdoor.py` |
| Routing policy | `mom_router.py` + `routing_bench.py` | decision table parity with `mom_router.zig` 1:1; router p99 | `python routing_bench.py --help` |
| Gate A (Track A) | `results/RESULTS.md` artifacts | trivial e2e under saturation, strict token equality per lane | `references/gate-a-runbook.md` (skill) |
| Reroute/KV isolation | `reroute_probe.py` | lane hops never read another model's cache | via runbook |

## S1 — cross-model KV translation (measured NOT VIABLE)

| File | Role |
|---|---|
| `s1_align.py`, `s1_fit.py` | ridge-fit harness; `--eval-corpus` on a DISJOINT corpus is the only decision-grade mode (same-conversation holdout memorises — see docs/12) |
| `s1_collect.py`, `s1_collect2.py` | kvdump-driven collection (engine `TF_MOM_KVDUMP` hook) |

Gate criterion (pre-registered): mean top-1 ≥ 0.99 on disjoint eval. Measured
0.3418 → NOT VIABLE; the nested upcycle superseded it.

## M-series — the nested upcycle

| Gate | Script | Criterion | Run |
|---|---|---|---|
| M1 extraction | `m1_nested_probe.py`, `nested_load.py` | loader accepts the surgery output; layer facts from `text_config` | `python m1_nested_probe.py` |
| M2 identity | `m2_identity_probe.py`, `m2_tf_probe.py`, `m2_tf_score.py`, `m2_diff.py` | teacher-forced: shared-prefix K/V rows BIT-identical on the same token stream (token-hash equality asserted first) | `python m2_tf_probe.py` |
| M3 adoption | `m3_adoption_probe.py` (`run_g3.sh`) | adopted replies finite + byte-identical across reboots; tail-zero policy fires (NESTED.json `frozen_layers`) | `bash run_g3.sh` (GPU, prod down) |
| M4 shared view | `m3_adoption_probe.py` rounds 5+6 | view fires (producer sink line + consumer resolve line) and replies byte-identical against the SAME-paged-config copy baseline | `M3_SKIP_DIAGNOSTIC_ROUNDS=1 bash run_g3.sh` |
| Logits gate | `engine_vs_hf_logits.py` | greedy-continuation agreement across implementations (logprobs unsupported on this backend) | see header |
| GDN head dup | `gdn_dup.py`, `gdn_header_probe.py` | 16v→32v duplication geometry; header taps | one-shot probes |

## Distillation (student-only, docs/13 part 3)

| Gate | Script | Criterion | Run |
|---|---|---|---|
| Collect | `distill_collect.py` | ≥400 deduped rows incl. thinking-block rows (spec data mix) | inside `run_distill2.sh` |
| Train | `distill_train.py` | tail-only SFT (frozen prefix never trains), eval split reported | inside `run_distill2.sh` |
| **Coherence gate** | `distill_gate.py` + `gate_qa.json` | PRE-REGISTERED: ≥10/12 disjoint QA items, zero template-token leakage; thresholds fixed before training (S1 lesson) | `python distill_gate.py --model <dir>` |
| Full window | `run_distill2.sh` | chain of the three, freshness + port-band guards | `systemd-run --user --scope -p MemoryMax=100G bash experiments/run_distill2.sh` |

## Retired

`hf_rec_reference.py` — HF GDN third reference: both naive and `fla` kernel
paths corr≈0 from layer 1; the probe is invalid, kept for the negative record.
`distill_smoke.py` — early pipeline smoke, superseded by `run_distill2.sh`.
