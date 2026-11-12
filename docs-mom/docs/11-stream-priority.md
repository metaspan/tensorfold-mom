# Component 3 leftover — GPU stream priority measurement

**Question.** With one process holding both lanes, does opening the trivial
lane's CUDA stream at `.highest` priority (the `cuda/stream.zig initPriority`
seam, previously uncalled) protect trivial-lane TTFT while the 35B lane is
mid-forward?

**Experiment (B7, `experiments/b7_priority_bench.py`).** mom-serve with
Qwen3.5-2B (trivial) + Qwen3.6-35B-A3B-NVFP4 (heavy), `--parallel 2`. Saturate
the heavy lane with 8 concurrent 200-token decodes; during saturation fire 10
streamed 16-token trivial probes; run A/B: `--lane-priority` ON vs OFF. GB10,
greedy, thinking-off, post-1.2.7 binary (commit `c1de3cc`).

**Result.**

| mode | trivial TTFT p50 | trivial TTFT p95 | heavy wall (8×200 tok) | heavy tok/s |
|---|---|---|---|---|
| OFF | **1.4 ms** | 1.8 ms | 15.0 s | 106.7 |
| ON  | **1.4 ms** | 1.8 ms | 15.3 s | 104.8 |

Delta: **0.0%** trivial TTFT; heavy cost +2.0% (noise). `priority_helps=false`,
`heavy_regression_ok=true`. B7_RC=0.

**Interpretation.** GPU-level priority is unnecessary *in this architecture*:
component 3's structural fairness is already delivered by per-lane batching —
each lane is its own engine + lanes-host batching loop, so trivial work is not
queued behind heavy work at the submission level, and the GPU interleaves the
2B lane's short kernels between the 35B batch's microsteps within the ~ms
scheduling granularity anyway. A saturated 35B decode batch never blocked the
2B lane beyond ~1.4 ms — the same as idle. Stream priority would matter only
if a single kernel ran long enough to dominate (a giant prefill); the probe's
saturation included the 8-request prefill ramp and still measured 1.4 ms.

**Disposition.** Ship `--lane-priority` as an inert-by-default escape hatch
(useful if a future workload shows kernel-granularity stalls, e.g. monolithic
prefills without the 4096-row chunking), but close component 3 on the evidence
above: per-model queues + per-lane batching are sufficient for the trivial-lane
latency goal (0.1 s budget; measured 1.4 ms under full heavy load).
