# B4 — Sleep/Wake + Memory Orchestrator (work spec)

Brief component 6: "Global memory accounting across models. Supports pinning
the small model, sleeping the large model, and rapid reactivation."
Pinning ✓ (B1 `--pin`); global accounting ✓ (mom_plan); this doc specifies
sleep/wake — the last system-path item before the v2 classifier slot.

## Why it matters on one GB10

Co-resident lanes cost 1.6 + 21.8 GiB weights + per-lane KV quotas. With the
heavy lane asleep, the trivial lane can take a far bigger KV window, and a
THIRD lane becomes affordable. Priority use case unaffected: the trivial lane
is permanently hot BY POLICY.

## Reuse: idle_unload.zig (upstream) is the template, not the mechanism

`Manager` (idle_unload.zig:29) already implements exactly the lifecycle we
need for ONE engine: `State = { resident, unloading, unloaded, loading,
stopped }`, mutex+cond, `active` request count, `last_activity_ns`, a timer
thread, and the `Reopen` triple (`open`, `close_held`, `replace`, optional
`halt_held`). It swaps `srv.engine`.

MoM needs it per LANE, swapping `registry.lanes[i].engine`. Generalize, do
not fork the file: extract the lifecycle into `mom_sleep.zig` as
`LaneLifecycle` (state machine + lease `begin()/end()` + park semantics),
parameterized by a per-lane `Reopen` closure. Upstream keeps its Manager for
the non-MoM single-engine path untouched (rebase rule: additive).

`api.Opened` / engine close stays exactly as mom_main already does at startup
(B1 sequential open); a wake is just that open again, plus the B1
context_owner custody handles the primary-context side effects.

## Design (v1)

- Policy flags (mom_main): `--lane-idle SECONDS` (0 = never sleep, default),
  applies to NON-trivial lanes only. `--sleep-wake-log PATH` records
  sleep/wake transitions with latency (observability: "switch latency").
- Registry lane gains: `last_used_ns`, `state`, `lease: u32`, `reopen: Reopen`.
  route() refreshes `last_used_ns` under the existing registry mutex; submit
  path takes a lease (`begin`) and drops it (`end`) exactly like idle_unload's
  route wrapper, so a park never closes an engine mid-request.
- Timer (one thread, shared): every tick, for each eligible lane, if
  `state == .resident and active == 0 and now - last_used > idle_s` →
  unloading → close (frees weights+KV) → unloaded. Log transition + free-mem
  delta via the same accounting mom_plan already prints.
- Wake-on-route: route() returning a lane whose state != resident parks the
  caller on the cond and reopens synchronously (open is the startup path,
  ~seconds; measured), then proceeds. Reply latency includes the wake; that
  is correct behaviour for a cold heavy lane and is exactly what the
  switch-latency metric must capture.
- Health/`/v1/models`: sleeping lane stays advertised, `"mom"` gains
  `"state": "asleep"` (metrics extension of 86e385c).

## Invariants

- Trivial lane NEVER sleeps (assert at setup if --lane-idle set and role trivial).
- N=1 parity: without --lane-idle nothing constructs a lifecycle; zero cost.
- Exact decoding: wake reopens the identical checkpoint/args; the tokenizer
  guard already ran at startup. A wake must produce byte-identical behaviour
  to the initial open (proven by rerun of b3 probe turns post-wake).
- context_owner custody: sleeping the LAST resident engine must not drop the
  primary context while another lane is resident (B1 refcount handles); test
  explicitly (sleep heavy, trivial stays, wake heavy).

## Test plan

- T0 host (test-mom-host, filter `sleep:`): state machine transitions incl
  unloading-cancel (request arrives mid-unload → resident, mirroring
  idle_unload.begin), lease counting, idle arithmetic with injected clock.
- T1 device (new window, b4_sleep_probe.py):
  1. `--lane-idle 20`; chat trivial; wait > idle_s; assert
     /v1/models heavy state == "asleep"; nvidia-smi free memory rose by the
     heavy lane's footprint; trivial still answers (TTFT unaffected).
  2. heavy request → wake: assert reply correct and wake latency logged
     (< 60 s bound, record actual); rerun K3-style continuity turns post-wake
     == pre-wake (exactness across sleep).
  3. sleep heavy while trivial resident and vice versa (custody cross-check).
- T2 stress: loop sleep/wake 5x, RSS + leak check (SafeAllocator clean).

## Order of work

1. mom_sleep.zig LaneLifecycle extraction + host tests (pure; clock injected).
2. Registry integration: lane fields, lease begin/end in the chat path,
   wake-on-route in route().
3. mom_main: flags, Reopen closures (already have the open code), timer thread.
4. /v1/models state field; sleep/wake JSONL.
5. Device probe + gate: state flips visible, memory freed/returned,
   post-wake byte-identity, custody cross-check, 5-cycle leak check.
