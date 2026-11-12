# Backlog

Origin: `docs-mom/reviews/2026-11-12-grok-review.md` (section 8), triaged
2026-11-12. `Status: check` lines are Hermes' repo verification at intake.
Priorities are the reviewer's unless amended here, with the amendment reason.

## P1 (MVP)

### BL-1: Commit missing harness and results cited in docs
- Priority: P1
- Area: repro
- Scope: MVP
- Why: docs cite scripts whose absence breaks the "reproduce or it didn't happen" bar.
- Done when: every script/result named in `docs-mom` and `experiments/README.md` exists on `mom/main`.
- Refs: review §2 weaknesses; `experiments/README.md`
- Status: check - all seven scripts (`smoke_frontdoor.py`, `mom_router.py`, `routing_bench.py`, `reroute_probe.py`, `b4_sleep_probe.py`, `b5_cache_probe.py`, `aggregate_cache.py`) exist LIVE-UNTRACKED in `/home/derek/tensorfold-mom/experiments/`; `experiments/results/` exists live with JSONs; `results/RESULTS.md` not found anywhere. Fix is the documented live→repo sync, not authorship.

### BL-2: Un-ignore docs-mom notes logs; remove hardcoded /home/derek paths
- Priority: P1
- Area: repro
- Scope: MVP
- Why: cited evidence (`notes/b1-*.log`) is excluded by `.gitignore: *.log`; scripts pin one machine.
- Done when: a clean clone resolves all cited notes and paths.
- Refs: review §2; `.gitignore`
- Status: check - confirmed: `git ls-files docs-mom/notes | grep -c log` = 0 while b1/b0 logs sit on disk; `/home/derek` hardcoded in run_distill2.sh, run_distill.sh, run_g3.sh. Needs `-f` add or `!docs-mom/notes/*.log` negation + `$TF_MOM_HOME`-style parameterization.

### BL-3: Single `run_gates` script; result JSONs carry provenance
- Priority: P1
- Area: repro
- Scope: MVP
- Why: gates are run ad hoc; results lack commit sha / env flags / driver / model ids.
- Done when: `run_gates` exists; every result JSON records commit sha, env flags, driver/CUDA version, model ids.
- Refs: review §7.1
- Status: check - no `run_gates` in repo or live experiments dir.

### BL-4: Enforce per-model prompt cache mode in mom-serve
- Priority: P1
- Area: mom-serve
- Scope: MVP
- Why: prompt cache is opt-in; mom-serve only warns when `TF_QWEN_REQUEST_STARTS` unset (B6). Measured: TTFT 46->11 trivial, 175->28 heavy.
- Done when: mom-serve sets or refuses without `TF_QWEN_REQUEST_STARTS` and a B5 rerun shows ~85% cold / ~99.5% warm prefill saved.
- Refs: `docs-mom/docs/09`, PROGRESS B6
- Status: check - matches B6 record exactly.

### BL-5: Cross-model background warming
- Priority: P1
- Area: mom-serve
- Scope: MVP
- Why: `warm.zig` prefills next turns only on `srv.engine`; the other model stays cold across hops.
- Done when: ON/OFF A/B on B5 cold pass + trivial TTFT under heavy saturation recorded.
- Refs: `zig/src/server/warm.zig`, `docs-mom/docs/10`
- Status: check - warm.zig warms srv.engine only (verified in B-line work); estimate-only claim accepted as P1, effort M.

### BL-6: Shared-KV go/no-go (time-boxed)
- Priority: P1
- Area: research
- Scope: MVP
- Why: shared KV's payoff is unmeasured; ceiling is ~90 ms heavy TTFT per mid-conversation hop and 0% warm headroom; sticky routing makes hops rare.
- Done when: 3-arm verdict (stock pair / nested no-adopt / nested adopt) on B5 alternating + corpus B, cold, `--no-sticky`; metrics p50 TTFT on hop turns per model, blended p50 at real router-logged hop rate, thin decode tok/s; written docs/12-style.
- GO gate (pre-registered per S1 lesson, per review §3): blended TTFT −15% vs stock AND thin decode ≥90% of stock 2B. Else park M4 + distillation.
- Refs: review §3; `docs-mom/docs/13` M3
- Status: check - IN FLIGHT: M4 device gate is running this very day (window rounds 0/1/5/6, attn-tiles fix awaiting verdict), so the review's "device gate not yet run" is stale; the 3-arm design is the correct frame and its GO thresholds are adopted verbatim as pre-registered. Note the review's upside table assumes post-distillation 3/7 shared layers; current pre-distillation nested adopts all 16 (optimistic bound, as it says).

### BL-7: Run deferred S3, E1–E5, S5 soak, O2 /metrics labels
- Priority: P1
- Area: qa
- Scope: MVP
- Why: named tests deferred with no recorded result.
- Done when: results with commit sha under `experiments/results/`.
- Refs: `docs-mom/docs/04`
- Status: check - no results/ committed (see BL-1); accepted as stated.

### BL-8: Hard "hop-proof" gate after model switches
- Priority: P1
- Area: qa
- Scope: MVP
- Why: post-switch incoherent/empty replies are the known bug class (docs/12 run 1, B3 template/thinking); Compute Lanes' "arm proof" analogue.
- Done when: gate script fails on empty or incoherent post-switch reply.
- Refs: review §6
- Status: check - no such gate exists; accepted.

## P2 (MVP)

### BL-9: Repetition discipline for benchmarks
- Priority: P2
- Area: eval
- Scope: MVP
- Why: single runs without spread (B5 28-turn, corpus B 20, gate_qa 12 substring checks).
- Done when: every benchmark ≥3 runs with thermal cool-downs; median+spread; per-task-family on/off + no-harm general + cap-hit count format (review §6 eval format).
- Refs: review §2, §6
- Status: check - accurate as stated; note distill v2's eval split already added today.

### BL-10: Cost-aware routing in v2 slot
- Priority: P2
- Area: router
- Scope: MVP
- Why: warm-heavy (28 ms) can beat cold-trivial (46 ms); decode dominates end-to-end; route on uncached prefill cost + expected tokens/rate; consider "compute" route (half of gate_qa is arithmetic).
- Done when: labelled set from router logs; misroute rate + end-to-end p50 on mixed traffic.
- Refs: `docs-mom/docs/09`; RouteLLM; semantic-router
- Status: check - v2 slot exists by design in mom_router.zig; accepted as P2.

### BL-11: Default no-sleep for this pair; rebalance KV quotas
- Priority: P2
- Area: mom-serve
- Scope: MVP
- Why: 12–14 s wake punishes sleep for this pair; heavy's prompt-cache budget mostly idle.
- Done when: capacity lines + B5 rerun recorded.
- Refs: `docs-mom/docs/09`, PROGRESS B4
- Status: check - accepted.

### BL-12: Distillation scale-up (ONLY if BL-6 GO)
- Priority: P2 (conditional)
- Area: distill
- Scope: MVP-if-GO
- Why: 175-row corpus can't reach coherence; scale from router logs; sweep frozen-layer count N.
- Done when: thousands of rows; held-out G2 set 200+ items vs stock Qwen3.5-2B vs plain truncation baseline; frozen-N sweep; early-exit-head baseline at 35B layer N.
- Refs: review item 12 arXiv links (LLM Pruning/Shearing, Minitron, arxiv 2403.17887)
- Status: check - distill v2 window (96 seeds, thinking rows, pre-registered ≥10/12 gate) is staged and queued; this item extends, not replaces, it. Gated on BL-6 verdict per reviewer.

## P3

### BL-13: Docs drift cleanup + gate glossary + PROGRESS ordering
- Priority: P3
- Area: docs
- Scope: MVP
- Why: README "frozen 16-layer prefix" vs frozen_layers:10+tail; `docs/14` names `TF_MOM_SHARED_KV` (implementation uses `TF_MOM_SHARED_VIEW`); `docs/13` still PROPOSED; S1 ambiguous across docs/04 & docs/12; M-numbering clash; PROGRESS not chronological.
- Done when: each claim above reconciled; gate glossary added; PROGRESS newest-first.
- Refs: review §2
- Status: check - VERIFIED: `TF_MOM_SHARED_KV` in docs-mom/docs/14 lines 79+95; "frozen 16-layer" in docs-mom/README line 43 (in context it says "frozen 16-layer" while NESTED.json says frozen_layers:10 + head-dup tail). Accepted.

### BL-14: Related-work section
- Priority: P3
- Area: docs
- Scope: MVP
- Why: DroidSpeak, Cache-to-Cache, MatFormer, LMCache are directly adjacent prior work; reviewers/upstream will ask.
- Done when: related-work section in docs-mom citing them + BL-12 refs.
- Refs: arXiv 2411.02820, 2510.03215, 2310.07707, LMCache repo
- Status: check - none present today; cheap and genuinely relevant; accepted.

## P3 — upstream, post-MVP (NOTE ONLY per Derek's scope rule)

### BL-15: Upstream issue: qwen3_5 prompt cache default-off
- Priority: P3 · Area: upstream · Scope: post-MVP (note only)
- Why: opt-in cache cost 77–84% TTFT here. Refs: `docs-mom/docs/09`.
- Status: check - noted, not pursued for MVP.

### BL-16: Upstream issue: radix mode observability / zero reuse
- Priority: P3 · Area: upstream · Scope: post-MVP (note only)
- Why: radix showed zero observable reuse and is unobservable (`lane_host.zig:453`). Refs: `docs-mom/docs/09`.
- Status: check - matches B5 finding; noted.

### BL-17: A/B CUDA forward-compat driver
- Priority: P3 · Area: upstream · Scope: post-MVP (note only)
- Why: ~13% reported externally (unverified here); needs cool-downs. Refs: review §6 URL.
- Status: check - third-party claim, unverified; noted.

### BL-18: Track Compute Lanes recipe
- Priority: P3 · Area: upstream · Scope: post-MVP (note only)
- Why: released on TensorFold; MoM and lanes both touch the stream path.
- Status: check - noted; coordinate before UPSTREAM-PRs overlap (see BL-20).

### BL-19: Load/wake speed + reserve sizing + SafeAllocator leaks (upstream issues)
- Priority: P3 · Area: upstream · Scope: post-MVP (note only)
- Why: 31.3 s heavy cold load, 12–14 s wake, 12.2 GiB reserve (may be conservative), 31 exit-path leak lines/run. Refs: PROGRESS B1/B4, `docs-mom/docs/01`.
- Status: check - numbers are ours (measured); good upstream issues later; noted.

### BL-20: Coordinate stream-path overlap with Compute Lanes before PRs
- Priority: P3 · Area: upstream · Scope: post-MVP (note only)
- Why: `docs-mom/UPSTREAM-PRs.md` PRs touch stream/pool paths; lanes does too; ATC overlap check + early contact.
- Status: check - folded into UPSTREAM-PRs.md ritual (atc overlap exists); noted.
