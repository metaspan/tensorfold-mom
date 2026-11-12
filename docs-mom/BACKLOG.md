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
- Done when: mom-serve sets or refuses without `TF_QWEN_REQUEST_STARTS` and a B5 rerun shows ~85% cold / ~99.5% warm prefill saved; plus a startup assertion on each lane's cache mode covered by a host test (perplexity §9.1).
- Refs: `docs-mom/docs/09`, PROGRESS B6
- Status: check - matches B6 record exactly.

### BL-5: Cross-model background warming
- Priority: P1
- Area: mom-serve
- Scope: MVP
- Why: `warm.zig` prefills next turns only on `srv.engine`; the other model stays cold across hops.
- Done when: ON/OFF A/B on B5 cold pass + trivial TTFT under heavy saturation recorded.
- Refs: `zig/src/server/warm.zig`, `docs-mom/docs/10`
- Status: check - warm.zig warms srv.engine only (verified in B-line work); estimate-only claim accepted as P1, effort M. SUPERSEDED-BY BL-24 (ShadowPrefill framing: preemptible, exactness-preserving, and the control arm for BL-6).

### BL-6: Shared-KV go/no-go (time-boxed)
- Priority: P1
- Area: research
- Scope: MVP
- Why: shared KV's payoff is unmeasured; ceiling is ~90 ms heavy TTFT per mid-conversation hop and 0% warm headroom; sticky routing makes hops rare.
- Done when: 3-arm verdict (stock pair / nested no-adopt / nested adopt) on B5 alternating + corpus B, cold, `--no-sticky`; metrics p50 TTFT on hop turns per model, blended p50 at real router-logged hop rate, thin decode tok/s; written docs/12-style.
- GO gate (pre-registered per S1 lesson, per review §3): blended TTFT −15% vs stock AND thin decode ≥90% of stock 2B. Else park M4 + distillation.
- Refs: review §3; `docs-mom/docs/13` M3
- Status: check - IN FLIGHT: M4 device gate is running this very day (window rounds 0/1/5/6, attn-tiles fix awaiting verdict), so the review's "device gate not yet run" is stale; the 3-arm design is the correct frame and its GO thresholds are adopted verbatim as pre-registered. Note the review's upside table assumes post-distillation 3/7 shared layers; current pre-distillation nested adopts all 16 (optimistic bound, as it says).
- Status: M4 toy device gate GREEN (2026-11-12, mom/main 19a969667): zero-copy shared-view reply byte-identical to the paged copy path (6 resolve lines), G3 decode-resume PASS x6 across boots. Root causes fixed: stateless `tf_q35_page_base` (the monotone-walk page-base cache broke under interleaved PageRuns + pending-row replay) on top of the four skvi reader-site fixes. Today's long red-run was phantom builds: `zig build … | tail; echo RC=$?` hid a failing exe compile (safe-mode `var poison` error) and the content-keyed fatbin step ignored header touches — no fixed kernel ever reached a tested binary until an honest rebuild. 35B BL-6 GO/NO-GO (TTFT −15% / decode ≥90%) still owed: needs the :8000 window.

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
- Done when: gate script fails on empty or incoherent post-switch reply; formulation per Compute Lanes "arm proof": require N tokens of continued decode matching the copy-path baseline, not merely "adoption fired"; include think-block/template tests across hops (EOS-in-forced-tokens, `<think>` reopening after injected content).
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
- Refs: arXiv 2411.02820, 2510.03215, 2310.07707, LMCache repo; plus ShadowPrefill (2611.13176), NVIDIA cross-model KV transfer (2608.03893), BridgeKV (2611.11761), ICaRus (2603.13281), PrefillShare (2602.12029), LayerSkip (2404.16710), MatFormer/Gemma 3n, Minitron (nvlabs/minitron), vLLM Semantic Router (naming prior art) — per perplexity review §8.
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

## Added 2026-11-12 from the Perplexity review

Origin: `docs-mom/reviews/2026-11-12-perplexity-review.md` (tasks in its §9, MVP table §5).

### BL-21: Offline G2 proxy + programme kill criterion (before further nested device spend)
- Priority: P1
- Area: distill
- Scope: MVP
- Why: G2 decides whether the thin model is worth existing; M4/M3 infrastructure is currently running ahead of it. Depth-pruned students in the Minitron line need 100B+-token distillation budgets (Mistral-NeMo-Minitron 8B: 380B); our 10/40-layer keep + 6-layer tail is an aggressive cut and 96 seeds won't close it.
- Done when: (1) bf16/HF offline tail distillation on hundreds of millions of mixed-chat tokens (logit KD, teacher = 35B); (2) parity eval vs stock Qwen3.5-2B on a few hundred held-out trivial items, threshold PRE-REGISTERED; (3) thin-decode tok/s measured (if slower than the 2B's 171 tok/s there is no routing case); (4) written programme kill criterion in the gate ledger (e.g. "no 2B parity within N GPU-hours -> re-scope to drafter/cascade, pause M4").
- Refs: perplexity review §4.1; Minitron (nvlabs/minitron)
- Status: check - distill v2 (queued) partially satisfies (1)-(2) at small scale; its gate (>=10/12 coherent) is not parity-vs-2B. This item gates all further device spend per reviewer §9.5.

### BL-22: Host-side shadow model + property tests for pool/table/AdoptMap/SharedView; point-of-use checksum
- Priority: P1
- Area: qa
- Scope: MVP
- Why: every recent M4 bug was a host-side index/lifetime error burned in a scarce device window (ed4334c pairing off-by-one, b55814f audit page-offset, 7618be0 table clobber sampled pre-clobber). The dual-site reader bug (attn tiles vs kv_row) is exactly what a host model would enumerate.
- Done when: `test-kv-pool` gains a randomized (adopt/growth/resume/evict/drop) sequence property test against a host shadow of pool, page table, AdoptMap, SharedView, including EVERY kernel-side resolution site as a modeled reader; plus debug-only point-of-use row checksum during decode behind `TF_MOM_SHARED_VERIFY`.
- Refs: perplexity review §4.4; commits ed4334c, b55814f, 7618be0
- Status: check - none exists; complements the toy-pair harness (engine-in-the-loop) with a pure-host model. To precede any M4-class device spend beyond the current window.
- Status: partial (2026-11-12, df94ab389 on mom/main) - `zig/src/core/kv_pool_sharedview_shadow_test.zig` added to `test-kv-pool`: host shadow of pool (pin/refcount/reissue guard), owner Wholesale reupload from offset 0, adopt-time pins, the view's own skvi, and ALL FOUR kernel reader sites (kv_row, Bf16Tiles kp/vp, near-page fast-path, tf_q35_attn_partial_paged) as independent formula transcriptions. Randomized property test (5 seeds; growth / resume reupload / parent decode) passes; the bite test injects each historical bug (shared-via-pt, near-ids-via-pt, adopt pairing off-by-one, audit zero-window) and each is caught. REMAINING for full BL-22: AdoptMap/evict/drop ops in the op alphabet, and the debug-only point-of-use row checksum under TF_MOM_SHARED_VERIFY.

### BL-23: Reframe S1 conclusion + GDN/memory honesty in docs
- Priority: P2
- Area: docs
- Scope: MVP
- Why: (a) our "translation NOT VIABLE" is broader than measured — Heo et al. (NVIDIA, arXiv 2608.03893) find real linear structure with RoPE stripped, top-k source layers, 128K-token calibration; our wording must be "single-layer ridge map on RoPE-applied keys calibrated on one conversation is not viable". (b) ~2/3 of each kept-state snapshot is fixed-size GDN state (87 MiB heavy); attention-only sharing cannot dedupe it — drop any KV-memory-saving headline; the payoff sentence is "hop cost O(context) -> O(tail)", shown as a curve vs unseen length L.
- Done when: docs/12 and docs/10/13/14 wording corrected; white paper uses the O(context)->O(tail) framing.
- Refs: perplexity review §3.3, §3.5
- Status: check - our own measurements support (b): M3 adopts state pieces byte-identical and attention rows are the minority of snapshot bytes.

### BL-24: Shadow prefill (preemptible cross-lane warming) as the exact control arm
- Priority: P1
- Area: mom-serve
- Scope: MVP
- Why: ShadowPrefill (arXiv 2611.13176) warms the non-chosen model's cache in idle capacity — each model computes its OWN exact KV, preemptible, no correctness exposure; reports 44-47% switch-TTFT cuts. It reaches the entire 47.3% cold ceiling exactly and training-free, so it is simultaneously (i) a likely MVP feature and (ii) the control arm every sharing scheme (M4 included) must beat. Supersedes/absorbs BL-5.
- Done when: idle-capacity prefill of the non-chosen lane using existing request-start states; B5-cold hop-turn TTFT reduced with no B7 active-decode regression; BL-6 verdict re-scored against it.
- Refs: perplexity review §4.2, P4, §9.6
- Status: check - supersedes metaspan BL-5 (same mechanism, stronger framing as control arm). Pre-emptible scheduling design needed (decode bandwidth-bound vs prefill compute-bound on GB10 — B7 says interference low).

### BL-25: Gate ledger
- Priority: P2
- Area: qa
- Scope: MVP
- Why: thresholds have occasionally drifted post-hoc; S1's false-GO came from a leak discovered after the fact. A ledger (id, criterion, threshold, result, n, commit, date, status) makes pre-registration checkable.
- Done when: `docs-mom/GATES.md` lists every gate incl. distill v2's >=10/12, BL-21 parity threshold, BL-6 GO gate, M4 identity gate.
- Refs: perplexity review §4.6
- Status: check - does not exist; cheap.

### BL-26: Rename MoM "lane" before upstreaming; position naming vs vLLM Semantic Router
- Priority: P2
- Area: upstream
- Scope: post-MVP (before PR submission)
- Why: "lane" currently means MoM engines, TensorFold `core/lanes`, and JJ Eaton's Compute Lanes; vLLM-SR already markets "Mixture of Models (MoM)". Upstream reviewers will reject the ambiguity.
- Done when: MoM concept renamed (candidate: "seat"/"tier") across mom-serve/router docs + code, and docs-mom positions the project as "single-process MoM with shared state".
- Refs: perplexity review §4.6
- Status: check - filed against the UPSTREAM-PRs.md re-apply ritual; must land before PR slice 4 (mom-serve).

### BL-27: Cascade-escalation and self-speculative reframes for the nested model
- Priority: P3
- Area: research
- Scope: post-MVP (note)
- Why: adoption is bidirectional (M3): "thin answers first, escalate on low confidence" costs the heavy side almost nothing — the expensive half of every cascade; structurally the thin model is also a LayerSkip-style self-speculative drafter over the shared prefix (compare acceptance vs the MTP drafter).
- Done when: recorded decision note with measurements if pursued.
- Refs: perplexity review §4.3; LayerSkip (2404.16710); vLLM-SR cascade docs
- Status: check - noted only.

### BL-28: Sleep/wake: confirm park's kept-state behaviour; spill-gib prompt cache; wake profiling
- Priority: P2
- Area: mom-serve
- Scope: MVP
- Why: if park drops kept states, every post-wake turn is a cold re-prefill after a 12-14 s wake; upstream CUDA supports `--spill-gib` for Qwen3.6-35B-A3B (#1413/#1471; M5 Ultra datapoint: 32k conversation resumed 1.3 s vs 4.9 s). Wake at 11.9-14.5 s for ~21.8 GiB (~1.6-1.9 GB/s) is probably not disk-bound — profile phases.
- Done when: (1) B5 replay across forced sleep establishes whether `resumed at` is non-zero post-wake today; (2) if dropped, spill enabled and byte-identical post-wake reply proven; (3) per-phase wake timing logged.
- Refs: perplexity review §5 P1/P2, §9.3; PROGRESS B4
- Status: check - reviewer explicitly flags park behaviour as unconfirmed; step 1 is a cheap factual probe.
