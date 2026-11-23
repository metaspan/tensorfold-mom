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
- Refs: `docs-mom/docs/09`, PROGRESS B6, `docs-mom/gates/20261123-spike2-rerun.md`
- Status: open (partial, 23 Nov 2026) — B5 rerun DONE (arm B: 85.3%/85.6% cold,
  99.0%/99.0% warm prefill saved, reuse 100%); interim guard landed
  (`experiments/mom_mvp_serve.sh` boot assertion + `tests/mom/test_cache.py`).
  Remaining: mom-serve sets/refuses the mode in-process + per-lane startup
  assertion covered by a host test.

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
- Status: BL-6 round 2 ran (experiments/bl6_gate.py, results/bl6/bl6-20261112-1637.json). TTFT ratios copy=0.183 view=0.190 looked like a huge win but were VOID: the adoption arms returned EMPTY completions (`NonfiniteQwenLogits`, HTTP 500) — 0 decoded tokens, so their TTFT measured failure, not adoption. Stock decode 142.1 tok/s vs pre-registered ≥154 → the decode condition genuinely fails; verdict NO-GO stands even before BL-29. Round 1 (one nested session replayed per arm) is superseded: it measured nothing (thin's own in-boot cache flattened every TTFT); bl6_gate.py now warms the parent per salted variant and times the cold thin. See BL-29.
- Status: BL-6 round 3 ran 2026-11-13 post-BL-29-fix (results/bl6/bl6-20261113-0904.json). Adoption arms REAL: 96 tokens × 3 variants × both arms, adopted_lines 18/18 (non-vacuous). TTFT ratios copy=0.405 view=0.416 — PASS (≤0.85), first valid TTFT measurement. Decode 141.9 tok/s stock (copy 142.8, view 143.3 mean) vs ≥154 → FAIL; thresholds NOT moved → verdict NO-GO stands, now attributable purely to the stock decode shortfall (uniform across arms ⇒ intrinsic to thin decode, not to adoption; ~0.92× the 0.90×171 bar). Open follow-up: root-cause the ~142 tok/s ceiling (KV dtype/context config, MTP drafts, launch config) before any round 4.

### BL-29: Adopted STATE BLOCK carries dead-head conv-tail NaNs on the production pair (NonfiniteQwenLogits) — RESOLVED 2026-11-13
- Priority: P1 (was: blocked any 35B adoption measurement; BL-6 TTFT arm and distill v2 sat behind it)
- Found: 2026-11-12, BL-6 round 2 + repro (experiments/repro_adopt_cold.py). Cold-thin adoption on parent (qwen36-35b-a3b-nvfp4) + qwen36-nested-16l aborted `reason=error (NonfiniteQwenLogits)` on the device-backed copy path (production default on GB10 integrated). (Was filed as "BL-26: device copy writes poison bytes" — superseded below; renumbered because BL-26 was already taken by the lane-rename item.)
- Narrowed 2026-11-12 evening (three device windows, 4e56ef1dd instrumentation):
  - adoptVerify is now DEVICE-capable (it used to self-skip on device copies — the reason this had zero byte visibility): **17/17 device pieces match the copy** — the copy bytes land correctly; save/restore addressing is fine. The "device copy writes garbage" hypothesis is DEAD.
  - The old audit's "EVERY piece DIFFERS at byte 0" was a FALSE ALARM: it compared the adopted bytes against the thin engine's OWN cold save (wrong baseline; tail regions legitimately differ) and self-skipped on device copies anyway.
  - **Bisection: `TF_MOM_ADOPT_ZERO_STATE=1` (zero the adopted state block) → device adoption decodes 48 real tokens.** The fault was confined to the state block; KV pages, shared-view resolution and the kernel read-path exonerated.
- ROOT CAUSE (2026-11-13, fix 918da4834): the adopt-time sanitizer scanned INTERLEAVED [rec|conv] regions while `cache.state` is PLANES (`cuda_cache.zig` Seg view: `rec = state.ptr`, `conv = state.ptr + deltas*recBytes`). It cleaned bytes no kernel reads and never once looked at the real conv planes, which carry **39–55 bf16 non-finite per delta — a dead head's stale conv tail** (the documented hazard: the parent's kernels never read dead-head tails; the nested CLONE does on resume). The old bf16 filter additionally required a nonzero mantissa, letting bf16 inf through. All earlier "finite-but-wrong rec values / 8-region save" readings were artifacts of the same misaligned scan (windows straddling planes read bf16 conv bytes as f32 NaNs).
- FIX: plane-correct scan of BOTH planes (f32 rec via isFinite, bf16 conv via exponent-all-ones incl. inf) + write-back. Verified on the 35B pair, two consecutive cold device boots under `TF_MOM_SHARED_VIEW=1`: 48 tokens generated (finish=length, cached 101/82), zero-count 1:1 with parent per delta (copy faithful), 545–572 conv bf16 zeroed per adoption. Host suites (mom-host/cache-host/kv-pool/adopt-host) green. Bisection levers kept: `TF_MOM_ADOPT_ZERO_REC/_ZERO_CONV/_ZERO_REC_FROM=N/_ZERO_CONV_FROM=N/_ZERO_STATE`; analyzer: `experiments/analyze_planes.py`.
- Done when: BL-6 round 3 completes with real adopted completions in B/C (usage.completion_tokens > 0, finite) without ZERO_STATE; host shadow covers the device-copy leg. ← adoption side now unblocked; BL-6 round 3 is the remaining measurement (its decode-arm 142<154 shortfall is a separate, still-open NO-GO).

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
- Status 2026-11-13: t1 COHERENCE GATE FAIL 0/12 (distill_gate --engine hf amendment — the native engine categorically refuses bf16 MoE, cuda_geometry.zig:290; items/bar/rules unchanged, pre-run amendment documented in the script docstring). Replies are G2-floor salad ('2', 'esesese' loops, multilingual fragments), 0 template leaks; weights load 279/279 (export structurally sound). Cause confirmed as scale: DISTILL.json shows 175 rows (~20K tokens) x 2 epochs @ lr 1e-5 on 104 tail tensors — far below the recipe's stated need. This is the measured instantiation of "175-row corpus can't reach coherence". Scale-up (this item) is the prerequisite for BL-30 head training (the head must train on the GATED student's hidden, or it retrains every tail epoch).
- Status 2026-11-13 (t2): FAIL 3/12 at 2,492 clean rows x 4 epochs (eval held 2,492->249 disjoint). Trend is right (0/12 -> 3/12; gibberish -> near-correct forms with wrong values + repetition tails) and frozen prefix stayed 176/176 byte-identical to the warm-start twin (adoption digest invariant preserved; 87/102 tail tensors changed). Decisive signal: EVAL loss plateau (2.42/2.11/2.03/1.99) while train hit 0.82 -> data-coverage bound, not train-depth bound: more epochs on this corpus will not move eval. Next iteration = corpus scale (target ~10K rows via distill_collect --gen 9900) then t3; gate unchanged (>=10/12).
- Status 2026-11-14 (t3): FAIL 3/12 at 12,226 clean rows x 3 epochs (eval 11,004/1,222 disjoint; eval loss 1.88/1.76/1.72 — NO plateau, unlike t2; frozen 176/176 byte-identical again). 4.9x data moved the loss curve but ZERO gate movement -> corpus sampling teaches the teacher's shape, not its answers (misses are wrong values: 23+19->22, 6x7->60, 10km->10.21mi; teacher spot-verified on these via collector probes — scored teacher-gate run queued, see calibration pre-registration below). User decision: build offline logit-KD (options declined: 50k re-scale, thinking-mix-only, write-up).
- AMENDMENT 2026-11-14 (PRE-RUN, KD hyperparameters fixed now per S1 rule; gate 12 items/10 bar UNCHANGED): distill_kd_prep.py computes teacher top-8 logits per position offline (teacher = official Qwen/Qwen3.6-35B-A3B-FP8 from HF cache — same base the NVFP4 collector teacher was quantized from, config-verified qwen3_5_moe_text/2048/40L/256e; tokenization verbatim-identical to trainer incl truncation 512; teacher row t predicts token t+1, trainer aligns shift index s -> logits[s] -> teacher row s). Trainer --kd adds loss = (1-beta)*CE + beta*T^2*KL with beta=0.5, T=1.0, top-k=8 (both distributions renormalized over the top-8 set; standard top-k KD). Two-pass by design: teacher pass and student pass never coexist in RAM (128 GB unified budget). t4 trains on the SAME 10k corpus + kd-10k.pt. Implementation addendum (pre-run, harness only): the finegrained-fp8 triton kernel from the hub fails on this stack (triton_kernels 'Unknown recipe'), so the teacher is a bf16 dequantized copy of the identical official snapshot (dequant_fp8_teacher.py; e4m3 + 128x128-block bf16 weight_scale_inv semantics per transformers finegrained_fp8) at qwen36-35b-a3b-fp8-deq. Dependency pin: kernels==0.12.3 + hub 1.x (0.18 breaks transformers hub_kernels import; <=0.11 force hub<1.0).
- Status 2026-11-14 (KD loader saga, resolved): teacher load needed a purpose-built path on GB10 unified RAM — from_pretrained's MoE custom conversion holds unfused+fused simultaneously (sampler: anon 104 GB -> system OOM), and a CPU-built model never returns its 67 GB highwater (torch CPU caching allocator), so even layer-by-layer .to(cuda) died (rss 51 GB + GPU 66 GB > 121 GB). Working design (committed): instantiate from_config directly under torch.device('cuda'), then per-shard load_file (1.6 GB transient) with dense 1:1 copies and expert tensors slice-copied into fused GPU params (gate_up_proj rows [0:I]=gate,[I:2I]=up per modeling chunk(2,-1); down_proj direct); skip mtp.*+model.visual.* (unused by text forward); safetensors fadvise wrapper as page-cache hygiene. Dry-run calibration: teacher top-1 vs corpus next-token 61/126 (48%) on sampled-text rows — healthy for T>0-sampled corpus (broken weights would show ~random). Full 12,226-row prep -> results/distill/kd-10k.pt; plan-of-record unchanged (t4 = same corpus + KL beta=0.5,T=1.0,k=8; gate 12 items/10 bar).
- Review adjudication 2026-11-14 (reviews filed docs-mom/reviews/2026.11.14-{perplexity,grok}-two-day-review.md):
  - GOVERNANCE AMENDMENT (owner-approved, Derek, 2026-11-14): BL-6's pre-registered consequence "NO-GO -> park M4 + distillation" is amended — owner decision keeps distillation + MTP (route b) alive because (i) BL-6T showed decode is not-config-addressable, (ii) MTP acceptance bar is far lower than previously modeled (see pre-registration below), (iii) M4 toy+device gates are green independent of student quality. BL-12's "ONLY if BL-6 GO" restriction is superseded by this amendment.
  - BL-21 KILL CRITERION (pre-registered 2026-11-14, per BL-21 requirement 4): if neither t4 (logit-KD as registered) nor one ladder step (5b corpus-diversity) reaches >=10/12 within a combined budget of 20 GPU-hours from 2026-11-14, the nested-student programme is re-scoped to drafter/cascade (BL-27) and M4 student dependency pauses (M4 core ships for the trained-pair case only).
  - MTP ACCEPTANCE PRE-REGISTRATION (adopted from perplexity review 2.1): back-out from our own measurements says the decode bar needs tokens/step >= 1.206. Measured no-MTP step 7.05 ms, MTP step 7.83 ms at 1/40 acceptance -> required trained-head acceptance ~= 21%; at the 1/40 CI upper edge (~13%) ~= 33%. Pre-registered reads on t3 (head trained on t3 is a valid DISTRIBUTION preview; final head retrains on the gated student): acceptance <15% = route (b) NO-GO (head can't clear bar on this student class); 15-35% = informative-borderline, proceed with retrain-on-gated-student only if t4 passes; >=35% = decode bar plausible, proceed. Threshold on tok/s remains the BL-6 bar >=154 on the A_stock line — unchanged.
  - MTP BAR VALIDATION (2026-11-14, owner directive: don't trust reviewer algebra): engine source settles it — `default_depth = 3` (cuda_mtp.zig:22, env TF_MTP_DRAFTS), done-line stats are `rounds= accepted=X/Y` with Y=DRAFTED (chat_generation.zig:354), so the reviewer's tokens/step = 1+a model conflated accepted/drafted with accepted/rounds. Original zero-shot log (mtp-serve3.log): rounds=94-95 for 96 tokens, drafted only 10-17/rounds-chain -> the planner THROTTLES a never-accepted drafter (limit_shared_drafts, cuda_mtp.zig:496); observed step at throttled drafting 7.51 ms (vs 7.05 no-MTP). Full-depth cost must be MEASURED (forced TF_MTP_DRAFTS=3, thinking-off, n>=40 rounds) — with 3-token chains the arithmetic per draft-token acceptance acc is tokens/round = 1 + 3*acc, making the required acc single-digit-to-low-teens % IF the planner lets drafts flow at full width. Pre-registration refined (PRE-RUN, zero trained-head measurements exist yet): the DECISIVE number is measured tok/s vs the BL-6 bar 154 on A_stock; acceptance stats (rounds, drafted/rounds, accepted/drafted) are diagnostics, and the <15/15-35/>=35 bands are hereby scoped to accepted/drafted. Probe must set enable_thinking:false (thin lane defaults thinking ON; confounds tok/s).
  - DIAGNOSTIC PRE-REGISTRATION (perplexity 2.2, run before interpreting t4): (D1) train-set accuracy: score t3 on 200 corpus rows sampled from the miss categories (arithmetic/unit/spelling); fail-on-train => capacity/optimisation, not distribution — deprioritises ladder 5b/5c. (D2) early-exit baseline: heavy final norm + lm_head on the frozen 10-layer prefix output, score the 12 gate items; bounds answer information at the split point.
  - TEACHER CALIBRATION PRE-REGISTRATION: scored teacher-gate on the dequantised bf16 teacher (greedy, 12 items) must score >=11/12; below that the loader fidelity or dequant path is suspect, not the student. Corpus provenance for the 48% top-1 calibration stat: completions collected at temperature 0.5 (thinking rows 0.8) from the native-engine teacher, top-1 agreement on sampled tokens, n=126 dry-run.
  - Teacher-gate run 1 (2026-11-14): 10/12 -> below bar; BOTH misses are 48-token truncations of visibly-correct teacher outputs (grammar item: correct tense analysis cut before the corrected sentence; reverse-'world' item: correct letter-by-letter walk d-l-r... cut before the answer) — the 48-token cap encodes the STUDENT's brevity requirement, not a teacher-knowledge test; zero signs of loader corruption (which degrades all items). PRE-RUN AMENDMENT (harness only, before rerun): teacher calibration budget max_new_tokens 48->128; bar 11/12 UNCHANGED; single rerun. If run 2 <11/12, loader/dequant suspicion stands per the original consequence.
  - Teacher-gate claim correction: the two-day review's "parent answers all 12 correctly" was spot-check inference, not a scored run — corrected in-place in the review doc; scored run = the calibration item above.
  - k-policy: t4 trains at k=8 exactly as registered from the existing kd-10k.pt. Perplexity's k=32+residual-bucket prep is adopted ONLY for ladder 5a (needs logsumexp/coverage logging, to be added when 5a is scheduled). Grok's k=16/32 ablation on a 500-row subset: deferred — t4 as registered first (S1: no config fishing); 5a provides the k signal.
  - Teacher choice: keep dequantised bf16 teacher (perplexity concurs; "prefer native NVFP4" from grok is a format slip — the native teacher is FP8; its HF finegrained_fp8 triton path is broken on this stack, dequant is linear e4m3*scale -> bf16).
  - NOTED, not MVP (general TensorFold per boundary): engine logprobs for qwen3_5 CUDA, native bf16-MoE support, Nsight decode profiling of the thin lane, tail quantisation (needs the open quantiser item).
  - CORRECTION 2026-11-15 (checkpoint beats spec): the "shared-expert-only dense tail, ~30 MiB/layer" (docs/13 pilot spec, repeated in the two-day review and by reviewers) does NOT match what nested_extract.py built: qwen36-nested-16l-bf16 has FULL routed experts in tail layers (layers.11.mlp.experts.gate_up_proj [256,1024,2048] verified on disk) — keep() byte-copied parent layers 10-15 wholesale, and distill_train trained them (5.5B trainable). The real defect is DEPTH-POSITION MISMATCH: the heavy's lm_head+final norm expects the post-layer-39 residual stream; our tail is parent layers 10-15 pretending to be the last six layers. That explains twin 0/12, D1 underfit, D2 0/12 (layer-10 stream + parent head = 0), and t1-t4 shape-only learning. Reviewer capacity framings are void.
  - 5e LAST-6 BYTE-COPY (pre-registered 2026-11-15, Gromov 2403.17887 keep-first-and-last + Draft&Verify 2309.08168 precedent): thin = parent layers 0-9 + parent layers 34-39 (renumbered 10-15), final norm + lm_head byte-copied, NVFP4, zero training. Serves natively. PASS BAR: native gate >=10/12 -> the nested recipe pivots to last-6 tails (trained healing optional, Minitron-style 2408.11796); <10/12 -> record as capacity/structure verdict for the trained-tail branch. nested_extract gets --tail-from for this.
    RESULT 2026-11-15: byte-verify CLEAN (18,616 tail tensors == parent 34-39, layer_types/quant-table re-key OK), native boot fine (10.1 GiB, 8.8 s load), GATE 1/12 FAIL. Zero-training layer surgery does not work: the 24 deleted layers are not paper-over-able (hidden-state distribution mismatch at the seam; Gromov's own recipe REQUIRES post-pruning healing). Combined with twin 0/12 and D2 0/12: every zero-training reconstruction fails; the trained tail must be trained FROM a compatible init. Natural next rung (5g, NOT yet run, owner decision — BL-21 budget ~14/20 GPU-h spent): same trainer/corpus/gate as t1-t4, but tail warm-start = parent layers 34-39 (depth-position-compatible) instead of 10-15. Predicted by the corrected model to be the best remaining trained-tail shot.
  - 5g RETRAIN (Derek GO 2026-11-23, pre-registered before GPU spend): previous 5g
    (built 2026-11-20/21, `qwen36-nested-16l-5g`) plateaued; the rung-acceptance model is
    RETRAIN FROM THE ACCEPTED STEP, NOT THE PLATEAU END. Fixed config (no fishing):
    `MOM_STUDENT=/home/derek/models/tensorfold-mom/qwen36-nested-16l-last6-bf16
    distill_train.py --data ~/.hermes/sp4-transfer/mom-repo/experiments/results/distill/rollouts-10k-filtered.jsonl
    --epochs 3 --lr 1e-5 --seed 7 --save-checkpoints` (same trainer/corpus/gate as t1-t4 and
    the prior 5g; venv-tensorfold-mom, stack pins per manifest). `--save-checkpoints`
    (added this run, harness-only) writes `<out>-ep{1,2,3}` + `<out>-epochs.json`.
    Adoption rule: gate EVERY epoch checkpoint with `distill_gate.py --engine hf` (bar
    >=10/12, gate_qa.json fixed); accepted step = HIGHEST-step checkpoint passing; if none
    passes -> 5g FAILS the rung -> GKD rung 5f with owner awareness. BL-21 GPU-h ledger
    records the spend (prior ~14/20; this run est 4-5 h, cap 5 h wall via timeout).
    Run files: experiments/results/night/5g-retrain-20261123/.
  - 5f GKD ON-POLICY KD (pre-registered, conditional, GKD 2306.13649): if 5e does not clear, final trained-tail attempt = student(t4)-rollout corpus + teacher top-8 on those rows + same trainer (fixes train/inference distribution mismatch that t1-t4 suffered). Runs ONLY with owner awareness; counts against the BL-21 kill budget.
  - Adopted follow-ups (queued, non-blocking): MTP head semantics docs page w/ file:line refs + host test comparing mtp_train forward vs one engine draft step on fixed inputs; bf16-prefix provenance check vs deployed NVFP4 (train/deploy skew, perplexity 2.6); BL-8-style adopted-vs-cold byte-identity assertion added to next BL-6 round; B5-long TTFT sweep at L=2k/8k/16k/32k with adoption on the untrained nested ckpt (core-hypothesis evidence, quality-independent); "intrinsic" reworded to "not config-addressable".

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
- Status 2026-11-13: DECISION (user): route (b) — build an MTP drafter head for the nested checkpoint. Zero-shot control: parent's own MTP head copied verbatim beside the thin checkpoint (experiments/mtp_merge.py -> qwen36-nested-16l-mtp), serve --drafter, drafts=on, accepted 1/40 (~2.5%) and decode 130.9 tok/s (slower than stock 141.9) — the parent-trained head cannot draft on the student tail's hidden; training the head on the student is required. Head training tracked as BL-30.

### BL-30: MTP drafter head trained on the nested student (unblocks the BL-6 decode bar / routing case)
- Priority: P1
- Area: distill
- Scope: MVP
- Why: thin decode 141.9 (stock) and 141.4 (BL-6T tuned) sit under the pre-registered >=154 bar; BL-6T showed the ceiling is intrinsic without a drafter. The engine already serves the MoE target's own MTP layer as a drafter (cuda_drafter_open.openMtp, tap = target's last layer, affine-4 quantized tables at load); the thin checkpoint just lacks mtp.* tensors. Zero-shot parent head accepted ~2.5% (BL-21 status above), so the head must be trained against the gated student's hidden.
- Done when: (1) PyTorch replica of the 19-tensor MTP layer (fc 2048x4096 from pre-normed [embedding;hidden], 1 GQA+MoE decoder layer, norm) trained with teacher-forced CE on student-captured pre-norm residual + next-token embeddings over the distill corpus + on-policy rollouts; (2) acceptance-rate measurement in serve: mean tok/s >= the BL-6 pre-registered decode bar 154 on the same 3 prompts x 3 variants; TTFT guard unchanged; (3) head bytes shipped beside the student via experiments/mtp_merge.py (NEVER requantized after training; the engine quantizes draft tables at load itself).
- Refs: docs-mom/gates/20261113-bl6t-tuned-decode.md (intrinsic-ceiling finding); zig/src/families/qwen3_5/cuda_mtp.zig (interface: hidden tap layers-1, follow ids, TF_MTP_DRAFTS depth); HF transformers 5.8.1 has NO qwen3_5_moe MTP module — replica from the parent's tensor shapes + engine absorb semantics.
- Status: open. Zero-shot control done (1/40 accepted); trainer not yet written. Depends on BL-12 scale-up producing a gate-PASSING student (t1 gated 0/12 FAIL 2026-11-13, undertrained): the head's inputs are the student's hidden states, so head training runs on the final gated tail only.

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
