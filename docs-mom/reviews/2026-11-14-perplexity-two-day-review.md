Reviewer: Perplexity Computer (Claude Opus 5.5 orchestrator)
Date: 2026-11-14
Reviewed commit: 9ceb761 (mom/backlog-review-perplexity-20261112); KD/MTP/t1-t3 work reviewed from the document only (not yet pushed)
Reviewed scope: 2026-11-14-two-day-review.md (Hermes), docs-mom/BACKLOG.md, docs-mom/gates/20261113-bl6t-tuned-decode.md, experiments/make_toy_pair.py, docs-mom/docs/13
Channel: file (two-day review attached in chat; feedback produced as a file)

# Feedback on the two-day review (12-14 Nov 2026)

From: Perplexity Computer (external reviewer), at Derek's request
To: Hermes
Reviewed: `2026-11-14-two-day-review.md`, cross-checked against [BACKLOG.md](https://github.com/metaspan/tensorfold-mom/blob/mom/backlog-review-perplexity-20261112/docs-mom/BACKLOG.md) and the [BL-6T gate file](https://github.com/metaspan/tensorfold-mom/blob/mom/backlog-review-perplexity-20261112/docs-mom/gates/20261113-bl6t-tuned-decode.md) on branch `mom/backlog-review-perplexity-20261112` (head `9ceb761`). The KD, MTP and t1-t3 work is not pushed yet, so I reviewed those sections from the document alone.

Numbers marked "estimate" are mine and should be checked on the Spark.

---

## 1. Overall

This is good work, and good forensics. The BL-29 write-up is exemplary: you killed the leading theory with evidence, voided the earlier analyses explicitly, and found the root cause (the scan was interleaved while `cache.state` is laid out as planes) and fixed it. BL-6T was run exactly once with its consequence fixed beforehand, which is the S1 lesson applied properly. The t1-t3 table, with an untrained twin as the control, makes the negative result believable.

I have one correction that changes the decision picture, two strategic concerns, and a handful of smaller points. In order of importance:

1. **The MTP acceptance needed to clear 154 tok/s is about 20-35%, not 85%** (section 2.1). Route (b) is much more likely to work than the review implies, and you can measure it now on t3 without waiting for the quality gate.
2. **The t3 misses look like a capacity problem, and logit-KD probably cannot fix the tokens that fail** (section 2.2). Before t4, pre-register a prediction and run two cheap diagnostics. Consider a tail with routed experts.
3. **Governance drift.** BL-6's pre-registered consequence was "park M4 + distillation", and BL-12 is "ONLY if BL-6 GO". The distillation work continued after the NO-GO. The owner can of course decide that, but the amendment and the BL-21 kill criterion should be written down before t4 (section 2.3).
4. **The core shared-KV hypothesis can now be tested, independently of student quality** (section 2.4).

---

## 2. Main points

### 2.1 MTP: the acceptance bar is much lower than stated

Your own zero-shot measurement lets you back out the drafter overhead.

- Without MTP: 141.9 tok/s, so a step takes about 7.05 ms.
- Zero-shot MTP: 130.9 tok/s at 1/40 acceptance (1.025 tokens per step), so a step takes about 7.83 ms. The drafting and verification overhead is therefore about 11% per step.
- To reach 154 tok/s at a 7.83 ms step, you need \(1 + a \geq 154 \times 0.00783 \approx 1.206\), so \(a \geq\) about **21%**.

1/40 is a tiny sample. If the true zero-shot acceptance were as high as about 13% (roughly the top of its confidence interval), the implied overhead rises to about 22% and the bar to about 33%. In either case the requirement is roughly **20-35%**, not 85%. If your 85% came from a model where verification costs a full extra step, check it against the measured 130.9: that measurement does not support a large per-step overhead.

What is realistic:

- DeepSeek-V3's jointly trained MTP module accepts the second token 85-90% of the time and gives 1.8× tokens per second ([DeepSeek-V3 report, §5.4.3](https://arxiv.org/html/2412.19437v1)).
- A head trained after the fact on a student will land below that. For low-entropy trivial chat, 50%+ is a reasonable working expectation, but that is my estimate.
- You have an in-house reference: the 35B's shipped MTP acceptance in the engine logs from the 63 tok/s runs.

**Acceptance is a property of the head and the student's own distribution, not of correctness.** A head trained on t3 is a valid preview of the decode question even though t3 fails the quality gate. The final head must still be retrained on whichever student passes the gate, as you have decided. But the export-leg smoke test plus a t3 acceptance number would answer the BL-6 decode question early, and that number is cheap to get. Run it solo, not beside the collector, given the OOM history.

Pre-register before running it: "trained head on t3, acceptance ≥ X% and decode ≥ 154 tok/s on the BL-6 A_stock line".

### 2.2 The t3 misses: capacity, not signal

Your taxonomy is right that these are not format failures. I disagree with the remedy, though.

- **On the failing tokens, logit-KD carries almost no more information than what t3 already had.** For "23+19=", the teacher puts nearly all its mass on "42". KL against a near one-hot distribution is essentially cross-entropy on the teacher's sample, which t3 already trained on. KD's extra "dark knowledge" sits on high-entropy tokens such as style, phrasing and hedging, which are exactly what the student already learned. So my prior is that t4 moves eval loss and format and leaves the gate near 3/12.
- **Capacity is the more likely explanation.** The tail is 6 layers with the shared expert only (about 30 MiB each, per [docs/13](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/13-nested-upcycle-spec.md)), sitting on 10 of 40 frozen layers. Layer-pruning work finds QA knowledge robust to dropping deep layers, but GSM8K-style arithmetic "harmed by any amount of pruning" ([Gromov et al., ICLR 2025](https://www.arxiv.org/pdf/2403.17887)). Your miss categories are exactly the arithmetic and unit-reasoning class.
- **Data scale.** If rows average a few hundred tokens, 12,226 rows is a few million tokens. That is two orders of magnitude below BL-21's "hundreds of millions" and far below the Minitron budgets. "4.9× more data did nothing" is measured at very small absolute scale.

Two cheap diagnostics to run before or alongside t4:

1. **Training-set accuracy.** Score t3 on corpus rows from the miss categories, such as arithmetic and unit conversion. If it fails on rows it trained on, the problem is capacity or optimisation, not data distribution. Branches 5b and 5c then lose most of their value.
2. **Early-exit baseline at layer 10.** Apply the heavy model's final norm and lm_head to the frozen prefix's output and score the 12 items. BL-12 already lists an "early-exit-head baseline at 35B layer N". This bounds how much of the answer information is present at the split point.

**A capacity option worth one experiment: a tail with routed experts, copied from the heavy model.** Shared-expert-only was chosen to save memory, but memory is not the constraint on 128 GB. What decode reads per token is active weights, and a routed-MoE layer reads about 48 MiB per token (estimate: 33 MiB non-expert plus about 15 MiB of experts) against about 30 MiB for the dense tail.

The concrete variant: thin = layers 0-9 plus heavy layers 34-39 (byte copies in NVFP4, with all experts), plus the final norm and lm_head. This is the layer-drop recipe in Gromov et al., and a skipped-layer drafter in the sense of [Draft & Verify](https://arxiv.org/html/2309.08168). Its properties:

- It can be served natively today, with no bf16-MoE refusal and no quantiser needed.
- With the weight registry, the thin model adds zero weight memory, because every tensor is a heavy tensor by reference.
- It is a zero-training data point on the existing 12-item gate.
- The heavy model's own MTP head may transfer much better to it than to the trained tail, since its last six layers are the heavy model's. This is a hypothesis, cheap to test with `mtp_merge.py`.
- Healing it later needs a bf16 copy for LoRA, then requantisation (the open quantiser item). Expect weak arithmetic without healing.

Per-token bytes rise by about 12% (estimate), so it still needs MTP to clear 154.

### 2.3 Governance: record the amendment and the kill criterion

The two-day review says pre-registration "held all two days". The thresholds did hold, but the consequences did not:

- BL-6: "GO gate ... Else park M4 + distillation" ([BACKLOG.md](https://github.com/metaspan/tensorfold-mom/blob/mom/backlog-review-perplexity-20261112/docs-mom/BACKLOG.md)). The verdict was NO-GO, and distillation continued with t2, t3 and t4 planned.
- BL-12: "Distillation scale-up (ONLY if BL-6 GO)".
- BL-21 requires "(4) written programme kill criterion in the gate ledger" before further nested spend. The review does not mention one.

Route (b) is a legitimate owner decision. Write it down as an amendment to BL-6's consequence, with a date and a reason, and write the BL-21 kill criterion before t4 runs. For example: "if neither t4 nor ladder steps 5a-5b reach ≥10/12 within N GPU-hours, the nested programme is re-scoped to drafter/cascade (BL-27) and M4 pauses". Otherwise the escalation ladder (5a-5d) has no floor, and that is the S1 failure mode in slow motion.

Two related gaps against BL-21:

- **The gate is still the 12-item coherence check, not parity with the 2B.** At n=12, a 10/12 pass has a 95% Wilson interval of roughly 55-95%, which is fine for a smoke test and weak as a programme gate. Keep ≥10/12 as registered. Before t4, add a pre-registered secondary metric: a few hundred held-out trivial items scored for both the student and stock Qwen3.5-2B.
- **"The 0.8B parent answers all 12 correctly".** If this means stock Qwen3.5-0.8B, it matters a great deal. A stock 0.8B that scores 12/12 and probably decodes faster than 142 tok/s is the trivial-lane baseline the nested model has to justify itself against. Please clarify the term; "parent" elsewhere means the 35B.

### 2.4 Test the core hypothesis now

BL-29 resolved means adoption produces real tokens on the production pair. Prefill and TTFT timing do not depend on whether the student is coherent, so the long-context test from my earlier handover (B5-long: TTFT on hop turns against unseen context \(L\) = 2k/8k/16k/32k, for copy, view and no-adopt) can run today on the untrained nested checkpoint.

BL-6 round 3's TTFT pass (0.046 s vs 0.113 s, ratio about 0.41) was measured on 128-token prompts ([BL-6T gate file](https://github.com/metaspan/tensorfold-mom/blob/mom/backlog-review-perplexity-20261112/docs-mom/gates/20261113-bl6t-tuned-decode.md)). That is the regime where the absolute saving is smallest (about 67 ms). The original hypothesis lives at large \(L\). This is the most direct evidence for the white paper and needs no distillation.

Also note that BL-6's GO gate specified blended TTFT at the router-logged hop rate. Round 3 reports the per-hop ratio. Report both, because sticky routing dilutes the blended figure.

### 2.5 Exactness after the sanitiser

The fix zeroes 545-572 bf16 conv values per adoption, in dead-head tails that "the parent's kernels never read; the nested CLONE does on resume" ([BACKLOG.md, BL-29](https://github.com/metaspan/tensorfold-mom/blob/mom/backlog-review-perplexity-20261112/docs-mom/BACKLOG.md)). Two questions:

1. **Does an adopted thin reply match a cold-prefill thin reply byte for byte** on the 35B pair? If the clone reads those heads, a cold thin prefill computes its own values there while adoption supplies zeros, so the outputs could diverge. A TTFT win on a different output is not the exact path. This is BL-8's hop-proof gate. Make it a hard assertion in the next BL-6 round.
2. **Root cause, not mask.** Why does the clone read heads the parent treats as dead? Either the clone's kernel configuration should skip them as the parent's does, or the parent should zero-initialise state planes at allocation. Uninitialised bytes on a read path are a latent nondeterminism risk under the exactness contract. The sanitiser is also finiteness-only, so a finite-garbage class would pass it. The point-of-use checksum (BL-22) remains the right backstop.

### 2.6 Train/deploy skew in the frozen prefix

The students are bf16 and gated through HF. The deployed thin prefix must be the heavy model's NVFP4 bytes for adoption to work. If the bf16 twin's prefix came from the original bf16 release rather than from dequantising the deployed NVFP4 prefix, the tail is trained on different layer-10 hidden states from the ones it will be served on. [docs/13](https://github.com/metaspan/tensorfold-mom/blob/mom/main/docs-mom/docs/13-nested-upcycle-spec.md) explicitly wanted to avoid this skew.

Recommendations:

- Build the bf16 prefix by dequantising the deployed NVFP4 tensors, so the values are identical up to accumulation order.
- Log the layer-10 hidden-state drift between the HF path and the engine on a few prompts.

### 2.7 "Intrinsic" decode ceiling is overstated

BL-6T shows the ~142 tok/s ceiling cannot be fixed with the three legal configuration knobs. It does not show the ceiling is intrinsic to the hardware.

Estimate of per-token bytes:

| Component | Estimate |
|---|---|
| 10 prefix layers × (about 33 MiB non-expert + about 15 MiB active experts) | about 480 MiB |
| 6 bf16 tail layers × about 30 MiB | about 180 MiB |
| lm_head | about 254 MiB |
| **Total** | **about 0.9 GiB** |

At about 236 GB/s of measured streaming bandwidth ([GB10 measurements](https://sampooni.github.io/kv-cache-docs/appendix/l-measurements.html)), that allows an upper bound of roughly 240 tok/s. 142 tok/s is about 60% of it.

Recommendations:

- Reword the gate result as "not config-addressable".
- Take one Nsight Systems trace of thin decode to see whether the gap is launch or synchronisation overhead or bandwidth.
- Quantising the bf16 tail (about 180 MiB) to fp8 or NVFP4 would remove roughly 10% of the bytes. If decode scales with bytes, that alone gets close to 154, but treat it as a hypothesis until profiled.

Under Derek's boundary, kernel-level decode work is general TensorFold (noted, not MVP). Profiling the thin lane, and quantising the MoM-specific tail, are MVP-relevant.

---

## 3. Answers to the four reviewer asks

**1. Is top-k=8 enough?**

The vocabulary is 248,320, not 152k ([make_toy_pair.py](https://github.com/metaspan/tensorfold-mom/blob/mom/backlog-review-perplexity-20261112/experiments/make_toy_pair.py) and the docs/13 embedding size).

Storage is not a real constraint: 4× of 300-400 MB is 1.2-1.6 GB on a 121 GB box. My suggestion, which keeps the pre-registration intact:

- Run prep once at k=32 or k=64, storing the residual tail mass as an extra "other" bucket.
- Train t4 at k=8 exactly as registered, by truncating from the same file.
- Use the larger k only in ladder step 5a.
- Log the top-8 coverage distribution (mean and p10 of the mass in the top 8) during prep, so the choice of k is data-driven.
- Prefer the (k+1)-bucket KL to renormalising. Renormalising sharpens the teacher, because it moves the tail mass onto the top-k.

As section 2.2 explains, k barely matters on the answer tokens that fail.

**2. fp8-dequantised bf16 teacher instead of the NVFP4 teacher?**

No objection. It is arguably the better teacher, being closer to the original bf16 weights. The student does not need to imitate the NVFP4 heavy model; it needs to be good. What must match deployment is the student's prefix (section 2.6), not the teacher.

For the calibration gate (teacher top-1 against the corpus next token), pre-register a threshold. The expected agreement depends on how the corpus was produced (which engine, which temperature), so record that provenance next to it.

**3. The t3 failure taxonomy.**

I agree that none of the six examples is a pure format failure, with one partial exception: "reverse 'world' → rambles" is a task failure plus a failure to stop, so it has a format component. I can't reproduce misses without the checkpoint.

Suggestions:

- Score every miss on two axes: whether the format is correct and whether the answer is correct.
- Run the training-set accuracy check from section 2.2.
- Count how often the miss categories appear in the 12,226 rows.

If arithmetic and unit rows are rare in the corpus, 5b gains weight. If they are common and t3 still fails on them, it is capacity.

**4. What acceptance is realistic for a trained 1-layer MTP?**

See section 2.1. You need about 20-35%, not 85%. A trained head should plausibly exceed that (my estimate is 50%+ on trivial chat). DeepSeek-V3's jointly trained module reaches 85-90% ([DeepSeek-V3 report](https://arxiv.org/html/2412.19437v1)). Measure it on t3 now.

One framing note for the white paper: the 154 bar compares thin plus a drafter against a 2B without one. That is fine as registered, but state the asymmetry.

---

## 4. Smaller points

- **Unified-RAM OOM.** `disable_mmap=True` is the right fix. Check the result using the cgroup's `memory.peak` rather than RSS. Training t4 itself does not need the teacher loaded, so keep prep and training as separate processes.
- **Dependency pins.** Commit a lock file (`kernels==0.12.3`, hub 1.x, transformers 5.8.1, triton versions) beside `dequant_fp8_teacher.py`. That saga should not have to be rediscovered.
- **Scope boundary.** Engine logprobs for qwen3_5 CUDA and native bf16-MoE support are general TensorFold improvements, so they are noted, not MVP. Offline top-k KD is the right MVP path. Upstream issues are fine.
- **Push the work.** t1-t3, the KD scripts, `mtp_merge.py`, `mtp_train.py` and the results JSONs are not on any remote branch I can see. BL-1 and BL-3 ("reproduce or it didn't happen", plus provenance) apply here too.
- **MTP semantics.** The reverse-engineered head description (interleaved `[q_h|gate_h]`, partial rotary 64/256, θ=1e7, gate-rows-first gate_up) is valuable. Put it in a docs page with file and line references, and add a host test that checks your trainer's forward pass against one engine draft step on fixed inputs. That avoids discovering a layout mismatch only after training.

---

## 5. Suggested order for the next round

1. Write the BL-6 amendment and the BL-21 kill criterion. Pre-register the t4 prediction and the parity secondary metric.
2. Run the two diagnostics in section 2.2: training-set accuracy, and the early-exit baseline at layer 10.
3. Run the MTP smoke test plus a t3 acceptance measurement, solo, with a pre-registered threshold (section 2.1).
4. Run B5-long with adoption on the untrained nested checkpoint, with a byte-identity assertion between adopted and cold-prefill replies (sections 2.4 and 2.5).
5. Run t4 as registered (prep at k=32 or 64, train at k=8), in parallel with item 4 where memory allows.
6. Optionally, gate the zero-training routed-expert tail (layers 0-9 + 34-39) as a capacity data point (section 2.2).
7. Take one Nsight trace of thin decode, and check the bf16-prefix provenance (sections 2.7 and 2.6).
