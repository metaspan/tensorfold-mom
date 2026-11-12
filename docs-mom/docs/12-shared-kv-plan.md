# Item 6 — shared KV / translated cache: status and S1 plan

**Status: not implemented.** Everything proven so far (docs/09, B5) is
lane-local caching. What IS shared today: one process, one endpoint, one
tokenizer (guard-verified), one global memory ledger, per-lane sleep/wake.
What is NOT shared: any prefill work or KV state. The measured prize
(docs/10): 47.3% of cold first-pass prefill; 0% warm.

This round's config audit (both checkpoints, config.json) sharpens the odds:

## Geometry — favorable for attention, unfavorable for the recurrent half

Both are qwen3_5 hybrids with `full_attention_interval = 4` and identical
attention geometry: kv_heads=2, head_dim=256, hidden_size=2048.

| state | Qwen3.5-2B | Qwen3.6-35B-A3B | shared-KV verdict |
|---|---|---|---|
| Full-attention layers | 6 (idx 3,7,…,23) | 10 (idx 3,7,…,39) | **same per-layer KV width (2×256)** — a 512→512 linear map per (2B→35B) layer pair is dimensionally trivial |
| GDN linear layers | 18: 16k×128, **16v**×128 | 30: 16k×128, **32v**×128 | value-head count differs (16 vs 32) — **state shapes are incompatible**; per-layer linear maps don't absorb it |
| GDN conv tails | k=4, 16k/16v | k=4, 16k/32v | same mismatch |
| Weights | 4-bit MLX | NVFP4 | both quantized — translated KV would feed the 35B's own kernels, so its quant format is native; the 2B side only needs to *emit* K/V |

Two consequences:
1. A translated cache can only ever cover the attention quarters (6→10 maps);
   the GDN state must still come from somewhere — either recomputed (GDN
   prefill is linear-time, plausibly cheap) or approximated by a bigger map
   across the head-count break (research-grade).
2. **Exact decoding forbids trusting any translation.** A translated KV is a
   prefill accelerator behind an exact-verify gate, never a truth source —
   same principle as the drafter path.

## S1 — fidelity gate (offline, serving untouched)

Decide whether translation is even in the right ballpark before touching
the serving path:

1. **Dump hook** (additive): for a token prefix, persist per-attention-layer
   K and V rows from each engine (2B and 35B) plus token ids — a debug-only
   writer in the prefill path, host-checkable, off unless TF_MOM_KVDUMP=dir.
2. **Corpus**: the B5 28-turn conversation (already exercises both lanes).
3. **Fit**: ridge-regression maps 2B attention layer l -> 35B attention layer
   f(l) (layer-index stretch 6->10, fitted independently, no pairing
   assumption), per K and V.
4. **Gate metric**: held-out prefixes — greedy top-1 agreement between 35B
   resumed-from-translated-KV (GDN recomputed) and 35B native prefill, as
   per-position agreement + first-divergence depth.
5. **Go/no-go** (pre-registered): go if held-out first-divergence depth
   >= one full turn median AND per-position top-1 >= 0.99; otherwise the
   translation dies here and item 6 closes as "measured, not viable" — also
   a publishable result.

S2 (verified adoption: translated KV as prefill warm-start with exact-verify
on a residual window, behind the docs/02 compat-digest extension point) and
S3 (device A/B on the B5 cold pass, ON/OFF) proceed only on a S1 go.

## S1 run 1 (2026-11-11) — PRELIMINARY NO-GO, too data-thin to conclude

Collection succeeded after one fix (`059a51e`: `buildBin` freed an `alignedAlloc`'d
buffer under the device's SafeAllocator — host tests on the testing allocator
can't see allocator-metadata misuse; a regression test now runs the path under a
SafeAllocator). 29 checkpoints per lane dumped, all attention layers, fp8 decoded
to f32.

The first fit scored **best-map mean top-1 0.6435, median first-divergence
11/23 tokens → offline NO-GO — but on only 2 usable checkpoints (23 positions)**.
Cause: the fitter aligned checkpoints by exact token-id equality, and the two
engines ship different chat templates (the 35B template carries thinking blocks),
so 27/29 checkpoint pairs mismatched. The verdict is NOT decision-grade.

Run 2 re-aligns checkpoint pairs by decoded piece text (same conversation, two
BPEs; only exactly-matching cumulative text is aligned, never force-fit) and
feeds the fit all usable positions from all 29 pairs. The gate is unchanged.

## S1 run 2 (2026-11-11) — offline GO, but NOT decision-grade (row leakage)

Text alignment (new `s1_align.py`: byte-BPE decode, turn-chunk LCS, two-pointer
cumulative-text alignment) recovered 8060 aligned positions across all 29 pairs.
The fit scored best-per-dst mean top-1 **1.0000** (cosine 0.9976, negative
controls at chance) — offline proxy GO.

Why that GO does not decide anything: checkpoints are **nested prefixes** and KV
rows are causal, so all 519 unique held-out rows occur **verbatim in the train
set** (verified in the report JSON: `heldout_positions_also_in_train: 519/519`).
The maps were scored on rows they had memorised. This certifies the ridge map
fits its training rows and that rows are mutually discriminable — not
generalization. (Meaningful positive signal nonetheless: a 256→256 linear map
reproduces a different model's K/V rows to cosine 0.998 on fit data.)

## S1 run 3 (in flight) — the generalization gate

New corpus, `s1_collect2.py`: a **disjoint 20-turn conversation** (different
system prompt, topics, phrasings) dumped on both lanes to
`results/kvdump-b/`. `s1_fit.py --eval-corpus` mode fits on all B5 checkpoints
and scores **only** on the fresh conversation's rows — rows no map has ever
seen — with a verbatim-leakage stat printed per run. The pre-registered gate
rule is unchanged (top-1 ≥ 0.99, median first-divergence ≥ one median turn).
This is the decision-grade offline number; the registered device gate
(decode-resume) follows only if this passes.

## S1 run 3 RESULT (2026-11-11) — DECISION-GRADE NO-GO: translation is dead

Fit on all 29 B5 checkpoints (6009 aligned rows), scored ONLY on the disjoint
corpus-B conversation (21 checkpoints, 5289 aligned positions, novel rows):

| metric (best map per dst layer) | run 2 (memorised) | run 3 (novel rows) | gate |
|---|---|---|---|
| mean per-position top-1 | 1.0000 | **0.3418** (max 0.4653) | >= 0.99 |
| mean row cosine | 0.9976 | **0.4071** | — |
| median first-divergence | 2051 tok | **6 tok** (turn = 437) | >= one turn |
| min retrieval margin | +0.0114 | **-0.4605** | — |
| negative controls | chance | chance (metrics discriminative) | — |

Interpretation: the 256x256 ridge maps (65k params/pair vs 6k train rows)
interpolate the training conversation and collapse to near-chance on unseen
text. Predicted rows retrieve WRONG rows better than their own (negative
margins). A real-but-weak shared subspace remains (cosine 0.41, top-1 0.34 vs
~0.002 chance) — exactly what two independently trained checkpoints should
share: some universal features of text, not a usable code.

**Verdict: per pre-registration, the linear-translation branch of item 6
closes "measured, not viable."** The registered device gate (decode-resume)
does not run — its offline pre-filter failed by ~0.65 at the mean, not at the
margin. Caveats kept honest:
- Stronger mappers (MLP, per-head Procrustes, permutation alignment) are a
  training project, not a fit; the bar (exact-decode-safe) is 0.99 and the
  gap is 0.34 — extrapolating ridge->MLP over that gap is not credible.
- S2 (translation as verified warm-start with residual recompute) would still
  reject ~2/3 of rows at these fidelity levels; with GDN recompute already
  linear-time, the residual economics do not rescue it.

**What this decides:** the cross-model KV story for this program is the
NESTED UPCYCLE — a thin student sharing the heavy model's frozen layers by
weight identity, where fidelity is 1.0 BY CONSTRUCTION and the shared cache
is one physical copy read via page tables (docs: nested spec). The 47.3%
cold-prefill headroom (docs/10) stays the prize; translation was the cheap
road to it and has now been measured dead.
