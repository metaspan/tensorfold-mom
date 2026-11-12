# Item 6 — duplication ceiling for cross-model cache translation

**Question.** The brief defers "PrefillShare-style or linear-map translated caches
across compatible models". What would such a mechanism actually save, given the
MoM's per-lane caches already work (docs/09)?

**Method.** Pure analysis of the B5 dataset (`experiments/results/cache-alternating.json`,
28 alternating turns x 2 passes, TF_QWEN_REQUEST_STARTS=1). "Computed" = prompt
tokens a lane actually prefilled (prompt - resumed). The "translated-cache ideal"
assumes each turn could resume from the deepest prefix ANY lane has computed — in
an alternating conversation, the other lane prefilled the whole history two turns
ago, minus the newest reply's tokens (~8 role/overhead proxy).

**Results.**

| pass | nominal prompt tokens | computed (lane-local caches) | translated-cache ideal | headroom |
|---|---|---|---|---|
| cold | 39,254 | 5,673 (85.4% saved) | 2,992 | **47.3% of cold prefill** |
| warm | 39,254 | 196 (99.5% saved) | 196 | **0%** |

**Interpretation.**
- **Warm conversations leave item 6 nothing.** Per-lane caches already resume
  ~everything; a cross-model translation layer adds risk (KV correctness across
  weight sets) for zero measured gain on the hot path.
- The entire prize is **cold first-pass traffic**: ~2,681 duplicated prefill
  tokens per 28-turn conversation, i.e. ~90 ms heavy-lane TTFT per mid-conversation
  turn at the measured ~6.5k tok/s prefill rate (176 -> ~100 ms), ~10 ms on the
  trivial lane. Real, but an order of magnitude below the wins already banked
  from simply enabling the per-family cache switch (425 -> 46 ms cold).
- Where cold-first-pass matters in production: bursty traffic where conversations
  are seen once (support triage), or after cache eviction/restart. Sticky routing
  (default on) reduces the duplication further by keeping conversations on one lane.

**Recommendation (SUPERSEDED 2026-11-11).** This section's "deprioritize" call
was a v1-economics judgment. Derek: shared KV "is the whole point of the
experiment" — item 6 is re-scoped as the program centerpiece; see
docs/12-shared-kv-plan.md (S1 fidelity gate). The ceiling numbers above
stand; note they understate the case for the *cold-first-pass* traffic class,
which is where the white paper's shared-KV story lives.
