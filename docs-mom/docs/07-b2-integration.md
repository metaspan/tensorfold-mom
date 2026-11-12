# B2 — Routing integration spec (openai.zig ↔ mom_router)

B1 gives one process, N opened engines, one endpoint, and advertises every id —
but every request still executes on the default engine. B2 makes the chosen
engine actually run the request. This is the surgery map, ordered by risk.

## Touch-points (upstream anchors at pin 14dea6c6)

| Site | File:line | Change |
|---|---|---|
| Model resolution | `server/openai.zig:110` (`replyModel` silent fallback) | Replace silent fallback with registry lookup: exact id → that lane; unknown non-empty id → **400** (`{"error":{"message":"model 'x' is not served here"}}`), matching the Python front door and the Zig `RouteError.UnknownModel`. |
| Decide point | plan path in `openai.zig` (before tokenization submit) | Call `mom_router.decide(model_field, last_user_text, conv_key)` once per request; stash `{lane, reason}` on the request context. |
| Submit | `server/chat.zig:249` | Submit to `registry.lanes[lane].engine` instead of `srv.engine`. |
| Warm/continuation submit | `server/warm.zig:58` | Same per-lane submit; warm sessions are keyed per lane (see KV note). |
| Server fields | `server/server.zig:136` (`Server.engine`) | Add `mom: ?MomRegistry` beside `engine`; `engine` stays the default lane so every non-MoM path is untouched (N=1 parity rule). |

## Decision inputs

- `model_field`: raw request `model` string (may be null/empty).
- `last_user_text`: last user message content (string parts joined) — the same
  extraction as the Python spec's `_last_user_text`.
- **Tools/functions present → never trivial**: pass an empty-ish sentinel or a
  `has_tools` flag so `trivialReason` returns null (Zig core keeps the text-only
  table; this guard lives in the caller, per the module doc).
- `conv_key: u64`: `x-tensorfold-conversation` / `x-conversation-id` /
  `session_id` header if present, else FNV-1a of first-system + first-user
  content (stable across turns — validated by K3 reroute probe on Track A).

## Per-lane identity for KV/prefix caches

The prefix trie key has NO model geometry (`sessions/prefix.zig:21,37`), so the
lane index must be folded into the session/prefix key: key = `(lane, token-page
digests…)`. B2 minimal: each lane owns an independent Store instance created at
open time (already per-engine in `api.Opened`); the shared session table gains
the lane in its key struct. A same-conversation re-route therefore CANNOT hit
another model's KV — proven by reroute_probe semantics re-run against the native
server in the B2 gate.

## Observability (brief component 7, cheap version now)

- Response headers: `x-tf-routed-model`, `x-tf-route-reason`, `x-tf-route-us`.
- One JSONL line per decision appended to `--mom-log PATH` (rotated by size):
  `{ts, conv_key, requested, routed, reason, us}`.
- `/health` gains `"mom": {"lanes": [...], "sticky": n}`.

## Config wiring

`mom-serve` flags: `--route-rules on|off` (rules; off = explicit/default only),
`--sticky on|off` (default on), `--sticky-capacity N`, `--trivial-max-chars N`,
`--mom-log PATH`. All default to the Track A tuned values (240 chars etc.).

## Invariants / tests

1. **Non-MoM byte-identical**: without `mom:` on Server, every touched site
   compiles to the same code path (`if (srv.mom) ... else` original branch).
2. Router unit tests already green (37 in `test-mom-host`); add HTTP-level
   table test mirroring smoke_frontdoor cases against the in-process server.
3. Explicit unknown id → 400 (replaces silent fallback; upstream test that
   asserted fallback must be updated in-tree — flagged for the ATC claim).
4. Per-lane KV: same conv on lane A then B never replays A's KV (host test with
   two fake engines counting prefix hits).
5. Router latency: decide() under 50 µs p99 on trivial inputs (perf test in the
   mom_host step; budget in brief is 5 ms including HTTP framing).

## B2 exclusions (B3+ work)

Trivial-queue priority in the shared batch loop (B3), sleep/wake + pin
eviction (B4), classifier v2 slot, cross-model prefill translation (B5+).

## B2 as actually built (2026-11-10)

Shipped beyond/next to the map above:
- `mom_registry.zig` owns lanes + Router + JSONL log (`--mom-log PATH`, one file per
  run); `route()` takes an Io.Mutex (server is thread-per-connection).
- Lane roles: `--lane trivial|default`, fallback heaviest-weight = default,
  lightest = trivial (deterministic; logged at startup as `mom: routing over N lanes`).
- Tokenizer-equality guard at open: sha256 of tokenizer.json (or vocab+merges) must
  match across lanes; refusal names the two specs. Reply decoding uses the
  server-wide tokenizer, so this guard is load-bearing, not belt-and-braces.
- Deviations: response headers (`x-tf-routed-*`) deferred — the reply's `model`
  field already names the executed lane (that is what S3 asserts); conv-key comes
  from content only (client headers land with the HTTP-layer pass); tools-present
  bypass is caller-side (empty text → no trivial rule can fire).
- Host coverage: 40 tests in `test-mom-host` incl. registry lastUserText/convKey
  (caught a first-vs-last-user port bug — the spec takes the LAST user message).
