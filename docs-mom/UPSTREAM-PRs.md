# Upstream PR plan — keeping the path open

Status: NOT STARTED (by design — wait until the M4 device gate is green; nothing
goes upstream that hasn't survived a byte-identity gate on device).

Upstream is contributor-active: every PR requires the ATC ritual first
(**installed on this box**: atc-github 0.9.6 — npm 12 blocks git-protocol
fetches, so it was installed from a clone: `git clone https://github.com/ashhart/ATC
~/tools/ATC && npm install -g ~/tools/ATC`). Command surface: bare `atc` = the
traffic board (work, collisions, duplicates), `atc begin <issue> <files...>` =
claim + declare planned files, `atc guard` before publishing, `atc overlap` to
test-merge against open PRs. No issue = no claim = no PR; open/discuss the issue
first. See the repo's `docs/atc/SKILL.md`.

## Slice strategy

Cherry-picks rot within days at upstream's cadence (~250 commits/weekend seen).
Each slice is RE-APPLIED onto current `origin/main` at PR time inside the
worktree `src/upstream-pr/` (created from `origin/main`; refresh before use).
Slices are ordered so each PR stands alone and lands without its successors.

| # | PR | Files | Standalone demo | Gate before pushing |
|---|---|---|---|---|
| 1 | **kv_pool page pins** — expose a public `retain`-based pin API and arena-held pin/release helpers | `zig/src/core/kv_pool.zig`, `cuda_pages.zig` | pin survives eviction pressure; release restores evictability | `zig build test-kv-pool` + new host test |
| 2 | **Read-only page-table views over a KV arena** (`SharedView`) | `qwen3_5/cuda_cache.zig` (SharedView/SharedHold/seg), `kernels/cuda/qwen35.cuh` (kv_row gate), `cuda_pages.zig` | second `Cache` resolving a prefix of a first cache's arena inside ONE engine; `sharedViewOk` refusals (fp8, mismatched ptok) stay copy | new host test in `cuda/root.zig` reach + device smoke: view decode byte-identical to copy |
| 3 | **Cross-engine adoption** (the MoM copy path + audit/verify) | `qwen3_5/cuda_reuse.zig` pieces | needs a second engine — motivate via sessions/idle_unload warm-start | full M3 probe suite green on current base |
| 4 | (maybe) **mom-serve multi-model front door** | `server/` | — | only if upstream shows appetite; otherwise stays fork-only |

`sharedAudit` (TF_MOM_SHARED_VERIFY) ships inside PR 2 as the test's device arm —
upstream's own review style (contributor skill, per-file host tests, pair-compared
baselines) suggests audits-in-PR land better than audits-in-comments.

## Rules that keep this practical (starting now)

1. **Commit separation**: engine-generic changes (pool/cache/kernel) never ride
   in a commit with MoM server/routing changes. If a probe file and an engine
   file change together for one gate, the engine commit message cites the probe,
   not vice versa.
2. **Every engine-generic change gets a host test at the time it's made** (the
   upstream drift gate), hooked via `cuda/root.zig` (cuda module) or the
   `test {}` block in `server/root.zig` — a green `test-mom-host` does not prove
   the native build; gate device runs on `zig build native` (see diary).
3. **Env-gated, fallback-by-default**: everything M4-shaped is behind
   `TF_MOM_*` env with silent fallback to the existing path — PR reviewers must
   see zero behaviour change when the flag is absent. Keep it that way.
4. **Upstream attribution both ways**: our diary credits kessel/mergekit ideas;
   PR descriptions cite the diary sections as the provenance record (docs-mom).
5. **Never break the mirror**: `main` stays a pure fast-forward mirror of
   upstream (cron-enforced). All our work stays on `mom/main` + `pr/*` branches
   cut from `origin/main`, never from `mom/main` as-is.

## Worktree

```
src/upstream-pr/    # git worktree, branch pr/* per table above
```
Refresh ritual: `git -C src/upstream-pr fetch origin && rebase/reset onto
origin/main`, re-apply the slice by hand, run its gate column, ATC claim, push
branch to metaspan, open PR from `dcolley/metaspan:pr/<name>` against
`ashhart/TensorFold:main`.
