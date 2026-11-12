# Model selection (grounded in the native release's qualified-model matrix)

Source: `src/tensorfold/README.md` "Pick a model" table (upstream HEAD `14dea6c6`, cloned
2026-11-10), plus this machine's local caches. Restriction that drives everything:
the **native engine only serves qualified families/checkpoints** — the brief's
Qwen2.5-1.5B / Gemma-2-2B / Qwen3-8B suggestions are *not* in the native matrix and
would require family work. They remain valid targets for the Python-line path only.

## Single-GB10 qualified checkpoints we can use today (CUDA, linux-aarch64)

| Checkpoint | Weights (device) | Notes |
|---|---|---|
| `mlx-community/Qwen3.5-2B-MLX-4bit` @93760be4 | ~1.5 GB class | Quick-start model, **checked on GB10**; text only; family `qwen3_5` |
| `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` @81169a9b | 20.9 GB checkpoint | MoE, A3B active; recipe `qwen3.6-moe.md`, native CUDA receipt |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` @1355db6a | 20.3 GiB (20.9 GiB with `--drafter` = same ckpt, MTP) | NVIDIA QAD; **best heavy-lifter for one GB10** |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 30B-A3B NVFP4 | Checked on GB10; already in local HF cache; includes MTP |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` @667408d4 | 24B-A2B | GB10-checked (4-bit export) |
| `mlx-community/gemma-4-31B-it-4bit` + DFlash drafter | 31B | GB10 CUDA qualified (1.1.4), `--vision` optional |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` @b9ce18a6 | ~96 GiB (loads ~81 GiB w/ vision) | The production :8000 model; too heavy to co-load experimentally next to a second large model |

Local cache already holds: Qwen3.8-Flash-Next-FP4-Local, Nemotron-3.5-Lightning-30B-A3B-NVFP4
(+DSpark drafter), Qwen3-32B-NVFP4, Qwen3.6-27B-NVFP4, Step-3.7-Flash variants, Qwen3.6-35B-A3B-FP8.

## Recommended MoM pair (matches the brief's intent, no family work needed)

- **Trivial / always-hot**: `mlx-community/Qwen3.5-2B-MLX-4bit` — the release's own
  GB10 quick-start model; sub-100 ms-class TTFT expected; tiny KV; family `qwen3_5`
  already registered (`zig/src/families/qwen3_5/`).
- **Real work**: `nvidia/Qwen3.6-35B-A3B-NVFP4` with `--drafter` (MTP) — 20.3 GiB
  resident, strong A3B active path, GB10-qualified, leaves >100 GB unified memory
  headroom on the Spark for KV pools, the router, and the second model. It also matches
  Derek's `qwen-optim` evaluation model family (Qwen3.6-35B-A3B), so quality judgements
  transfer.
- **Optional third**: reuse the NVFP4 checkpoint's own MTP drafter as the "second
  network in-process" testbed, or add `TensorFold/GLM-5.3-Flash-DFlash2`-style GGUF
  drafter later. A DFlash2 GGUF exists locally (`/home/derek/models/gguf/GLM-5.3-Flash-DFlash2-GGUF`).

Alternate heavy lane if Flash-Next must be the "real" model: serve the MoM pair while
production :8000 is down for a window, or run the experiment fleet on the second Spark
(<redacted-lan-host> — SSH host-key re-provisioning needed first).

## Memory math (one GB10, 128 GB unified)

Approximate v1 budget for Qwen3.5-2B + Qwen3.6-35B-A3B-NVFP4:

| Component | Estimate |
|---|---|
| Qwen3.6-35B-A3B-NVFP4 weights + MTP drafter | ~21 GiB |
| Qwen3.5-2B weights | ~1.5 GiB |
| Engine runtime reserve (existing policy) | 9 GiB |
| KV pool A (heavy, e.g. 8 streams × 32k window, fp8) | ~12–16 GiB (per-stream sizing read from the engine's CUDA startup capacity line) |
| KV pool B (small, 8 streams × 8k) | ~1–2 GiB |
| Router + orchestrator + slack | <2 GiB |
| **Total** | **~45–50 GiB** — comfortable |

Reference datapoint from the recipe README: full 262k windows for the 125B model cost
4.78 GiB per stream at fp8 KV; MoM pools are sized per model at startup and admission
refuses over-budget configurations rather than degrading — preserve that behavior.

## What we deliberately do NOT do in v1

- No Qwen2.5-1.5B / Gemma-2-2B ports: adding a family is a project in itself
  (`docs/recipes/adding-a-cuda-family.md`) and buys nothing the 2B doesn't already give.
- No co-residency with the 96 GiB Flash-Next FP4 checkpoint on the same GPU during
  experiments (starves KV and violates the reserve policy).
- No third model lane until the two-model path is qualified.
