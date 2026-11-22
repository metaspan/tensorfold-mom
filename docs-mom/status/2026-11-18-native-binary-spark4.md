# Getting tensorfold-native on spark4 (for qwen3_5_moe_text / BL-21 probes)
Date: 2026-11-18 — from hermes on spark-2d3b. Read alongside 2026-11-16-spark4-handoff.md.

## Reality check on the "unachievable" verdict
spark4 ALREADY runs `tensorfold-native serve` for the production Qwen3.8 model right now
(pid visible in `pgrep -af tensorfold-native`). The runtime environment is proven. What the
stock release binary lacks is our mom-fork support for the `qwen3_5_moe_text` family +
`NESTED.json` checkpoints. That is one binary + a fatbin directory away. No CUDA "GPU build"
is needed — nvcc compiles for the target arch, the build is CPU work.

## Option A — copy our prebuilt binary (recommended, ~1 min of transfer)
Built on identical hardware (DGX Spark GB10, aarch64, same 580 driver), only libc dependency.

    cd ~/tensorfold-mom-data   # or anywhere
    rsync -avP derek@192.168.10.166:~/sp4-transfer/tensorfold-native-mom.tar.gz .
    tar -xzf tensorfold-native-mom.tar.gz   # yields native-bin/ (tensorfold-native) + fatbin/
    ./native-bin/tensorfold-native --version

The tarball layout mirrors `zig-out/` — if you unpack into your repo copy as
`zig-out/native/bin/tensorfold-native` + `zig-out/fatbin/`, experiments/distill_gate.py
--engine native finds it the same way it does on spark-2d3b.
Verify sha256 against the value listed in ~/sp4-transfer/sha256sums.txt after unpack.

## Option B — build it yourself from mom/main
Repo: metaspan/tensorfold-mom, branch mom/main (you already have mom-repo clone).

    # zig 0.17.0 (exact version required — 0.18/0.19 will break on removed preview APIs)
    curl -LO https://ziglang.org/download/0.17.0/zig-linux-aarch64-0.17.0.tar.xz
    tar -xf zig-linux-aarch64-0.17.0.tar.xz && export PATH=$PWD/zig-linux-aarch64-0.17.0:$PATH
    zig version   # must print 0.17.0

    # nvcc required (PATH or /usr/local/cuda/bin). DGX Spark images ship it; else apt nvidia-cuda-toolkit.
    nvcc --version   # any CUDA 12.x/13.x is fine

    cd <repo>   # repo root, where build.zig lives
    zig build native -Doptimize=ReleaseFast
    # ~15-45 min on GB10 CPUs. NO GPU NEEDED. Products:
    #   zig-out/native/bin/tensorfold-native
    #   zig-out/fatbin/*.fatbin   (kernels compiled by nvcc — ship these alongside the binary)

Build gotchas (learned the hard way on spark-2d3b):
- Plain `zig build` does NOT build tensorfold-native. The step is `zig build native`.
- If a previous partial build is broken, wipe `.zig-cache/` and the stale `zig-out/fatbin`
  before retrying (stale fatbins are the classic "runs old kernels" trap).
- Host-step compilation does not check unreferenced pub fns of non-root files; the exe
  build is the real gate.

## Hard constraints you must not fight
- The native engine REFUSES plain bf16 MoE (cuda_geometry.zig:290) BY DESIGN. Serve NVFP4
  exports of students (see experiments export path in the 2026-11-16 handoff doc).
- Python transformers is NOT a substitute for BL-21(b) tok/s measurement — the decode bar
  (>=154 tok/s) is only meaningful on the native engine. Use HF engine for correctness
  gates only.
- One model per native process pays ~12.2 GiB MemAvailable reserve (check via /v1/models
  mom.memory + MemAvailable; nvidia-smi memory.used is N/A on DGX Spark).
- Do not disturb the production tensorfold-native serving Qwen3.8-Flash-Next.

## Smoke when you have the binary
    ./native-bin/tensorfold-native serve <nvfp4-student-dir> --name t3-probe --port 8123 --context 8192
    curl -s localhost:8123/v1/models | head -c 300
Then run experiments/distill_gate.py --engine native per the handoff doc.
If boot prints MissingTensor for layers.16 or similar: that is the VL-form config trap —
the engine reads text_config fields; see docs-mom diary M1 entry.
