#!/usr/bin/env python3
"""Materialize the nested-16l checkpoint as a plain bf16 sharded HF checkpoint.

The M0 checkpoint is the engine's view: parent bytes under `model.language_model.*`,
experts per-expert NVFP4-packed (ModelOpt: uint8 two-e2m1 bytes, fp8-e4m3 block
scales of 16, fp32 global scale_2; some non-experts are F8_E4M3), plus `mtp.*`
the HF model has no slots for. The HF Qwen3_5MoeForCausalLM wants bare names
and FUSED expert tensors (measured on meta 2026-11-11):
  experts.gate_up_proj [256, 1024, 2048]  (rows 0:512 gate, 512:1024 up; ffn=512)
  experts.down_proj    [256, 2048, 512]

Plain from_pretrained on the raw checkpoint would train garbage. This script
streams tensors through dequant and writes ordinary bf16 SHARDS + the index,
flushing every few GiB so memory stays bounded (an all-in-RAM version OOMed
its worker cgroup at 4 GiB anon-rss). distill_train loads the OUTPUT directory
with plain from_pretrained.

Gate: run as a script; exits nonzero on any name, shape, or coverage mismatch.
"""
import json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

STUDENT = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"
OUT = "/home/derek/models/tensorfold-mom/qwen36-nested-16l-bf16"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
SKIP_SUFFIX = ("_scale", "_scale_2", "_input_scale")
SHARD_BYTES = 4 << 30


def dequant_nvfp4(w_u8, scale, scale_2, block=16):
    out_p, in_half = w_u8.shape
    lo, hi = w_u8 & 0x0F, w_u8 >> 4
    vals = torch.empty(out_p, in_half * 2, dtype=torch.float32)
    vals[:, 0::2] = E2M1[(lo & 0x07).long()] * torch.where(lo & 0x08 > 0, -1.0, 1.0)
    vals[:, 1::2] = E2M1[(hi & 0x07).long()] * torch.where(hi & 0x08 > 0, -1.0, 1.0)
    s = scale.to(torch.float32).repeat_interleave(block, dim=1)
    return (vals * s * scale_2.to(torch.float32)).to(torch.bfloat16)


class ShardWriter:
    def __init__(self, out_dir):
        self.out, self.buf, self.live, self.n, self.weight_map = out_dir, {}, 0, 0, {}

    def add(self, name, tensor):
        self.buf[name] = tensor
        self.live += tensor.numel() * tensor.element_size()
        if self.live >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.n += 1
        self.pending = self.n
        fname = f"model-{self.n:05d}.safetensors"  # renamed with the total at finish()
        save_file(self.buf, f"{self.out}/{fname}", metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fname
        print(f"shard {fname}: {len(self.buf)} tensors", flush=True)
        self.buf, self.live = {}, 0

    def finish(self):
        self.flush()
        total = self.n
        for old in sorted(os.listdir(self.out)):
            if re.match(r"model-\d{5}\.safetensors$", old):
                idx = int(old[6:11])
                new = f"model-{idx:05d}-of-{total:05d}.safetensors"
                os.rename(f"{self.out}/{old}", f"{self.out}/{new}")
        wmap = {k: re.sub(r"\.safetensors$", f"-of-{total:05d}.safetensors", v)
                for k, v in self.weight_map.items()}
        json.dump({"metadata": {"total_size": 0}, "weight_map": wmap},
                  open(f"{self.out}/model.safetensors.index.json", "w"), indent=1)


def main():
    cfg = AutoConfig.from_pretrained(STUDENT)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    want = {n: tuple(p.shape) for n, p in model.named_parameters()}
    del model

    os.makedirs(OUT, exist_ok=True)
    for stale in os.listdir(OUT):
        os.remove(f"{OUT}/{stale}")
    filled = set()
    w = ShardWriter(OUT)
    path = f"{STUDENT}/model.safetensors"

    with safe_open(path, "pt") as f:
        keys = set(f.keys())

        def unpack(name):
            t = f.get_tensor(name)
            if t.dtype == torch.uint8 and f"{name}_scale" in keys:
                return dequant_nvfp4(t, f.get_tensor(f"{name}_scale"),
                                     f.get_tensor(f"{name}_scale_2"))
            if t.dtype == torch.float8_e4m3fn:
                return t.to(torch.float32).to(torch.bfloat16)
            return t.to(torch.bfloat16)

        def put(name, tensor):
            assert name in want, f"checkpoint tensor has no model param: {name}"
            assert tuple(tensor.shape) == want[name], \
                f"shape {name}: got {tuple(tensor.shape)} want {want[name]}"
            w.add(name, tensor)
            filled.add(name)

        for k in sorted(keys):
            if k.startswith("mtp.") or EXPERT_RE.match(k) or k.endswith(SKIP_SUFFIX):
                continue
            if k == "lm_head.weight" or k.startswith("lm_head."):
                name = k
            elif k.startswith("model.language_model."):
                name = "model." + k.removeprefix("model.language_model.")
            else:
                name = k
            put(name, unpack(name if name in keys else k))

        layers = {}
        for k in keys:
            m = EXPERT_RE.match(k)
            if m:
                layers.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        for L in sorted(layers):
            n_e = max(layers[L]) + 1
            assert len(layers[L]) == n_e, f"layer {L}: expert ids not dense"
            gu_name = f"model.layers.{L}.mlp.experts.gate_up_proj"
            dn_name = f"model.layers.{L}.mlp.experts.down_proj"
            gu = torch.empty(want[gu_name], dtype=torch.bfloat16)
            dn = torch.empty(want[dn_name], dtype=torch.bfloat16)
            half = want[gu_name][1] // 2
            for e in range(n_e):
                base = f"model.language_model.layers.{L}.mlp.experts.{e}."
                gu[e, :half, :] = unpack(base + "gate_proj.weight")
                gu[e, half:, :] = unpack(base + "up_proj.weight")
                dn[e] = unpack(base + "down_proj.weight")
            w.add(gu_name, gu)
            w.add(dn_name, dn)
            filled.update({gu_name, dn_name})
            print(f"layer {L}: {n_e} experts fused", flush=True)

    missing = set(want) - filled
    assert not missing, f"model params never filled ({len(missing)}): {sorted(missing)[:6]}"
    w.finish()
    cfg = json.load(open(f"{STUDENT}/config.json"))
    cfg.pop("quantization_config", None)  # weights are bf16 now; a stale modelopt block makes HF warn/skip
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)
    for extra in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        if os.path.exists(f"{STUDENT}/{extra}"):
            shutil.copy(f"{STUDENT}/{extra}", f"{OUT}/{extra}")
    print(f"MATERIALIZED {OUT}: {len(w.weight_map)} tensors bf16 sharded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
