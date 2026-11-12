#!/usr/bin/env python3
"""Load the nested-16l checkpoint into the HF Qwen3_5MoeForCausalLM.

The checkpoint (M0) is the engine's view of the student: parent bytes under
`model.language_model.*`, MoE experts per-expert and NVFP4-packed (ModelOpt:
uint8 two-e2m1-per-byte weights, fp8-e4m3 block scales of block 16, an fp32
global scale_2; non-expert weights e.g. linear_attn.in_proj_z are F8_E4M3),
plus `mtp.*` tensors the HF model does not define. The HF model wants bare
names and one FUSED tensor per expert projection (measured on meta, 2026-11-11):
  experts.gate_up_proj [256, 1024, 2048]  (rows 0:512 gate, 512:1024 up; ffn=512)
  experts.down_proj    [256, 2048, 512]
This loader remaps, dequantizes to bf16, stacks, and asserts every parameter
is covered exactly once. Everything trains in v1 distillation; the frozen
prefix is re-fused from the parent's bytes at deploy (nested_extract route).

Gate: run as a script; exits nonzero on any name, shape, or coverage mismatch.
"""
import re
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

STUDENT = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
SKIP_SUFFIX = ("_scale", "_scale_2", "_input_scale")


def dequant_nvfp4(w_u8: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor,
                  block: int = 16) -> torch.Tensor:
    """ModelOpt NVFP4: low nibble first; scale [out, in/block] e4m3; scale_2 global fp32."""
    out_p, in_half = w_u8.shape
    lo, hi = w_u8 & 0x0F, w_u8 >> 4
    vals = torch.empty(out_p, in_half * 2, dtype=torch.float32)
    vals[:, 0::2] = E2M1[(lo & 0x07).long()] * torch.where(lo & 0x08 > 0, -1.0, 1.0)
    vals[:, 1::2] = E2M1[(hi & 0x07).long()] * torch.where(hi & 0x08 > 0, -1.0, 1.0)
    s = scale.to(torch.float32).repeat_interleave(block, dim=1)
    return vals * s * scale_2.to(torch.float32)


def load_student(device="cpu", dtype=torch.bfloat16):
    cfg = AutoConfig.from_pretrained(STUDENT)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model.to_empty(device=device)
    params = dict(model.named_parameters())
    filled = set()
    path = f"{STUDENT}/model.safetensors"

    with safe_open(path, "pt") as f:
        keys = set(f.keys())

        def unpack(name):
            """One linear's weights as `dtype`, from whichever form the header carries."""
            t = f.get_tensor(name)
            if t.dtype == torch.uint8 and f"{name}_scale" in keys:
                return dequant_nvfp4(t, f.get_tensor(f"{name}_scale"),
                                     f.get_tensor(f"{name}_scale_2")).to(dtype)
            if t.dtype == torch.float8_e4m3fn:
                return t.to(torch.float32).to(dtype)
            return t.to(dtype)

        def put(name, tensor):
            p = params.get(name)
            assert p is not None, f"checkpoint tensor has no model param: {name}"
            assert tuple(p.shape) == tuple(tensor.shape), \
                f"shape {name}: got {tuple(tensor.shape)} want {tuple(p.shape)}"
            with torch.no_grad():
                p.copy_(tensor.to(p.dtype))
            filled.add(name)

        # fused experts: stack per-expert dequantized weights into the HF buffers
        packed = {k for k in keys if EXPERT_RE.match(k)}
        layers = {}
        for k in packed:
            m = EXPERT_RE.match(k)
            layers.setdefault(int(m.group(1)), {})[int(m.group(2))] = m.group(3)
        for L, per_e in layers.items():
            n_e = max(per_e) + 1
            assert len(per_e) == n_e, f"layer {L}: expert ids not dense 0..{n_e - 1}"
            for kind, tgt in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
                name = f"model.layers.{L}.mlp.experts.{tgt}"
                rows = params[name].shape[1]
                buf = torch.empty(params[name].shape, dtype=torch.float32)
                for e in range(n_e):
                    base = f"model.language_model.layers.{L}.mlp.experts.{e}."
                    if kind == "down":
                        buf[e] = unpack(base + "down_proj.weight").to(torch.float32)
                    else:
                        half = rows // 2
                        buf[e, 0:half, :] = unpack(base + "gate_proj.weight").to(torch.float32)
                        buf[e, half:, :] = unpack(base + "up_proj.weight").to(torch.float32)
                with torch.no_grad():
                    params[name].copy_(buf.to(params[name].dtype))
                filled.add(name)

        # everything else: prefix remap, header-driven unpack
        for k in sorted(keys):
            if k.startswith("mtp.") or EXPERT_RE.match(k) or k.endswith(SKIP_SUFFIX):
                continue  # MTP re-merges at deploy; experts stacked above; scales consumed
            if k == "lm_head.weight" or k.startswith("lm_head."):
                name = k
            elif k.startswith("model.language_model."):
                name = "model." + k.removeprefix("model.language_model.")
            else:
                name = k
            put(name, unpack(name if name in keys else k))

    missing = set(params) - filled
    assert not missing, f"model params never filled ({len(missing)}): {sorted(missing)[:6]}"
    return model


if __name__ == "__main__":
    mdl = load_student()
    n = sum(p.numel() for p in mdl.parameters())
    print(f"LOADED {type(mdl).__name__} params={len(dict(mdl.named_parameters()))} numel={n/1e9:.2f}B")
