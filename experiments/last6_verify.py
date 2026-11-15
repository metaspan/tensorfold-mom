#!/usr/bin/env python
"""5e pre-registration check: last6 thin tail must be BYTE-IDENTICAL to parent
layers 34-39 (renumbered 10-15), and the renumbered config must line up.
Prints 'tail bytes: mismatched N missing-parent M' — the phase2 gate token is
'mismatched 0 missing-parent 0'."""
import json, hashlib, os, re, sys
from safetensors import safe_open

P = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
O = "/home/derek/models/tensorfold-mom/qwen36-nested-16l-last6"
FROZEN, TAIL, TAIL_FROM = 10, 6, 34


def load(d):
    ip = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(ip):
        return json.load(open(ip))["weight_map"]
    with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as g:
        return {n: "model.safetensors" for n in g.keys()}


def main() -> int:
    po, oo = load(P), load(O)
    bad = miss = seen = 0
    for f in set(oo.values()):
        with safe_open(os.path.join(O, f), framework="pt") as g:
            for n in g.keys():
                if ".layers." not in n:
                    continue
                i = int(n.split("layers.")[1].split(".")[0])
                if not (FROZEN <= i < FROZEN + TAIL):
                    continue
                seen += 1
                psrc = n.replace(f".layers.{i}.", f".layers.{i + (TAIL_FROM - FROZEN)}.", 1)
                if psrc not in po:
                    miss += 1
                    continue
                with safe_open(os.path.join(P, po[psrc]), framework="pt") as gp:
                    a = gp.get_tensor(psrc).numpy().tobytes()
                b = g.get_tensor(n).numpy().tobytes()
                if hashlib.sha256(a).digest() != hashlib.sha256(b).digest():
                    bad += 1
    print(f"tail bytes: mismatched {bad} missing-parent {miss} (checked {seen})")
    cfg = json.load(open(O + "/config.json"))
    tc = cfg.get("text_config", cfg)
    assert tc.get("num_hidden_layers") == FROZEN + TAIL, "num_hidden_layers"
    lt = tc["layer_types"]
    plt = json.load(open(P + "/config.json"))
    plt = plt.get("text_config", plt)["layer_types"]
    assert lt[:FROZEN] == plt[:FROZEN], "prefix layer_types drift"
    assert lt[FROZEN:] == plt[TAIL_FROM:TAIL_FROM + TAIL], "tail layer_types drift"
    qc = tc.get("quantization_config") or cfg.get("quantization_config") or {}
    idxs = sorted({int(m.group(1)) for k in qc.get("quantized_layers", {})
                   for m in [re.match(r"^model\.language_model\.layers\.(\d+)\.", k)] if m})
    assert idxs == list(range(FROZEN + TAIL)), f"quant table indices {idxs}"
    print("config checks: OK")
    return 0 if bad == 0 and miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
