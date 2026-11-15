#!/usr/bin/env python3
"""M0: extract the nested checkpoint (frozen prefix + warm-start tail) from
the heavy 35B, per docs/13-nested-upcycle-spec.md.

Pure stdlib: reads the parent's safetensors headers, byte-copies the frozen
prefix (embed_tokens, layers 0..F-1, final norm, lm_head — NEVER requantized)
and the tail initialisation (the parent's next T layers, warm start so the
student begins as a coherent (F+T)-layer truncation), emits config.json with
the heavy's tokenizer/template (shared cache needs shared token streams), and
writes NESTED.json with the frozen_prefix_digest that is the adoption
permission (docs/13 section 4).

  python3 nested_extract.py --out /home/derek/models/tensorfold-mom/qwen36-nested-16l
  python3 nested_extract.py --out <same> --verify   # re-check digests + bytes
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

PARENT = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
LP = "model.language_model.layers."
DROP_PREFIXES = ("model.visual.",)  # mtp.* STAYS: the engine loads the MTP
# drafter as virtual layer <num_hidden_layers> from mtp.layers.0.* (M1: MissingTensor
# layers.16.linear_attn.in_proj_qkv.weight when dropped)
TOK_FILES = ("tokenizer.json", "vocab.json", "tokenizer_config.json",
             "chat_template.jinja", "generation_config.json", "preprocessor_config.json")


def parent_tensors(parent):
    """name -> (dtype, shape, (shard_file, off, off2)) across all parent shards."""
    out = {}
    for fn in sorted(os.listdir(parent)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(parent, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        data_start = 8 + n  # data_offsets are relative to the byte after the header
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            out[k] = (v["dtype"], v["shape"], (fn, a + data_start, b + data_start))
    return out


def keep(name, frozen, tail, tail_from=None):
    if name.startswith(DROP_PREFIXES):
        return False
    if name.startswith(LP):
        i = int(name[len(LP):].split(".", 1)[0])
        if tail_from is None or tail_from == frozen:
            return i < frozen + tail  # frozen prefix + warm-start tail from the parent
        return i < frozen or tail_from <= i < tail_from + tail
    return True  # embed_tokens, final norm, lm_head, rope tables


def rename_layer(name, frozen, tail_from):
    # parent layer index -> thin layer index (5e last-6 byte-copy: 34-39 -> 10-15)
    if not name.startswith(LP):
        return name
    i = int(name[len(LP):].split(".", 1)[0])
    if i < frozen:
        return name
    return f"{LP}{i - tail_from + frozen}." + ".".join(name.split(".")[4:])


def sha_of(f, off, length, chunk=1 << 20):
    f.seek(off)
    h = hashlib.sha256()
    left = length
    while left:
        b = f.read(min(chunk, left))
        if not b:
            raise SystemExit("short read")
        h.update(b)
        left -= len(b)
    return h.hexdigest()


def frozen_set(ts, frozen):
    # frozen = shared-prefix surface: embed, layers 0..frozen-1, final norm,
    # lm_head. mtp.* is the drafter (irrelevant to shared rows) -> excluded.
    return [n for n in sorted(ts) if keep(n, frozen, 0) and not n.startswith("mtp.") and not
            (n.startswith(LP) and int(n[len(LP):].split(".", 1)[0]) >= frozen)]


def extract(parent, out, frozen, tail, tail_from=None):
    ts = parent_tensors(parent)
    # 5e last-6 byte-copy: tail layers may come from a different parent range
    # (tail_from..tail_from+tail-1) and are renumbered to frozen..frozen+tail-1.
    names = [(rename_layer(n, frozen, tail_from) if tail_from is not None else n, n)
             for n in sorted(ts) if keep(n, frozen, tail, tail_from)]
    os.makedirs(out, exist_ok=True)
    # plan offsets for one output shard
    offs, cur = {}, 0
    for o, p in names:
        ln = ts[p][2][2] - ts[p][2][1]
        offs[o] = (cur, cur + ln)
        cur += ln
    hdr = {o: {"dtype": ts[p][0], "shape": ts[p][1],
               "data_offsets": list(offs[o])} for o, p in names}
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-len(hb) - 8) % 8
    hb += b" " * pad
    with open(os.path.join(out, "model.safetensors"), "wb") as w:
        w.write(struct.pack("<Q", len(hb)))
        w.write(hb)
        srcs = {}
        for o, p in names:
            fn, a, b = ts[p][2]
            f = srcs.setdefault(fn, open(os.path.join(parent, fn), "rb"))
            f.seek(a)
            left = b - a
            while left:
                blk = f.read(min(1 << 22, left))
                w.write(blk)
                left -= len(blk)
        for f in srcs.values():
            f.close()

    cfg = json.load(open(os.path.join(parent, "config.json")))
    # The engine reads the TOWER (text_config) — VL-form config. A top-level
    # num_hidden_layers is DEAD (M1: loader kept the parent's 40 and refused at
    # the first absent layer). Edit the tower, prune its per-layer quant table,
    # keep mtp_num_hidden_layers: the MTP layer loads from mtp.* (mtp* shards).
    tc = cfg.get("text_config", cfg)
    tc["num_hidden_layers"] = frozen + tail
    if isinstance(tc.get("layer_types"), list):
        lt = tc["layer_types"]
        if tail_from is not None and tail_from != frozen:
            tc["layer_types"] = lt[:frozen] + lt[tail_from:tail_from + tail]
        else:
            tc["layer_types"] = lt[: frozen + tail]
    qc = tc.get("quantization_config") or cfg.get("quantization_config")
    if qc and isinstance(qc.get("quantized_layers"), dict):
        import re as _re
        # rebuild the per-layer quant table: keep 0..frozen-1 as-is, RENAME
        # tail_from..tail_from+tail-1 -> frozen..frozen+tail-1, drop the rest.
        new_q = {}
        for k, v in qc["quantized_layers"].items():
            m = _re.match(r"^model\.language_model\.layers\.(\d+)\.", k)
            if not m:
                new_q[k] = v
                continue
            L = int(m.group(1))
            if L < frozen:
                new_q[k] = v
            elif tail_from is not None and tail_from <= L < tail_from + tail:
                new_q[k.replace(f".layers.{L}.",
                                f".layers.{L - tail_from + frozen}.", 1)] = v
            elif tail_from is None and L < frozen + tail:
                new_q[k] = v
        qc["quantized_layers"] = new_q
    cfg.pop("num_hidden_layers", None)  # remove the earlier dead top-level write
    cfg["nested_frozen_layers"] = frozen
    cfg["nested_parent"] = os.path.basename(parent.rstrip("/"))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    for fn in TOK_FILES:
        p = os.path.join(parent, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, fn))

    # NESTED.json: the adoption permission (docs/13 s4)
    with open(os.path.join(out, "model.safetensors"), "rb") as w:
        base = 8 + len(hb)
        fz = frozen_set(ts, frozen)
        dig = []
        pfiles = {}
        for n in fz:
            fn, a, b = ts[n][2]
            pf = pfiles.setdefault(fn, open(os.path.join(parent, fn), "rb"))
            dig.append((n, sha_of(pf, a, b - a)))
        digest = hashlib.sha256()
        for n, s in dig:
            digest.update(n.encode()); digest.update(s.encode())
        for fn in ("vocab.json", "chat_template.jinja"):
            p = os.path.join(out, fn)
            if os.path.exists(p):
                with open(p, "rb") as g:
                    digest.update(fn.encode()); digest.update(hashlib.sha256(g.read()).hexdigest().encode())
        for f in pfiles.values():
            f.close()
        json.dump({
            "format": "tf-nested-v1",
            "frozen_prefix_digest": digest.hexdigest(),
            "frozen_tensors": len(fz),
            "frozen_layers": frozen, "tail_layers": tail,
            "parent": os.path.basename(parent.rstrip("/")),
            "geometry": {"kv_heads": cfg.get("num_key_value_heads"),
                         "head_dim": cfg.get("head_dim") or cfg.get("hidden_size", 0) // max(1, cfg.get("num_attention_heads", 1))},
        }, open(os.path.join(out, "NESTED.json"), "w"), indent=1)
    print(f"extracted {len(names)} tensors -> {out}/model.safetensors "
          f"({(8 + len(hb) + cur) / 2**30:.2f} GiB); frozen tensors {len(fz)}, "
          f"digest {digest.hexdigest()[:16]}...")
    return 0


def verify(parent, out, frozen):
    ts, os_ = parent_tensors(parent), parent_tensors(out)
    nested = json.load(open(os.path.join(out, "NESTED.json")))
    pfiles, ofiles = {}, {}
    bad = 0
    fz = frozen_set(ts, frozen)
    for n in fz:
        pf, a, b = ts[n][2]
        of, c, d = os_[n][2]
        f = pfiles.setdefault(pf, open(os.path.join(parent, pf), "rb"))
        g = ofiles.setdefault(of, open(os.path.join(out, of), "rb"))
        if sha_of(f, a, b - a) != sha_of(g, c, d - c):
            print("MISMATCH", n); bad += 1
    for f in list(pfiles.values()) + list(ofiles.values()):
        f.close()
    # digest reproducibility
    pfiles = {}
    digest = hashlib.sha256()
    for n in fz:
        fn, a, b = ts[n][2]
        pf = pfiles.setdefault(fn, open(os.path.join(parent, fn), "rb"))
        digest.update(n.encode()); digest.update(sha_of(pf, a, b - a).encode())
    for f in pfiles.values():
        f.close()
    for fn in ("vocab.json", "chat_template.jinja"):
        p = os.path.join(out, fn)
        if os.path.exists(p):
            with open(p, "rb") as g:
                digest.update(fn.encode()); digest.update(hashlib.sha256(g.read()).hexdigest().encode())
    ok = digest.hexdigest() == nested["frozen_prefix_digest"] and bad == 0
    print(f"verify: frozen bytes {'OK' if bad == 0 else f'{bad} MISMATCH'}; "
          f"digest {'OK' if ok else 'MISMATCH'} ({digest.hexdigest()[:16]}...)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=PARENT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen", type=int, default=10)
    ap.add_argument("--tail", type=int, default=6)
    ap.add_argument("--tail-from", type=int, default=None,
                    help="copy tail layers from parent range tail-from..+tail-1, "
                         "renumbered into the tail slot (5e last-6 byte-copy)")
    ap.add_argument("--force", action="store_true", help="overwrite existing out dir")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(verify(a.parent, a.out, a.frozen))
    if os.path.isdir(a.out) and not a.force:
        sys.exit(f"{a.out} exists; --force to overwrite")
    sys.exit(extract(a.parent, a.out, a.frozen, a.tail, a.tail_from))
