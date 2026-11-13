#!/usr/bin/env python3
"""BL-21(b): assemble a nested thin checkpoint whose directory also carries the
parent's own MTP head (mtp.* bytes, copied NEVER requantized), so mom-serve can
boot the thin lane with --drafter and draft with the target's own MTP layer.

Layout: hardlink every nested-16l shard (read-only source), append one new shard
holding the parent's 19 mtp.* tensors verbatim, write a merged
model.safetensors.index.json. Config is copied unchanged (nested-16l already
carries text_config.mtp_num_hidden_layers=1).
"""
import argparse, glob, json, os, shutil, struct
from collections import defaultdict

PARENT = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
NESTED = "/home/derek/models/tensorfold-mom/qwen36-nested-16l"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=PARENT)
    ap.add_argument("--nested", default=NESTED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) hardlink the nested shards + sidecars (engine never writes these)
    index_map = {}
    for p in glob.glob(os.path.join(args.nested, "model*.safetensors")):
        dst = os.path.join(args.out, os.path.basename(p))
        if os.path.exists(dst):
            os.remove(dst)
        os.link(p, dst)
    idx = os.path.join(args.nested, "model.safetensors.index.json")
    if os.path.exists(idx):
        index_map = json.load(open(idx))["weight_map"]
    else:  # single-file checkpoint
        h = read_header(os.path.join(args.nested, "model.safetensors"))
        for k in h:
            if k != "__metadata__":
                index_map[k] = "model.safetensors"

    # 2) pull the parent's mtp.* tensors verbatim into one new shard
    mtp_tensors, mtp_meta, count, total = {}, {}, 0, 0
    for p in sorted(glob.glob(os.path.join(args.parent, "model-*.safetensors"))):
        h = read_header(p)
        names = [k for k in h if k.startswith("mtp.")]
        if not names:
            continue
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            f.read(n)
            base = 8 + n
            for name in names:
                off = h[name]["data_offsets"]
                f.seek(base + off[0])
                blob = f.read(off[1] - off[0])
                mtp_tensors[name] = blob
                mtp_meta[name] = h[name]
                count += 1
                total += len(blob)
    assert count == 19, f"expected 19 mtp tensors, got {count}"

    # write the shard with the safetensors format (header json + blobs;
    # data_offsets are relative to the end of the header, starting at 0)
    hdr, cur = {}, 0
    order = sorted(mtp_tensors)
    for name in order:
        blob = mtp_tensors[name]
        hdr[name] = dict(mtp_meta[name])
        hdr[name]["data_offsets"] = [cur, cur + len(blob)]
        cur += len(blob)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-len(hb)) % 8
    hb += b" " * pad
    shard = "model-mtp-of-mtp.safetensors"
    with open(os.path.join(args.out, shard), "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name in order:
            f.write(mtp_tensors[name])
    for name in order:
        index_map[name] = shard

    json.dump({"metadata": {"total_size": total}, "weight_map": index_map},
              open(os.path.join(args.out, "model.safetensors.index.json"), "w"))

    # 3) sidecars: config (nested, carries mtp cfg) + tokenizer files
    for f in glob.glob(os.path.join(args.nested, "*.json")) + \
             glob.glob(os.path.join(args.nested, "*.txt")) + \
             glob.glob(os.path.join(args.nested, "*.jinja")):
        if os.path.basename(f).startswith("model-") or "index" in f:
            continue
        shutil.copy(f, os.path.join(args.out, os.path.basename(f)))
    print(f"{args.out}: {len(index_map)} tensors, mtp shard {total/2**30:.2f} GiB, {count} mtp tensors")


if __name__ == "__main__":
    main()
