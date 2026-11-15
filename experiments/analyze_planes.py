#!/usr/bin/env python3
"""BL-29: analyze adopt/save dumps with the PLANE layout cache.state actually uses.
cache.state = [rec planes: deltas*recBytes][conv planes: deltas*convBytes]
(cuda_cache.zig's Seg view: rec = state.ptr, conv = state.ptr + deltas*recBytes).
Usage: analyze_planes.py FILE DELTAS [RECB CONVB]"""
import sys, struct

def main():
    path = sys.argv[1]
    deltas = int(sys.argv[2])
    recb = int(sys.argv[3]) if len(sys.argv) > 3 else 2097152   # v_heads*dk*dk*4 = 32*128*128*4
    convb = int(sys.argv[4]) if len(sys.argv) > 4 else 49152    # v_heads*4*dk*2 (bf16 taps)
    data = open(path, "rb").read()
    total = deltas * (recb + convb)
    print(f"{path}: {len(data)} bytes (plane layout needs {total}), deltas={deltas}")
    if len(data) < total:
        print(f"  !! file shorter than one state block ({total}); analyzing min only")
        total = len(data)
    conv_base = deltas * recb
    for d in range(deltas):
        rs, re_ = d * recb, min((d + 1) * recb, len(data))
        cs, ce = conv_base + d * convb, min(conv_base + (d + 1) * convb, len(data))
        rec = data[rs:re_]
        conv = data[cs:ce] if cs < len(data) else b""
        nrec = len(rec) // 4
        recnf = 0
        reczero = rec.count(0)
        # count non-finite f32 (exp all-ones)
        for i in range(nrec):
            w = struct.unpack_from("<I", rec, i * 4)[0]
            if (w & 0x7F800000) == 0x7F800000:
                recnf += 1
        nconv = len(conv) // 2
        convnf = 0
        convzero = conv.count(0)
        for i in range(nconv):
            h = struct.unpack_from("<H", conv, i * 2)[0]
            if (h & 0x7F80) == 0x7F80:
                convnf += 1
        print(f"  delta {d:2d}: rec {len(rec):>9}B nonfinite={recnf:<6} zerobytes={reczero:<9} | conv {len(conv):>7}B nonfinite={convnf:<5} zerobytes={convzero}")

if __name__ == "__main__":
    main()
