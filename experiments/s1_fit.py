#!/usr/bin/env python3
"""S1 offline fit harness (docs/12, item 6: cross-model translated KV cache feasibility).

Loads TF_MOM_KVDUMP dumps from two qwen3_5 engines (a small "src" lane and a large "dst" lane),
keeps the checkpoint pairs whose token ids agree (the same conversation prefix captured at each
request-start checkpoint by both engines), ridge-fits a per-(src-layer, dst-layer) linear map for
K and for V — every pair, no layer-pairing assumption (6 -> 10 is a fitted index stretch) — and
scores the maps on held-out checkpoints with the pure-offline parts of the gate metric:

  * per-position cosine between predicted and true rows (per head, and over the concatenated
    kv_heads x head_dim row a position presents to attention), and
  * per-position top-1 agreement: argmax(predicted row) == argmax(true row), per head.
  * first-divergence depth: the first held-out position where top-1 stops agreeing (any head) or
    cosine drops below --cos-floor; reported per checkpoint and as a median.

The pre-registered go/no-go (docs/12): go iff held-out first-divergence depth >= one full turn
median AND per-position top-1 >= 0.99. This harness reports that on the OFFLINE proxy. The real
gate metric — greedy top-1 agreement of the 35B resumed from translated KV (GDN recomputed) vs the
35B's native prefill — needs a device run and is the documented stub in eval_decode_resume().

GDN recurrent state is out of scope (the harness assumes it is recomputed), and dumps carry only
full-attention layers' K/V. Rows are bf16 (decoded to f32 here) or fp8 rows+scale decoded to f32
by the writer. Format spec: zig/src/families/qwen3_5/kvdump_format.zig.

Usage:
    python3 -m py_compile experiments/s1_fit.py
    python3 experiments/s1_fit.py --src <dumpdir>/qwen35-2b --dst <dumpdir>/qwen36-35b-a3b-nvfp4
    python3 experiments/s1_fit.py --selftest        # synthetic corpus, exercises every code path

stdlib + numpy only (torch is not required and not installed). The system python3 may lack numpy;
a venv that has it works, e.g. /home/derek/.venv/bin/python.
"""

import argparse
import json
import math
import os
import struct
import sys
import tempfile

try:
    import numpy as np
except ImportError:  # pragma: no cover - the honest failure mode on a bare interpreter
    sys.stderr.write(
        "s1_fit: numpy is required (stdlib fallbacks are not provided). The system python3 may not\n"
        "have it; try a venv that does, e.g.:  /home/derek/.venv/bin/python experiments/s1_fit.py\n"
    )
    raise

MAGIC = b"TFKVDMP1"
VERSION = 1
HEADER = struct.Struct("<8s9I")  # magic, version, elem, n_layers, kv_heads, head_dim, seq_len, ckpt, page_tokens, header_bytes
ELEM_BF16, ELEM_F32 = 0, 1


class Dump:
    """One ckpt<N>.bin sidecar pair: metadata plus K/V rows as f32 [kv_heads, seq, head_dim]."""

    def __init__(self, path, meta, token_ids, kv):
        self.path = path
        self.meta = meta
        self.ckpt = int(meta["ckpt"])
        self.model = meta["model"]
        self.layers = list(meta["layers"])
        self.kv_heads = int(meta["kv_heads"])
        self.head_dim = int(meta["head_dim"])
        self.token_ids = token_ids
        self.kv = kv  # {(layer, 'K'|'V'): f32 [heads, seq, head_dim]}

    @property
    def at(self):
        return int(self.meta["at"])

    def feats(self, layer, which):
        """Positions x (kv_heads*head_dim): the row one position presents for that tensor."""
        arr = self.kv[(layer, which)]
        return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)).reshape(arr.shape[1], -1))


def load_model_dir(path):
    """Every ckpt*.json/bin pair under a model's dump directory, ordered by checkpoint number."""
    dumps = {}
    for name in sorted(os.listdir(path)):
        if not name.startswith("ckpt") or not name.endswith(".json"):
            continue
        json_path = os.path.join(path, name)
        with open(json_path) as f:
            meta = json.load(f)
        if meta.get("format") != "tfmom-kvdump" or meta.get("version") != VERSION:
            raise ValueError(f"{json_path}: not a {MAGIC!r}/v{VERSION} sidecar")
        bin_path = os.path.join(path, meta["bin"])
        with open(bin_path, "rb") as f:
            blob = f.read()
        if len(blob) < HEADER.size:
            raise ValueError(f"{bin_path}: shorter than the header")
        (magic, version, elem, n_layers, kvh, hd, seq, ckpt, _page, header_bytes) = HEADER.unpack_from(blob, 0)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"{bin_path}: bad magic/version {magic!r}/{version}")
        # the sidecar is the human view; the header is truth — cross-check them
        for k, v in (("kv_heads", kvh), ("head_dim", hd), ("seq_len", seq), ("ckpt", ckpt)):
            if int(meta[k]) != int(v):
                raise ValueError(f"{bin_path}: sidecar {k}={meta[k]} disagrees with header {v}")
        if len(meta["layers"]) != int(n_layers):
            raise ValueError(f"{bin_path}: sidecar layers count disagrees with header n_layers")
        lay_off, tok_off = HEADER.size, HEADER.size + 4 * n_layers
        layers = list(struct.unpack_from(f"<{n_layers}I", blob, lay_off))
        tokens = np.frombuffer(blob, dtype=np.uint32, count=seq, offset=tok_off).copy()
        if list(layers) != [int(x) for x in meta["layers"]]:
            raise ValueError(f"{bin_path}: header layer table disagrees with the sidecar")
        want = n_layers * 2 * kvh * seq * hd * (2 if elem == ELEM_BF16 else 4)
        if len(blob) < header_bytes + want:
            raise ValueError(f"{bin_path}: data region short ({len(blob) - header_bytes} < {want})")
        if elem == ELEM_BF16:
            raw = np.frombuffer(blob, dtype=np.uint16, count=n_layers * 2 * kvh * seq * hd, offset=header_bytes)
            data = (raw.astype(np.uint32) << 16).view(np.float32)
        elif elem == ELEM_F32:
            data = np.frombuffer(blob, dtype=np.float32, count=n_layers * 2 * kvh * seq * hd, offset=header_bytes).astype(np.float32)
        else:
            raise ValueError(f"{bin_path}: unknown elem {elem}")
        data = data.reshape(n_layers, 2, kvh, seq, hd)
        kv = {}
        for i, layer in enumerate(layers):
            kv[(layer, "K")] = np.ascontiguousarray(data[i, 0])
            kv[(layer, "V")] = np.ascontiguousarray(data[i, 1])
        if ckpt in dumps:
            raise ValueError(f"{path}: ckpt{ckpt} appears twice")
        dumps[ckpt] = Dump(bin_path, meta, tokens, kv)
    return [dumps[k] for k in sorted(dumps)]


def align(src_dumps, dst_dumps):
    """Checkpoint pairs whose token ids agree position for position (the same conversation prefix)."""
    pairs = []
    for s in src_dumps:
        for d in dst_dumps:
            if s.at == d.at and s.at > 1 and np.array_equal(s.token_ids, d.token_ids):
                pairs.append((s, d))
                break
    return pairs


def ridge_fit(X, Y, lam):
    """argmin_W ||XW - Y||^2 + lam*||W_unpenalized||^2 with an unpenalized bias column."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    d = np.ones(Xb.shape[1], dtype=X.dtype)
    d[-1] = 0.0  # the bias enters unpenalized
    A = Xb.T @ Xb + lam * np.diag(d)
    return np.linalg.solve(A, Xb.T @ Y)


def ridge_apply(W, X):
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)]) @ W


def _cos_rows(P, T):
    num = np.sum(P * T, axis=1)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)


def score_map(Xtr, Ytr, Xev, Yev, head_dim, cos_floor, lam):
    """Held-out cosine, top-1 and first-divergence for one fitted map."""
    W = ridge_fit(Xtr, Ytr, lam)
    P = ridge_apply(W, Xev)
    cos = _cos_rows(P, Yev)  # concatenated kv_heads x head_dim row a position presents
    P32 = P.reshape(P.shape[0], -1, head_dim)
    T32 = Yev.reshape(Yev.shape[0], -1, head_dim)
    top_ok = np.all(np.argmax(P32, axis=2) == np.argmax(T32, axis=2), axis=1)
    bad = (~top_ok) | (cos < cos_floor)
    diverg = int(np.argmax(bad)) if bad.any() else Xev.shape[0]
    return {
        "cosine": float(np.mean(cos)) if len(cos) else float("nan"),
        "top1": float(np.mean(top_ok)) if len(top_ok) else float("nan"),
        "positions": int(Xev.shape[0]),
        "first_divergence": diverg,
        "map": W,
    }


def fit_and_score(src, dst, pairs, heldout_frac, cos_floor, lam=1e-2):
    """Fit every (src layer, dst layer) K/V map on train checkpoints; score on the held-out tail."""
    n_ev = max(1, int(math.ceil(heldout_frac * len(pairs))))
    if len(pairs) <= n_ev:
        raise SystemExit(f"need more aligned checkpoints than the {n_ev} held out (got {len(pairs)})")
    train, ev = pairs[: -n_ev], pairs[-n_ev:]
    src_layers, dst_layers = src.layers, dst.layers
    rows, best_per_dst = [], {d: None for d in dst_layers}
    for sl in src_layers:
        for dl in dst_layers:
            for which in ("K", "V"):
                Xtr_w = np.concatenate([s.feats(sl, which) for s, _ in train])
                Ytr_w = np.concatenate([d.feats(dl, which) for _, d in train])
                Xev_w = np.concatenate([s.feats(sl, which) for s, _ in ev])
                Yev_w = np.concatenate([d.feats(dl, which) for _, d in ev])
                sc = score_map(Xtr_w, Ytr_w, Xev_w, Yev_w, src.head_dim, cos_floor, lam)
                row = {"src_layer": sl, "dst_layer": dl, "tensor": which,
                       "cosine": sc["cosine"], "top1": sc["top1"],
                       "positions": sc["positions"], "first_divergence": sc["first_divergence"],
                       "train_positions": int(Xtr_w.shape[0])}
                rows.append(row)
                if best_per_dst[dl] is None or row["top1"] > best_per_dst[dl]["top1"]:
                    best_per_dst[dl] = row
    turn_tokens = int(np.median([d.at for _, d in ev])) or 1
    best = list(best_per_dst.values())
    gate = {
        "median_first_divergence": float(np.median([r["first_divergence"] for r in best])),
        "mean_top1": float(np.mean([r["top1"] for r in best])),
        "turn_tokens": turn_tokens,
        "rule": "go iff median first-divergence >= one turn median AND mean per-position top-1 >= 0.99 (docs/12)",
    }
    gate["proxy_go"] = bool(gate["median_first_divergence"] >= turn_tokens and gate["mean_top1"] >= 0.99)
    return {"train_checkpoints": len(train), "heldout_checkpoints": len(ev),
            "src_layers": src_layers, "dst_layers": dst_layers,
            "pairs": rows, "best_per_dst_layer": best, "offline_gate_proxy": gate}


def eval_decode_resume(src, dst, pairs, maps, **_kw):
    """THE gate metric (docs/12 S1 step 4): greedy agreement of the 35B resumed from translated KV.

    TODO(S1-device): run the dst engine twice over each held-out prefix —
      (a) native prefill, greedy continuation;
      (b) GDN recomputed, then inject W-mapped K/V rows into the dst lane's KV pages (the S2
          warm-start path behind the docs/02 compat-digest extension point; TF_MOM_KVDUMP's
          checkpoints are the injection sites), and greedy-decode.
    Report per-position top-1 agreement between (a) and (b) and the first-divergence depth in
    tokens. Needs a GPU; deliberately out of this offline harness's scope, and never a substitute
    for it — the offline scores above are only a cheap pre-filter.
    """
    raise NotImplementedError("decode-resume needs a device run; see the TODO(S1-device) above")


# --------------------------------------------------------------------------- selftest


def _bf16_grid(a):
    """Snap to the bf16 lattice (f32 with the low 16 mantissa bits zero): what the writer stores."""
    return (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _write_dump(dirpath, model, model_dir, ckpt, layers, kv_heads, token_ids, kv, elem=ELEM_BF16):
    """Write one dump pair in the exact kvdump_format.zig layout (selftest fixtures)."""
    os.makedirs(dirpath, exist_ok=True)
    seq = len(token_ids)
    n = len(layers)
    header_bytes = ((HEADER.size + 4 * n + 4 * seq) + 7) // 8 * 8
    if elem == ELEM_BF16:
        blocks = []
        for layer in layers:
            for which in ("K", "V"):
                a = kv[(layer, which)].astype(np.float32)
                assert np.array_equal(_bf16_grid(a), a), "fixture must be bf16-representable"
                u = (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)  # bf16 bits = the f32 lead
                blocks.append(u)
        data = np.concatenate([b.reshape(-1) for b in blocks])
        blob = bytearray(HEADER.pack(MAGIC, VERSION, ELEM_BF16, n, kv_heads, 256, seq, ckpt, 0, header_bytes))
        blob += struct.pack(f"<{n}I", *layers)
        blob += np.asarray(token_ids, dtype=np.uint32).tobytes()
        blob += bytes(header_bytes - len(blob))  # the lead is 8-aligned; data starts at header_bytes
        blob = bytes(blob) + data.tobytes()
    else:
        raise NotImplementedError("selftest covers the bf16 path; the f32 decode path is covered by the Zig tests")
    bin_name = f"ckpt{ckpt}.bin"
    with open(os.path.join(dirpath, bin_name), "wb") as f:
        f.write(blob)
    meta = {"format": "tfmom-kvdump", "version": 1, "model": model, "model_dir": model_dir,
            "ckpt": ckpt, "at": seq, "seq_len": seq, "layers": layers, "kv_heads": kv_heads,
            "head_dim": 256, "elem": "bf16", "cache_format": "bf16", "page_tokens": 0,
            "bin": bin_name,
            "data_bytes": n * 2 * kv_heads * seq * 256 * 2,
            "token_hash": 0, "model_layers": layers[-1] + 1}
    with open(os.path.join(dirpath, f"ckpt{ckpt}.json"), "w") as f:
        json.dump(meta, f)


def _selftest():
    rng = np.random.default_rng(7)
    kvh, hd = 2, 256
    src_layers = [3, 7, 11, 15, 19, 23]
    dst_layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    seqs = [64, 96, 128, 160, 192, 224, 256]
    tokens_all = rng.integers(1000, 60000, size=seqs[-1], dtype=np.uint32)
    # true maps: every dst row is a ridge-recoverable affine image of the src rows, plus bf16 noise
    Wtrue = {(sl, dl, w): rng.normal(0, 1 / math.sqrt(kvh * hd), size=(kvh * hd, kvh * hd)).astype(np.float32)
             for sl in src_layers for dl in dst_layers for w in ("K", "V")}
    btrue = {(sl, dl, w): rng.normal(0, 0.05, size=(kvh * hd,)).astype(np.float32)
             for sl in src_layers for dl in dst_layers for w in ("K", "V")}
    with tempfile.TemporaryDirectory() as root:
        src_dir, dst_dir = os.path.join(root, "qwen35-2b"), os.path.join(root, "qwen36-35b-a3b-nvfp4")
        for ci, seq in enumerate(seqs):
            ids = tokens_all[:seq]
            src_kv, dst_kv = {}, {}
            for sl in src_layers:
                for w in ("K", "V"):
                    a = rng.normal(0, 1, size=(kvh, seq, hd)).astype(np.float32)
                    # a dominant per-position signal so a recovered map's argmax is stable under bf16
                    for t in range(seq):
                        a[(t % (kvh * hd)) // hd, t, (t % (kvh * hd)) % hd] += 6.0
                    src_kv[(sl, w)] = _bf16_grid(a)
            for dl in dst_layers:
                for w in ("K", "V"):
                    sl = src_layers[min(range(len(src_layers)), key=lambda i: abs(src_layers[i] - dl * 6 / 10))]
                    X = np.transpose(src_kv[(sl, w)], (1, 0, 2)).reshape(seq, -1)
                    Y = _bf16_grid(X @ Wtrue[(sl, dl, w)].T + btrue[(sl, dl, w)])  # bf16 lattice: the writer's u16 path
                    dst_kv[(dl, w)] = np.transpose(Y.reshape(seq, kvh, hd), (1, 0, 2))
            _write_dump(src_dir, "qwen35-2b", "/models/qwen35-2b", ci, src_layers, kvh, ids, src_kv)
            _write_dump(dst_dir, "qwen36-35b-a3b-nvfp4", "/models/qwen36-35b-a3b-nvfp4", ci, dst_layers, kvh, ids, dst_kv)
        src_d, dst_d = load_model_dir(src_dir), load_model_dir(dst_dir)
        assert len(src_d) == len(seqs) and len(dst_d) == len(seqs)
        pairs = align(src_d, dst_d)
        assert len(pairs) == len(seqs), f"alignment dropped checkpoints: {len(pairs)}"
        report = fit_and_score(src_d[0], dst_d[0], pairs, heldout_frac=2.0 / len(seqs), cos_floor=0.5, lam=1e-3)
        g = report["offline_gate_proxy"]
        for r in report["best_per_dst_layer"]:
            # only the fitted best map per dst layer must recover the synthetic signal
            assert r["cosine"] > 0.99, f"synthetic linear map scored cosine {r['cosine']:.4f} for {r}"
            assert r["top1"] > 0.9, f"synthetic linear map scored top1 {r['top1']:.4f} for {r}"
        # the registered gate thresholds (top-1 >= 0.99, divergence >= a turn) are research criteria,
        # not correctness checks: the harness must report them for any corpus, synthetic included.
        assert set(g) >= {"median_first_divergence", "mean_top1", "turn_tokens", "proxy_go", "rule"}
        assert g["median_first_divergence"] >= 0 and 0.0 <= g["mean_top1"] <= 1.0
        try:
            eval_decode_resume(None, None, [], {})
            raise AssertionError("eval_decode_resume must be a stub")
        except NotImplementedError:
            pass
        print(f"SELFTEST PASS: {report['train_checkpoints']} train / {report['heldout_checkpoints']} held-out checkpoints; "
              f"best-per-dst mean cosine {np.mean([r['cosine'] for r in report['best_per_dst_layer']]):.4f}, "
              f"mean top-1 {g['mean_top1']:.4f}, median first-divergence {g['median_first_divergence']:.0f} "
              f"(turn {g['turn_tokens']} tokens), proxy_go={g['proxy_go']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="src model's dump directory, e.g. <dumpdir>/qwen35-2b")
    ap.add_argument("--dst", help="dst model's dump directory, e.g. <dumpdir>/qwen36-35b-a3b-nvfp4")
    ap.add_argument("--heldout-frac", type=float, default=0.25, help="fraction of aligned checkpoints held out (tail; default 0.25)")
    ap.add_argument("--ridge", type=float, default=1e-2, help="ridge penalty (default 1e-2)")
    ap.add_argument("--cos-floor", type=float, default=0.5, help="cosine below which a position counts divergent (default 0.5)")
    ap.add_argument("--out", help="write the full report JSON here")
    ap.add_argument("--selftest", action="store_true", help="synthetic end-to-end run; needs no dumps")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not a.src or not a.dst:
        ap.error("--src and --dst are required (or --selftest)")
    src_d, dst_d = load_model_dir(a.src), load_model_dir(a.dst)
    pairs = align(src_d, dst_d)
    if not pairs:
        raise SystemExit("no checkpoint pair shares token ids: dump both lanes over the SAME corpus "
                         "(the B5 28-turn conversation) with TF_MOM_KVDUMP set")
    report = fit_and_score(src_d[0], dst_d[0], pairs, a.heldout_frac, a.cos_floor, lam=a.ridge)
    report["src"], report["dst"] = a.src, a.dst
    print(f"aligned checkpoints: {len(pairs)}  (train {report['train_checkpoints']} / held out {report['heldout_checkpoints']})")
    print(f"{'src':>4} {'dst':>4} {'T':>2} {'cosine':>8} {'top1':>7} {'firstDiv':>9} {'pos':>7}")
    for r in report["pairs"]:
        print(f"{r['src_layer']:>4} {r['dst_layer']:>4} {r['tensor']:>2} {r['cosine']:>8.4f} {r['top1']:>7.4f} {r['first_divergence']:>9} {r['positions']:>7}")
    g = report["offline_gate_proxy"]
    print(f"\nbest map per dst layer: mean top-1 {g['mean_top1']:.4f}, median first-divergence "
          f"{g['median_first_divergence']:.0f} tokens (one turn = {g['turn_tokens']})")
    print(f"offline proxy gate: {'GO' if g['proxy_go'] else 'NO-GO'} — {g['rule']}")
    print("NOTE: the registered gate is the device decode-resume agreement (eval_decode_resume, TODO) — this is its pre-filter.")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report written: {a.out}")


if __name__ == "__main__":
    main()
