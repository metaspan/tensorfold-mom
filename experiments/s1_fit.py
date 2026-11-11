#!/usr/bin/env python3
"""S1 offline fit harness (docs/12, item 6: cross-model translated KV cache feasibility).

Loads TF_MOM_KVDUMP dumps from two qwen3_5 engines (a small "src" lane and a large "dst" lane),
pairs each checkpoint N on the src lane with checkpoint N on the dst lane (the same conversation
prefix captured at the same request-start), ridge-fits a per-(src-layer, dst-layer) linear map for
K and for V — every pair, no layer-pairing assumption (6 -> 10 is a fitted index stretch) — and
scores the maps on held-out checkpoints with the pure-offline parts of the gate metric.

Two alignment modes select which (src position, dst position) pairs feed the fit:

  * --align text (primary): the engines use different chat templates and keep *their own*
    assistant generations in history, so token ids/lengths disagree for most checkpoint pairs and
    strict id equality leaves ~1 usable pair. Text mode decodes both token streams to piece
    bytes (byte-level BPE, see s1_align.py) and aligns positions whose accumulated decoded text is
    exactly equal within byte-identical turn chunks — mismatched text is never force-aligned.
  * --align strict (kept mode): only pairs whose token ids agree exactly, positions matched 1:1.

Metric definitions (all on held-out checkpoints):
  * cosine: mean over held-out rows of cos(pred, true) between the predicted and true
    concatenated kv_heads x head_dim row a position presents to attention.
  * top-1 (the gate metric): retrieval-style. For each held-out position p, take the predicted
    row's cosine against ONE representative true row per distinct held-out dst position (rows of
    the same logical position recur across checkpoints and are near-identical, so duplicates are
    grouped by dst position with the first occurrence as representative); the argmax must land on
    p's own row's group. top-1 = fraction of held-out rows that hit.
  * top1_self (legacy definition, kept as an extra column): per head, argmax over head_dim of the
    predicted row equals argmax of the true row at the SAME position; a position counts only if
    every kv head agrees.
  * first-divergence depth: walking held-out rows in checkpoint order (checkpoints ascending,
    aligned positions ascending within each), the index of the first row that is a top-1 miss or
    has cosine below --cos-floor; = number of held-out rows if none. Reported per fitted map; the
    gate takes the median over the best map per dst layer.

The pre-registered go/no-go (docs/12): go iff held-out first-divergence depth >= one full turn
median AND per-position top-1 >= 0.99. This harness reports that on the OFFLINE proxy; the
alignment change widens the corpus, it does NOT weaken the gate. The real gate metric — greedy
top-1 agreement of the 35B resumed from translated KV (GDN recomputed) vs the 35B's native
prefill — needs a device run and is the documented stub in eval_decode_resume().

GDN recurrent state is out of scope (the harness assumes it is recomputed), and dumps carry only
full-attention layers' K/V. Rows are bf16 (decoded to f32 here) or fp8 rows+scale decoded to f32
by the writer. Format spec: zig/src/families/qwen3_5/kvdump_format.zig.

Usage:
    python3 -m py_compile experiments/s1_fit.py
    /home/derek/.venv/bin/python experiments/s1_fit.py --src <dumpdir>/qwen35-2b \
        --dst <dumpdir>/qwen36-35b-a3b-nvfp4 --out fit-report-aligned.json
    /home/derek/.venv/bin/python experiments/s1_fit.py --align strict --src ... --dst ...
    /home/derek/.venv/bin/python experiments/s1_fit.py --selftest   # synthetic, all code paths

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_align  # noqa: E402  (sibling module: vocab decode + text alignment)

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

    def feats(self, layer, which, positions=None):
        """Positions x (kv_heads*head_dim): the row one position presents for that tensor.

        positions=None takes every position; otherwise only the given (aligned) positions.
        """
        arr = self.kv[(layer, which)]
        m = np.ascontiguousarray(np.transpose(arr, (1, 0, 2)).reshape(arr.shape[1], -1))
        if positions is not None:
            m = m[np.asarray(positions, dtype=np.intp)]
        return m


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


# --------------------------------------------------------------------------- alignment

def align(src_dumps, dst_dumps):
    """Strict mode: checkpoint pairs whose token ids agree position for position.

    Returns [(src, dst, pos_pairs)] with pos_pairs = [(i, i), ...] over the shared prefix.
    """
    pairs = []
    for s in src_dumps:
        for d in dst_dumps:
            if s.at == d.at and s.at > 1 and np.array_equal(s.token_ids, d.token_ids):
                pairs.append((s, d, [(i, i) for i in range(int(s.at))]))
                break
    return pairs


def align_text(src_dumps, dst_dumps):
    """Text mode: checkpoint N pairs with checkpoint N (same conversation prefix); positions are
    matched by decoded piece text (s1_align) instead of by token id. Returns
    [(src, dst, pos_pairs, align_stats)], dropping pairs with nothing alignable.
    """
    ds = {d.ckpt: d for d in dst_dumps}
    vs = s1_align.Vocab(src_dumps[0].meta["model_dir"])
    vd = s1_align.Vocab(dst_dumps[0].meta["model_dir"])
    im = vs.piece(s1_align.IM_START_ID)
    out = []
    for s in src_dumps:
        d = ds.get(s.ckpt)
        if d is None:
            continue
        pos_pairs, stats = s1_align.align_pieces(
            s1_align.pieces_of(s.token_ids, vs), s1_align.pieces_of(d.token_ids, vd), im)
        if pos_pairs:
            out.append((s, d, pos_pairs, stats))
    return out


# --------------------------------------------------------------------------- fit + metrics

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


def _unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(n, 1e-30)


def score_map(Xtr, Ytr, Xev, Yev, head_dim, cos_floor, lam, ev_pos=None):
    """Held-out cosine, retrieval top-1, legacy top-1 and first-divergence for one fitted map.

    Metric definitions (see module docstring; each is documented at its computation below).
    ev_pos: aligned DST position of every held-out row (same order as Xev/Yev rows). Required
    for the retrieval top-1; without it top-1 falls back to the legacy same-position definition.
    """
    W = ridge_fit(Xtr, Ytr, lam)
    P = ridge_apply(W, Xev)
    # cosine: predicted vs true concatenated kv_heads x head_dim row at the same position
    cos = _cos_rows(P, Yev)
    P32 = P.reshape(P.shape[0], -1, head_dim)
    T32 = Yev.reshape(Yev.shape[0], -1, head_dim)
    # top1_self (legacy): per-head argmax over head_dim agrees, every head must match
    top_ok_self = np.all(np.argmax(P32, axis=2) == np.argmax(T32, axis=2), axis=1)
    # top-1 (gate metric): retrieval against one representative true row per distinct held-out
    # dst position (near-duplicate rows of the same logical position across checkpoints are
    # grouped; grouping cannot weaken discrimination — duplicates carry identical information).
    hit = None
    gallery_size = None
    margin_mean = margin_min = float("nan")
    if ev_pos is not None and len(ev_pos):
        group_of, rep_rows, rep_of_pos = {}, [], {}
        tgt = np.empty(Yev.shape[0], dtype=np.intp)
        for r, p in enumerate(ev_pos):
            g = rep_of_pos.get(p)
            if g is None:
                g = len(rep_rows)
                rep_of_pos[p] = g
                rep_rows.append(r)
            tgt[r] = g
        G = _unit(Yev[np.asarray(rep_rows)])
        C = _unit(P) @ G.T  # cosine of each predicted row against every held-out dst position
        hit = np.argmax(C, axis=1) == tgt
        gallery_size = len(rep_rows)
        # separation margin: own-row cosine minus the best WRONG gallery cosine. A top-1 of 1.0
        # with tiny margins means the rows are nearly collinear and the metric is fragile.
        if gallery_size > 1:
            own = C[np.arange(C.shape[0]), tgt]
            Cx = C.copy()
            Cx[np.arange(C.shape[0]), tgt] = -np.inf
            marg = own - Cx.max(axis=1)
            margin_mean = float(np.mean(marg))
            margin_min = float(np.min(marg))
    else:
        hit = top_ok_self
    bad = (~hit) | (cos < cos_floor)
    diverg = int(np.argmax(bad)) if bad.any() else Xev.shape[0]
    return {
        "cosine": float(np.mean(cos)) if len(cos) else float("nan"),
        "top1": float(np.mean(hit)) if len(hit) else float("nan"),
        "top1_self": float(np.mean(top_ok_self)) if len(top_ok_self) else float("nan"),
        "gallery_size": gallery_size,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "positions": int(Xev.shape[0]),
        "first_divergence": diverg,
        "map": W,
    }


def _split(pairs, mode, heldout_frac, holdout_every):
    """Checkpoint-disjoint train/held-out split. Text mode: every `holdout_every`-th checkpoint
    (ckpt % holdout_every == holdout_every-1) is held out. Strict mode keeps the old tail split."""
    if mode == "text":
        keep = holdout_every - 1
        ev = [p for p in pairs if p[0].ckpt % holdout_every == keep]
        tr = [p for p in pairs if p[0].ckpt % holdout_every != keep]
        if not ev or not tr:
            raise SystemExit(f"split leaves train={len(tr)} heldout={len(ev)} — need more checkpoints")
        return tr, ev
    n_ev = max(1, int(math.ceil(heldout_frac * len(pairs))))
    if len(pairs) <= n_ev:
        raise SystemExit(f"need more aligned checkpoints than the {n_ev} held out (got {len(pairs)})")
    return pairs[: -n_ev], pairs[-n_ev:]


def fit_and_score(src, dst, pairs, mode, heldout_frac, cos_floor, lam=1e-2, holdout_every=4,
                  pairs_ev=None):
    """Fit every (src layer, dst layer) K/V map on train checkpoints; score on held-out ones.

    pairs: [(src_dump, dst_dump, pos_pairs, stats?)] as produced by align()/align_text().
    pairs_ev: when given, train on ALL of `pairs` and score on ALL of `pairs_ev` — a
    DISJOINT conversation, so every eval row is novel (nested-prefix holdout cannot give
    that: checkpoints share their rows verbatim with the train set).
    """
    if pairs_ev is not None:
        train = [(p[0], p[1], p[2]) for p in pairs]
        ev = [(p[0], p[1], p[2]) for p in pairs_ev]
        mode = "text-cross"
    else:
        train, ev = _split(pairs, mode, heldout_frac, holdout_every)
        train = [(p[0], p[1], p[2]) for p in train]  # normalize to (src, dst, pos_pairs)
        ev = [(p[0], p[1], p[2]) for p in ev]
    src_layers, dst_layers = src.layers, dst.layers
    ev_pos = [int(q) for _s, _d, pp in ev for _i, q in pp]
    train_pos = set(int(q) for _s, _d, pp in train for _i, q in pp)
    rng_ctl = np.random.default_rng(1234)
    rows = []
    verbatim_fracs = []
    best_per_dst: dict = {d: None for d in dst_layers}
    controls = {"shuffled": [], "constant": []}
    for sl in src_layers:
        for dl in dst_layers:
            for which in ("K", "V"):
                Xtr_w = np.concatenate([s.feats(sl, which, [i for i, _ in pp]) for s, _d, pp in train])
                Ytr_w = np.concatenate([d.feats(dl, which, [j for _, j in pp]) for _s, d, pp in train])
                Xev_w = np.concatenate([s.feats(sl, which, [i for i, _ in pp]) for s, _d, pp in ev])
                Yev_w = np.concatenate([d.feats(dl, which, [j for _, j in pp]) for _s, d, pp in ev])
                # verbatim leakage: share of eval dst-rows that occur byte-identically in the
                # train set (common chat-template/system prefixes do this across corpora; a
                # high fraction means the cross-corpus gate is really re-scoring train rows).
                tr_keys = {r.tobytes() for r in Ytr_w}
                verbatim_fracs.append(float(np.mean([r.tobytes() in tr_keys for r in Yev_w])))
                sc = score_map(Xtr_w, Ytr_w, Xev_w, Yev_w, src.head_dim, cos_floor, lam, ev_pos=ev_pos)
                # negative controls: a top-1 near 1.0 only means something if (a) fitting on
                # randomly PERMUTED train targets (no src->dst signal left) and (b) predicting a
                # CONSTANT row both score at chance. If either control scores high, the retrieval
                # metric is degenerate (gallery rows too collinear) and a GO would be meaningless.
                perm = rng_ctl.permutation(Ytr_w.shape[0])
                sc_shuf = score_map(Xtr_w, Ytr_w[perm], Xev_w, Yev_w, src.head_dim, cos_floor, lam, ev_pos=ev_pos)
                sc_cons = score_map(Xtr_w, np.repeat(Ytr_w[:1], Ytr_w.shape[0], axis=0),
                                    Xev_w, Yev_w, src.head_dim, cos_floor, lam, ev_pos=ev_pos)
                controls["shuffled"].append(sc_shuf["top1"])
                controls["constant"].append(sc_cons["top1"])
                row = {"src_layer": sl, "dst_layer": dl, "tensor": which,
                       "cosine": sc["cosine"], "top1": sc["top1"], "top1_self": sc["top1_self"],
                       "gallery_size": sc["gallery_size"],
                       "margin_mean": sc["margin_mean"], "margin_min": sc["margin_min"],
                       "positions": sc["positions"], "first_divergence": sc["first_divergence"],
                       "train_positions": int(Xtr_w.shape[0])}
                rows.append(row)
                cur = best_per_dst[dl]
                if cur is None or (row["top1"], row["cosine"]) > (cur["top1"], cur["cosine"]):
                    best_per_dst[dl] = row  # ties on top-1 broken by held-out cosine
    turn_tokens = int(np.median([d.at for _, d, _ in (p[:3] for p in ev)])) or 1
    best = list(best_per_dst.values())
    gate = {
        "median_first_divergence": float(np.median([r["first_divergence"] for r in best])),
        "mean_top1": float(np.mean([r["top1"] for r in best])),
        "max_top1": float(np.max([r["top1"] for r in best])),
        "mean_top1_self": float(np.mean([r["top1_self"] for r in best])),
        "mean_cosine": float(np.mean([r["cosine"] for r in best])),
        "min_margin_over_best": float(np.min([r["margin_min"] for r in best])),
        "turn_tokens": turn_tokens,
        "rule": "go iff median first-divergence >= one turn median AND mean per-position top-1 >= 0.99 (docs/12)",
    }
    gate["proxy_go"] = bool(gate["median_first_divergence"] >= turn_tokens and gate["mean_top1"] >= 0.99)
    # diagnostic controls (not part of the pre-registered rule): if fitting on permuted targets or
    # predicting a constant row still retrieves at high top-1, the gallery is too collinear for
    # top-1 to certify anything and a GO must be treated as unproven.
    gate["control_shuffled_top1_mean"] = float(np.mean(controls["shuffled"]))
    gate["control_shuffled_top1_max"] = float(np.max(controls["shuffled"]))
    gate["control_constant_top1_mean"] = float(np.mean(controls["constant"]))
    gate["controls_ok"] = bool(gate["control_shuffled_top1_max"] < 0.9
                               and gate["control_constant_top1_mean"] < 0.9)
    ev_pos_set = set(ev_pos)
    counts = {
        "mode": mode,
        "checkpoint_pairs_loaded": len(pairs),
        "train_checkpoints": len(train),
        "heldout_checkpoints": len(ev),
        "total_aligned_positions": int(sum(len(p[2]) for p in pairs)),
        "train_positions": int(sum(len(p[2]) for p in train)),
        "heldout_positions": len(ev_pos),
        "heldout_unique_positions": len(ev_pos_set),
        "heldout_positions_also_in_train": len(ev_pos_set & train_pos),
    }
    if pairs_ev is not None:
        counts["eval_rows_verbatim_in_train_frac"] = float(np.mean(verbatim_fracs))
    return {"counts": counts, "alignment": [dict(p[3], ckpt=p[0].ckpt) for p in pairs if len(p) > 3],
            "train_checkpoints": len(train), "heldout_checkpoints": len(ev),
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


def _write_dump(dirpath, model, model_dir, ckpt, layers, kv_heads, token_ids, kv, elem=ELEM_BF16, head_dim=256):
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
        blob = bytearray(HEADER.pack(MAGIC, VERSION, ELEM_BF16, n, kv_heads, head_dim, seq, ckpt, 0, header_bytes))
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
            "head_dim": head_dim, "elem": "bf16", "cache_format": "bf16", "page_tokens": 0,
            "bin": bin_name,
            "data_bytes": n * 2 * kv_heads * seq * head_dim * 2,
            "token_hash": 0, "model_layers": layers[-1] + 1}
    with open(os.path.join(dirpath, f"ckpt{ckpt}.json"), "w") as f:
        json.dump(meta, f)


def _selftest_aligner():
    """Pure aligner checks that need no dumps and no vocab file."""
    im = b"<|im_start|>"
    # two BPEs of the same text: only shared token boundaries (exactly-equal accumulated text) align
    assert s1_align.align_chunk([b"ab", b"c", b"de"], [b"a", b"bc", b"de"]) == [(1, 1), (2, 2)]
    # a diverged turn chunk never pairs; identical later chunks still align (no force-alignment)
    pairs, st = s1_align.align_pieces([im, b"x", b"y", im, b"u1"], [im, b"X!", b"y", im, b"u1"], im)
    # diverged chunk0 does not pair (matched=1); the identical later chunk aligns fully
    assert st["chunks_matched"] == 1 and pairs == [(3, 3), (4, 4)], (pairs, st)
    # equal-length, different accumulated text advances both pointers and cannot align
    assert s1_align.align_chunk([b"aa"], [b"bb"]) == []
    print("aligner selftest: ok")


def _selftest_text(root, src_layers, dst_layers, kvh, hd, rng):
    """Text-mode pipeline check: the same scripted conversation on both lanes, but the engines'
    assistant turns DIFFER (as in the real dumps) and the dst lane merges one shared token pair
    into a single token (pure BPE drift). Only text-identical positions may reach the fit, and the
    synthetic rows are built once over the maximal stream so a position's row is consistent across
    checkpoints (prefixes), exactly like a real engine's causal KV. Returns True if run.
    """
    vdir = "/home/derek/models/tensorfold-mom/qwen35-2b"
    if not os.path.exists(os.path.join(vdir, "vocab.json")):
        print("text-mode selftest: SKIP (no vocab.json at " + vdir + ")")
        return False
    vocab = s1_align.Vocab(vdir)
    bytes2id = {b: i for i, b in vocab.id2bytes.items()}
    im_id = s1_align.IM_START_ID
    im_piece = vocab.piece(im_id)
    words = [b"hello", b"world", b"alpha", b"beta", b"gamma", b"delta"]
    wid = []
    for w in words:
        i = bytes2id.get(w) or bytes2id.get(b" " + w)
        assert i is not None, f"fixture vocab lacks {w!r}"
        wid.append(i)
    # force the FIRST user query to be a mergeable adjacent pair so every checkpoint carries the
    # same dst-side BPE drift (keeps checkpoint streams prefix-consistent)
    merge = None
    for k in range(len(wid) - 1):
        joined = vocab.piece(wid[k]) + vocab.piece(wid[k + 1])
        if joined in bytes2id:
            merge = (wid[k], wid[k + 1], bytes2id[joined])
            break
    n_turns, n_ckpt = 15, 28  # checkpoint ci replays the first (ci+1)//2+1 turns
    # one maximal token stream per lane: turn0 = the mergeable query, later queries rotate words
    s_full = [im_id, 8678, 8678]  # system chunk text is identical both lanes
    d_full = list(s_full)
    for t in range(n_turns):
        if t == 0:
            q = [wid[0], wid[1]]
        else:
            q = [wid[t % len(wid)], wid[(t + 3) % len(wid)]]
        user = [im_id, 198] + q + [248046, 198]
        s_full += user
        d_full += user
        if t == 0 and merge:  # dst merges the query pair into one token (same decoded text)
            for k in range(len(d_full) - 1):
                if d_full[k] == merge[0] and d_full[k + 1] == merge[1]:
                    d_full[k:k + 2] = [merge[2]]
                    break
        s_full += [im_id, 198, 8678 + t, 248046, 198]     # THIS lane's assistant text
        d_full += [im_id, 198, 61446 + t, 248046, 198]    # OTHER lane's assistant text (differs)
    # rows for the maximal streams; per-position rows are then identical across checkpoints
    Smax, Dmax = len(s_full), len(d_full)
    Wtrue = {(sl, dl, w): rng.normal(0, 1 / math.sqrt(kvh * hd), size=(kvh * hd, kvh * hd)).astype(np.float32)
             for sl in src_layers for dl in dst_layers for w in ("K", "V")}
    btrue = {(sl, dl, w): rng.normal(0, 0.05, size=(kvh * hd,)).astype(np.float32)
             for sl in src_layers for dl in dst_layers for w in ("K", "V")}
    Xfull = {}
    for sl in src_layers:
        for w in ("K", "V"):
            a = rng.normal(0, 1, size=(kvh, Smax, hd)).astype(np.float32)
            for t in range(Smax):
                a[(t % (kvh * hd)) // hd, t, (t % (kvh * hd)) % hd] += 6.0
            Xfull[(sl, w)] = _bf16_grid(a)
    Yfull = {}
    for dl in dst_layers:
        sl = src_layers[min(range(len(src_layers)), key=lambda i: abs(src_layers[i] - dl * 6 / 10))]
        # the true pairing over the maximal streams decides where mapped rows land
        pos_full, _st = s1_align.align_pieces(s1_align.pieces_of(s_full, vocab),
                                              s1_align.pieces_of(d_full, vocab), im_piece)
        for w in ("K", "V"):
            X = np.transpose(Xfull[(sl, w)], (1, 0, 2)).reshape(Smax, -1)
            Y = _bf16_grid(X @ Wtrue[(sl, dl, w)].T + btrue[(sl, dl, w)])
            out = _bf16_grid(rng.normal(0, 1, size=(Dmax, kvh * hd)).astype(np.float32))
            for (i, j) in pos_full:
                out[j] = Y[i]
            Yfull[(dl, w)] = np.transpose(out.reshape(Dmax, kvh, hd), (1, 0, 2))
    src_dir, dst_dir = os.path.join(root, "t-qwen35-2b"), os.path.join(root, "t-qwen35b")
    for ci in range(n_ckpt):
        turn = (ci + 1) // 2 + 1
        # cut after the `turn`-th <|im_end|>+"\n" pair following the system chunk
        cut_s = len(s_full)
        seen = 0
        for k in range(3, len(s_full) - 4):
            if s_full[k] == 248046:
                seen += 1
                if seen == 2 * turn:  # user + assistant closers per turn
                    cut_s = k + 2
                    break
        seen = 0
        cut_d = len(d_full)
        for k in range(3, len(d_full) - 4):
            if d_full[k] == 248046:
                seen += 1
                if seen == 2 * turn:
                    cut_d = k + 2
                    break
        s_ids, d_ids = s_full[:cut_s], d_full[:cut_d]
        src_kv = {(sl, w): Xfull[(sl, w)][:, :len(s_ids), :] for sl in src_layers for w in ("K", "V")}
        dst_kv = {(dl, w): Yfull[(dl, w)][:, :len(d_ids), :] for dl in dst_layers for w in ("K", "V")}
        _write_dump(src_dir, "t-2b", vdir, ci, src_layers, kvh, s_ids, src_kv, head_dim=hd)
        _write_dump(dst_dir, "t-35b", vdir, ci, dst_layers, kvh, d_ids, dst_kv, head_dim=hd)
    sd, dd = load_model_dir(src_dir), load_model_dir(dst_dir)
    pairs = align_text(sd, dd)
    assert len(pairs) == n_ckpt, f"text alignment dropped checkpoints: {len(pairs)}"
    assert merge is None or any(p[3]["dst_tokens"] < p[3]["src_tokens"] for p in pairs), \
        "BPE-drift fixture produced no length drift"
    for p in pairs:
        assert p[3]["aligned"] > 0
        assert p[3]["chunks_matched"] < p[3]["chunks_src"], "diverged assistant turns must not pair"
    report = fit_and_score(sd[0], dd[0], pairs, mode="text", heldout_frac=0.25,
                           cos_floor=0.5, lam=1e-3, holdout_every=4)
    for r in report["best_per_dst_layer"]:
        assert r["cosine"] > 0.99, f"text-mode synthetic cosine {r['cosine']:.4f} for {r}"
        assert r["top1"] > 0.9, f"text-mode synthetic retrieval top-1 {r['top1']:.4f} for {r}"
    c = report["counts"]
    assert c["heldout_positions"] > 100, f"too few held-out rows to trust: {c}"
    g = report["offline_gate_proxy"]
    assert g["controls_ok"] and g["control_shuffled_top1_max"] < 0.5, \
        f"text-mode controls did not collapse: {g}"
    print(f"text-mode selftest: ok ({c['total_aligned_positions']} aligned positions, "
          f"{c['train_positions']} train / {c['heldout_positions']} held-out rows over "
          f"{report['train_checkpoints']}+{report['heldout_checkpoints']} checkpoints)")
    return True


def _selftest():
    rng = np.random.default_rng(7)
    kvh, hd = 2, 256
    src_layers = [3, 7, 11, 15, 19, 23]
    dst_layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    _selftest_aligner()
    seqs = [256, 384, 512, 640, 768, 896, 1024]  # max TRAIN seq >= kvh*hd: distinct prefix rows must span R^512
    tokens_all = rng.integers(1000, 60000, size=seqs[-1], dtype=np.uint32)
    # true maps: every dst row is a ridge-recoverable affine image of the src rows, plus bf16 noise
    Wtrue = {(sl, dl, w): rng.normal(0, 1 / math.sqrt(kvh * hd), size=(kvh * hd, kvh * hd)).astype(np.float32)
             for sl in src_layers for dl in dst_layers for w in ("K", "V")}
    btrue = {(sl, dl, w): rng.normal(0, 0.05, size=(kvh * hd,)).astype(np.float32)
             for sl in src_layers for dl in dst_layers for w in ("K", "V")}
    with tempfile.TemporaryDirectory() as root:
        src_dir, dst_dir = os.path.join(root, "qwen35-2b"), os.path.join(root, "qwen36-35b-a3b-nvfp4")
        # rows are drawn ONCE for the maximal sequence and sliced per checkpoint prefix: like a
        # real causal engine, a position's row must be identical across checkpoints, else the
        # retrieval metric's "group rows by dst position" premise breaks.
        src_full, dst_full = {}, {}
        for sl in src_layers:
            for w in ("K", "V"):
                a = rng.normal(0, 1, size=(kvh, seqs[-1], hd)).astype(np.float32)
                # a dominant per-position signal so a recovered map's argmax is stable under bf16
                for t in range(seqs[-1]):
                    a[(t % (kvh * hd)) // hd, t, (t % (kvh * hd)) % hd] += 6.0
                src_full[(sl, w)] = _bf16_grid(a)
        for dl in dst_layers:
            for w in ("K", "V"):
                sl = src_layers[min(range(len(src_layers)), key=lambda i: abs(src_layers[i] - dl * 6 / 10))]
                X = np.transpose(src_full[(sl, w)], (1, 0, 2)).reshape(seqs[-1], -1)
                Y = _bf16_grid(X @ Wtrue[(sl, dl, w)].T + btrue[(sl, dl, w)])  # bf16 lattice: the writer's u16 path
                dst_full[(dl, w)] = np.transpose(Y.reshape(seqs[-1], kvh, hd), (1, 0, 2))
        for ci, seq in enumerate(seqs):
            ids = tokens_all[:seq]
            src_kv = {(sl, w): src_full[(sl, w)][:, :seq, :] for sl in src_layers for w in ("K", "V")}
            dst_kv = {(dl, w): dst_full[(dl, w)][:, :seq, :] for dl in dst_layers for w in ("K", "V")}
            _write_dump(src_dir, "qwen35-2b", "/models/qwen35-2b", ci, src_layers, kvh, ids, src_kv)
            _write_dump(dst_dir, "qwen36-35b-a3b-nvfp4", "/models/qwen36-35b-a3b-nvfp4", ci, dst_layers, kvh, ids, dst_kv)
        src_d, dst_d = load_model_dir(src_dir), load_model_dir(dst_dir)
        assert len(src_d) == len(seqs) and len(dst_d) == len(seqs)
        pairs = align(src_d, dst_d)
        assert len(pairs) == len(seqs), f"alignment dropped checkpoints: {len(pairs)}"
        report = fit_and_score(src_d[0], dst_d[0], pairs, mode="strict",
                               heldout_frac=2.0 / len(seqs), cos_floor=0.5, lam=1e-3)
        g = report["offline_gate_proxy"]
        for r in report["best_per_dst_layer"]:
            # only the fitted best map per dst layer must recover the synthetic signal
            assert r["cosine"] > 0.99, f"synthetic linear map scored cosine {r['cosine']:.4f} for {r}"
            assert r["top1"] > 0.9, f"synthetic linear map scored top1 {r['top1']:.4f} for {r}"
        # the registered gate thresholds (top-1 >= 0.99, divergence >= a turn) are research criteria,
        # not correctness checks: the harness must report them for any corpus, synthetic included.
        assert set(g) >= {"median_first_divergence", "mean_top1", "turn_tokens", "proxy_go", "rule"}
        assert g["median_first_divergence"] >= 0 and 0.0 <= g["mean_top1"] <= 1.0
        # the negative controls must score at chance on recoverable synthetic data
        assert g["controls_ok"] and g["control_shuffled_top1_max"] < 0.5, \
            f"controls did not collapse on synthetic data: {g}"
        try:
            eval_decode_resume(None, None, [], {})
            raise AssertionError("eval_decode_resume must be a stub")
        except NotImplementedError:
            pass
        print(f"SELFTEST PASS (strict): {report['train_checkpoints']} train / {report['heldout_checkpoints']} held-out checkpoints; "
              f"best-per-dst mean cosine {np.mean([r['cosine'] for r in report['best_per_dst_layer']]):.4f}, "
              f"mean top-1 {g['mean_top1']:.4f}, median first-divergence {g['median_first_divergence']:.0f} "
              f"(turn {g['turn_tokens']} tokens), proxy_go={g['proxy_go']}")
        _selftest_text(root, src_layers, dst_layers, kvh, 32,  # smaller hd: distinct aligned train rows (~90) must span R^(kvh*hd)
                       np.random.default_rng(11))


# --------------------------------------------------------------------------- main


def _print_report(report, a):
    c = report["counts"]
    mode_tag = c["mode"]
    print(f"alignment mode: {c['mode']}")
    print(f"checkpoint pairs loaded: {c['checkpoint_pairs_loaded']}  "
          f"total aligned positions: {c['total_aligned_positions']}  "
          f"(train {c['train_positions']} / heldout {c['heldout_positions']} positions; "
          f"{report['train_checkpoints']} train / {report['heldout_checkpoints']} held-out checkpoints)")
    print(f"held-out unique dst positions: {c['heldout_unique_positions']} "
          f"(of which {c['heldout_positions_also_in_train']} also appear in train rows)")
    if "eval_rows_verbatim_in_train_frac" in c:
        f = c["eval_rows_verbatim_in_train_frac"]
        print(f"eval rows occurring VERBATIM in train: {f:.4f} "
              + ("— corpus genuinely novel" if f < 0.2 else
                 "— WARNING: much of the eval set is memorised train rows; treat GO cautiously"))
    print(f"{'src':>4} {'dst':>4} {'T':>2} {'cosine':>8} {'top1':>7} {'top1self':>9} {'firstDiv':>9} {'pos':>7}")
    for r in report["pairs"]:
        print(f"{r['src_layer']:>4} {r['dst_layer']:>4} {r['tensor']:>2} {r['cosine']:>8.4f} "
              f"{r['top1']:>7.4f} {r['top1_self']:>9.4f} {r['first_divergence']:>9} {r['positions']:>7}")
    g = report["offline_gate_proxy"]
    print("\nbest map per dst layer:")
    print(f"{'dst':>4} {'T':>2} {'src':>4} {'cosine':>8} {'top1':>7} {'top1self':>9} {'firstDiv':>9} {'margMin':>8}")
    for r in report["best_per_dst_layer"]:
        print(f"{r['dst_layer']:>4} {r['tensor']:>2} {r['src_layer']:>4} {r['cosine']:>8.4f} "
              f"{r['top1']:>7.4f} {r['top1_self']:>9.4f} {r['first_divergence']:>9} {r['margin_min']:>8.4f}")
    print(f"\nbest-per-dst mean top-1 {g['mean_top1']:.4f} (max {g['max_top1']:.4f}, "
          f"legacy top1_self mean {g['mean_top1_self']:.4f}, mean cosine {g['mean_cosine']:.4f}, "
          f"min retrieval margin {g['min_margin_over_best']:.4f}), "
          f"median first-divergence {g['median_first_divergence']:.0f} tokens (one turn = {g['turn_tokens']})")
    print(f"negative controls: top-1 after fitting on shuffled train targets "
          f"mean {g['control_shuffled_top1_mean']:.4f} / max {g['control_shuffled_top1_max']:.4f}; "
          f"constant-row predictor {g['control_constant_top1_mean']:.4f} "
          f"({'ok — retrieval is discriminative' if g['controls_ok'] else 'FAILED — gallery too collinear, top-1 certifies nothing'})")
    print(f"offline proxy gate: {'GO' if g['proxy_go'] else 'NO-GO'} — {g['rule']}")
    if not g["proxy_go"]:
        print(f"NOTE(ceiling): best-per-dst held-out top-1 is {g['mean_top1']:.4f} mean / "
              f"{g['max_top1']:.4f} max against a pre-registered bar of 0.99 — "
              + ("FAR below it: a single ridge-linear map per (src,dst,tensor) pair does not "
                 "retrieve the correct 35B row at most positions. " if g["max_top1"] < 0.9 else
                 "close to but below the bar. ")
              + (f"With {c['heldout_positions']} held-out aligned positions "
                 "(vs 23 in the old strict-match run) this is no longer a data-thin verdict; "
                 "the gate itself was not weakened — only the corpus feeding it was widened to "
                 "text-aligned rows." if mode_tag == "text" else
                 "Strict mode runs on the legacy id-equal corpus only; use --align text for the "
                 "widened corpus. The gate itself is unchanged in both modes."))
    else:
        print(f"NOTE(ceiling): the offline proxy passes, but read it honestly: "
              f"{c['heldout_positions_also_in_train']}/{c['heldout_unique_positions']} held-out "
              "positions also occur in train rows (checkpoints are nested prefixes and rows are "
              "bit-identical across checkpoints), so the maps saw the evaluated rows during "
              "training; a position-disjoint slice is empty because the longest checkpoint is in "
              "train. This GO certifies linear fit/retrieval fidelity on the shared corpus, NOT "
              "generalization to unseen positions. Aligned positions also share exact text only: "
              "each lane's conditioning context includes its own (diverged) assistant generations.")
    if not g["controls_ok"]:
        print("NOTE: negative controls FAILED — treat any GO above as unproven; the retrieval "
              "gallery does not discriminate and the gate must be re-designed before deciding.")
    print("NOTE: the registered gate is the device decode-resume agreement (eval_decode_resume, TODO) — this is its pre-filter.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="src model's dump directory, e.g. <dumpdir>/qwen35-2b")
    ap.add_argument("--dst", help="dst model's dump directory, e.g. <dumpdir>/qwen36-35b-a3b-nvfp4")
    ap.add_argument("--align", choices=("text", "strict"), default="text",
                    help="text (default): pair ckpt N<->N and align positions by decoded text; "
                         "strict: legacy mode, only pairs with exactly equal token ids")
    ap.add_argument("--heldout-frac", type=float, default=0.25,
                    help="strict mode: fraction of aligned checkpoints held out (tail; default 0.25)")
    ap.add_argument("--holdout-every", type=int, default=4,
                    help="text mode: hold out every Nth checkpoint (default 4)")
    ap.add_argument("--ridge", type=float, default=1e-2, help="ridge penalty (default 1e-2)")
    ap.add_argument("--cos-floor", type=float, default=0.5, help="cosine below which a position counts divergent (default 0.5)")
    ap.add_argument("--eval-corpus", metavar="DIR",
                    help="generalization mode: contains the same two model subdirs as --src/--dst; "
                         "fit on all --src/--dst checkpoints, score ONLY on DIR's checkpoints "
                         "(novel conversation => novel rows)")
    ap.add_argument("--out", help="write the full report JSON here")
    ap.add_argument("--selftest", action="store_true", help="synthetic end-to-end run; needs no dumps")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not a.src or not a.dst:
        ap.error("--src and --dst are required (or --selftest)")
    src_d, dst_d = load_model_dir(a.src), load_model_dir(a.dst)
    if a.align == "text":
        pairs = align_text(src_d, dst_d)
        if not pairs:
            raise SystemExit("no checkpoint pair has any text-alignable position: dump both lanes "
                             "over the SAME corpus (the B5 28-turn conversation) with TF_MOM_KVDUMP set")
    else:
        pairs = align(src_d, dst_d)
        if not pairs:
            raise SystemExit("no checkpoint pair shares token ids: dump both lanes over the SAME "
                             "corpus (the B5 28-turn conversation) with TF_MOM_KVDUMP set (or use "
                             "--align text)")
    pairs_ev = None
    if a.eval_corpus:
        if a.align != "text":
            ap.error("--eval-corpus requires --align text")
        ev_src_d = load_model_dir(os.path.join(a.eval_corpus, os.path.basename(a.src)))
        ev_dst_d = load_model_dir(os.path.join(a.eval_corpus, os.path.basename(a.dst)))
        pairs_ev = align_text(ev_src_d, ev_dst_d)
        if not pairs_ev:
            raise SystemExit(f"eval corpus {a.eval_corpus}: no text-alignable positions")
    report = fit_and_score(src_d[0], dst_d[0], pairs, a.align, a.heldout_frac, a.cos_floor,
                           lam=a.ridge, holdout_every=a.holdout_every, pairs_ev=pairs_ev)
    report["src"], report["dst"] = a.src, a.dst
    report["eval_corpus"] = a.eval_corpus
    _print_report(report, a)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, default=lambda o: np.asarray(o).tolist())
        print(f"report written: {a.out}")


if __name__ == "__main__":
    main()
