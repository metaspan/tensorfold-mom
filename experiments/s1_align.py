#!/usr/bin/env python3
"""Text-level position alignment for cross-engine kvdump checkpoint pairs (S1, docs/12).

Why this exists
---------------
s1_fit.py's strict mode only keeps checkpoint pairs whose token *ids* agree exactly. The two
engines apply different chat templates (the 35B lane carries thinking-block markup and drops the
leading <|im_start|> on some checkpoints) and each lane replays the B5 conversation with *its own*
assistant generations in history, so ids/lengths differ and only 2-3/29 pairs ever agreed. Both
lanes do tokenize the same vocab (their vocab.json files are byte-identical), so the two token
streams are two BPE renderings of mostly the same text — positions can be matched by decoded
piece text instead of by ids.

How alignment works
-------------------
1. Decode every token id to its raw bytes: vocab.json gives piece strings in GPT-2 byte-level BPE
   form (each char names a byte via bytes_to_unicode); added_tokens from tokenizer.json decode
   special ids (e.g. <|im_start|>) to their literal marker text. Unknown ids become "<|unk:N|>".
2. Split each side's piece list into turn *chunks*: a run of tokens starting at an <|im_start|>
   piece (tokens before the first one form a leading preamble chunk). Template turns are
   <|im_start|>role ... <|im_end|>\n, so chunks are turn-granular.
3. Pair chunks by an LCS over chunk texts with EXACT byte equality (monotonic, gap-tolerant).
   A diverged assistant turn (the two models answered "What is 2+2?" with "4" vs "2+2 equals 4.")
   simply fails to pair and contributes no positions — mismatched text is never force-aligned.
4. Within a matched chunk, align positions greedily by cumulative character (byte) offset with a
   two-pointer walk: advance the pointer whose accumulated text is shorter; positions whose
   accumulated texts are EXACTLY equal are ALIGNED. Equal-length-but-different accumulated text
   advances both pointers (dead branch — it can never re-equalise).

Caveat recorded in the fit report: a position inside user turn k aligns to the byte-identical text
in the other lane, but the *prior* context each engine saw includes that lane's own (diverged)
assistant generations — aligned text is identical, conditioning context is not bit-identical.

Run standalone for alignment stats only:
    /home/derek/.venv/bin/python s1_align.py --src <dumpdir>/qwen35-2b --dst <dumpdir>/qwen36-35b-a3b-nvfp4
stdlib + numpy only (numpy is needed by s1_fit, not by this module's pure functions).
"""

from __future__ import annotations

import json
import os


def bytes_to_unicode():
    """GPT-2/Qwen byte-level BPE byte<->char map (vocab pieces name bytes via these chars)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    ci = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            ci.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, ci)}


_BYTE_DECODER = {c: b for b, c in bytes_to_unicode().items()}


class Vocab:
    """id -> raw piece bytes, from a model dir's vocab.json + tokenizer.json added_tokens."""

    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "vocab.json")) as f:
            vocab = json.load(f)
        self.id2bytes = {i: bytes(_BYTE_DECODER[c] for c in piece) for piece, i in vocab.items()}
        self.special = {}
        tj_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tj_path):
            with open(tj_path) as f:
                tj = json.load(f)
            self.special = {a["id"]: a["content"].encode("utf-8") for a in tj.get("added_tokens", [])}

    def piece(self, tid) -> bytes:
        tid = int(tid)
        if tid in self.special:
            return self.special[tid]
        p = self.id2bytes.get(tid)
        if p is None:
            return f"<|unk:{tid}|>".encode("ascii")
        return p


IM_START_ID = 248045  # shared by both lanes' tokenizers (see tokenizer.json added_tokens)


def pieces_of(token_ids, vocab: Vocab):
    return [vocab.piece(t) for t in token_ids]


def turn_chunks(pieces, im_start: bytes):
    """Runs of tokens starting at an im_start piece; tokens before the first one form a preamble."""
    out, cur, start = [], None, None
    for i, p in enumerate(pieces):
        if p == im_start:
            if cur:
                out.append((start, cur))
            cur, start = [p], i
        else:
            if cur is None:
                cur, start = [], i
            cur.append(p)
    if cur:
        out.append((start, cur))
    return out


def lcs_match_chunks(chunks_s, chunks_d):
    """Monotonic LCS pairing of chunks whose full decoded texts are byte-identical.

    Returns [(ci_src, ci_dst), ...]. Only exact text equality can pair a chunk; nothing is
    force-aligned. Gap-tolerant so an extra/missing chunk (template drift, diverged turn that
    happens to be identical on one side) can't cascade into losing every later user turn.
    """
    ts = [b"".join(p for p in ch[1]) for ch in chunks_s]
    td = [b"".join(p for p in ch[1]) for ch in chunks_d]
    n, m = len(ts), len(td)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if ts[i] == td[j] else max(dp[i + 1][j], dp[i][j + 1])
    out = []
    i = j = 0
    while i < n and j < m:
        if ts[i] == td[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def align_chunk(ps, pd):
    """Two-pointer greedy alignment inside one text-identical chunk (the spec's rule).

    Positions i,j are ALIGNED iff the bytes accumulated from the chunk start through token i (src)
    and through token j (dst) are exactly equal; otherwise the pointer with the shorter
    accumulated text advances. Different BPE splits of the same text re-converge at every shared
    token boundary; boundary-internal positions stay unaligned. Equal-length, different text
    advances both (permanently dead branch). Returns [(i, j), ...] chunk-local indices.
    """
    cum_s, acc = [], b""
    for p in ps:
        acc += p
        cum_s.append(acc)
    cum_d, acc = [], b""
    for p in pd:
        acc += p
        cum_d.append(acc)
    pairs = []
    i = j = 0
    while i < len(ps) and j < len(pd):
        a, b_ = cum_s[i], cum_d[j]
        if a == b_:
            pairs.append((i, j))
            i += 1
            j += 1
        elif len(a) < len(b_):
            i += 1
        elif len(b_) < len(a):
            j += 1
        else:
            i += 1
            j += 1
    return pairs


def align_pieces(pieces_s, pieces_d, im_start: bytes):
    """Align one checkpoint pair's token positions by decoded text.

    Returns (pairs, stats): pairs = [(src_pos, dst_pos), ...] ascending on both sides;
    stats = {chunks_src, chunks_dst, chunks_matched, aligned, src_tokens, dst_tokens}.
    """
    cs, cd = turn_chunks(pieces_s, im_start), turn_chunks(pieces_d, im_start)
    matched = lcs_match_chunks(cs, cd)
    pairs = []
    for ci, cj in matched:
        st_s, pc_s = cs[ci]
        st_d, pc_d = cd[cj]
        for i, j in align_chunk(pc_s, pc_d):
            pairs.append((st_s + i, st_d + j))
    stats = {"src_tokens": len(pieces_s), "dst_tokens": len(pieces_d),
             "chunks_src": len(cs), "chunks_dst": len(cd),
             "chunks_matched": len(matched), "aligned": len(pairs)}
    return pairs, stats


def _selftest_aligner():
    """Pure-function checks that need no vocab file."""
    im = b"<|im_start|>"
    # two BPEs of the same text: different splits must re-converge at shared boundaries
    ps = [b"ab", b"c", b"de"]
    pd = [b"a", b"bc", b"de"]
    got = align_chunk(ps, pd)
    assert got == [(1, 1), (2, 2)], got  # only exact accumulated-text equality aligns
    # a diverged chunk never pairs; the identical later chunk aligns fully (global positions 3,4)
    p1 = [im, b"x", b"y", im, b"u1"]
    p2 = [im, b"X!", b"y", im, b"u1"]
    pairs, st = align_pieces(p1, p2, im)
    assert st["chunks_matched"] == 1 and pairs == [(3, 3), (4, 4)], (pairs, st)
    # equal-length different accumulated text advances both, cannot align
    assert align_chunk([b"aa"], [b"bb"]) == []
    print("s1_align selftest: ok")


if __name__ == "__main__":
    import argparse

    _selftest_aligner()
    ap = argparse.ArgumentParser(description="alignment stats for kvdump checkpoint pairs")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    import s1_fit  # reuse the dump loader

    sd, dd = s1_fit.load_model_dir(a.src), s1_fit.load_model_dir(a.dst)
    vs = Vocab(sd[0].meta["model_dir"])
    vd = Vocab(dd[0].meta["model_dir"])
    tot = 0
    for s, d in zip(sd, dd):
        assert s.ckpt == d.ckpt
        pairs, st = align_pieces(pieces_of(s.token_ids, vs), pieces_of(d.token_ids, vd),
                                 vs.piece(IM_START_ID))
        tot += len(pairs)
        print(f"ckpt{s.ckpt}: {st}")
    print(f"TOTAL aligned position pairs: {tot} over {min(len(sd), len(dd))} checkpoint pairs")
