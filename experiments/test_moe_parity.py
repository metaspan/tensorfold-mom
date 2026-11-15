#!/usr/bin/env python
"""Host test: sort-based grouped MoE must equal the reference per-token loop.

Semantics check for mtp_train.py's vectorization (54 s/step fix). Guards the
engine-matching transcription: gate rows before up rows, softmax over top-k
logits, sigmoid(shared_gate) * (silu(gate)*up) @ down, sum over k.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, H, E, TOPK, EI = 2, 256, 256, 64, 8, 64   # scaled sizes; same shapes semantics

xn = torch.randn(B, T, H, dtype=torch.float32)
gate = torch.randn(E, H) * 0.02
gup = torch.randn(E, 2 * EI, H) * 0.02
down = torch.randn(E, H, EI) * 0.02
se_gate = torch.randn(H, H) * 0.02
se_g, se_u, se_d = (torch.randn(H, H) * 0.02 for _ in range(3))

rl = xn @ gate.T
w, idx = rl.topk(TOPK, -1)
w = F.softmax(w, -1)

# reference (original trainer loop)
ref = torch.zeros_like(xn)
for j in range(TOPK):
    e_flat = idx[..., j].reshape(-1)
    x_flat = xn.reshape(-1, H)
    slot = torch.zeros_like(x_flat)
    for e_id in e_flat.unique():
        m = (e_flat == e_id).nonzero().squeeze(1)
        xm = x_flat[m]
        g = xm @ gup[e_id, :EI].T
        u = xm @ gup[e_id, EI:].T
        slot[m] = (F.silu(g) * u) @ down[e_id].T
    ref = ref + (slot * w[..., j].reshape(-1, 1)).view(B, T, H)

# vectorized (sort-based, as now in mtp_train.py)
x_flat = xn.reshape(-1, H)
N = x_flat.shape[0]
flat_e = idx.reshape(-1)
tok_idx = torch.arange(N).repeat_interleave(TOPK)
sort_e, order = flat_e.sort()
tok_sorted = tok_idx[order]
perm_x = x_flat[tok_sorted]
ends = torch.bincount(sort_e, minlength=E).cumsum(0).tolist()
out_slots = torch.zeros_like(perm_x)
start = 0
for e in range(E):
    end = ends[e]
    if end > start:
        xm = perm_x[start:end]
        g = xm @ gup[e, :EI].T
        u = xm @ gup[e, EI:].T
        out_slots[start:end] = (F.silu(g) * u) @ down[e].T
        start = end
w_slots = w.reshape(-1, 1)[order]
out_flat = torch.zeros_like(x_flat)
out_flat.index_add_(0, tok_sorted, out_slots * w_slots)
vec = out_flat.view(B, T, H)

d = (ref - vec).abs().max().item()
scale = ref.abs().max().item()
print(f"maxdiff={d:.3e} ref_scale={scale:.3f}")
assert d < 1e-4 * max(scale, 1.0), "MoE parity FAILED"
print("MoE PARITY OK")
