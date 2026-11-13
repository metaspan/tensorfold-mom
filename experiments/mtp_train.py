#!/usr/bin/env python3
"""BL-30: train the MTP drafter head on the gated student (docs/13 part 3, BL-21 route b).

Semantics pinned from the engine (cuda_mtp.zig:319-323 + qwen36_mtp.cu):
  en = rmsnorm(emb(follow_token), mtp.pre_fc_norm_embedding)
  hn = rmsnorm(target_hidden_t,   mtp.pre_fc_norm_hidden)
  x0 = en @ fc[:, :H].T + hn @ fc[:, H:].T          (fc is [H, 2H])
  x1 = x0 + attn(input_layernorm(x0))               (1 GQA layer, 16q/2kv,
       head_dim 256, rotary on the first 25% dims, theta 1e7, q/k per-head norm)
  x2 = x1 + moe(post_attention_layernorm(x1))       (top-8 of 256 + shared x sigmoid gate)
  logits = rmsnorm(x2, mtp.norm) @ lm_head.T         (shared with the target)
Training: teacher-forced CE predicting x[t+2] from (hidden_t, embed x[t+1]) —
the draft position, exactly the serve-time contract (hidden tap = target's
last pre-norm residual; follow = the token sampled after it).

Only the 19 mtp.* tensors train; the student is frozen. Export writes
model-mtp-trained.safetensors, consumable by mtp_merge.py (--parent <this out>).

Run (GPU window, after the student gates >=10/12):
  python mtp_train.py --student <gated-student-dir> --data <rollouts.jsonl> \
      --out-dir <ckpt> [--epochs 3 --lr 1e-4]
"""
from __future__ import annotations
import argparse, json, math, os, sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PARENT_MTP = "/home/derek/models/tensorfold-mom/qwen36-35b-a3b-nvfp4"
H, QH, KVH, HD, ROT = 2048, 16, 2, 256, 64          # hidden, q heads, kv heads, head_dim, rotary dims
E, TOPK, EI = 256, 8, 512                            # experts, top-k, expert intermediate
THETA = 1e7


def rms(x, w, eps=1e-6):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


class MTPHead(torch.nn.Module):
    def __init__(self, state, embed_w, lm_head_w, rotary):
        super().__init__()
        s = {k: v.cuda().float() for k, v in state.items()}
        self.norm_e = torch.nn.Parameter(s["mtp.pre_fc_norm_embedding.weight"])
        self.norm_h = torch.nn.Parameter(s["mtp.pre_fc_norm_hidden.weight"])
        fc = s["mtp.fc.weight"]                      # [H, 2H]: cols 0:H emb, H:2H hidden
        self.fc_e = torch.nn.Parameter(fc[:, :H].contiguous())
        self.fc_h = torch.nn.Parameter(fc[:, H:].contiguous())
        self.in_ln = torch.nn.Parameter(s["mtp.layers.0.input_layernorm.weight"])
        self.post_ln = torch.nn.Parameter(s["mtp.layers.0.post_attention_layernorm.weight"])
        self.q = torch.nn.Parameter(s["mtp.layers.0.self_attn.q_proj.weight"])
        self.k = torch.nn.Parameter(s["mtp.layers.0.self_attn.k_proj.weight"])
        self.v = torch.nn.Parameter(s["mtp.layers.0.self_attn.v_proj.weight"])
        self.o = torch.nn.Parameter(s["mtp.layers.0.self_attn.o_proj.weight"])
        self.qn = torch.nn.Parameter(s["mtp.layers.0.self_attn.q_norm.weight"])
        self.kn = torch.nn.Parameter(s["mtp.layers.0.self_attn.k_norm.weight"])
        self.gate = torch.nn.Parameter(s["mtp.layers.0.mlp.gate.weight"])
        self.gup = torch.nn.Parameter(s["mtp.layers.0.mlp.experts.gate_up_proj"])   # [E,2EI,H]
        self.down = torch.nn.Parameter(s["mtp.layers.0.mlp.experts.down_proj"])     # [E,H,EI]
        self.se_g = torch.nn.Parameter(s["mtp.layers.0.mlp.shared_expert.gate_proj.weight"])
        self.se_u = torch.nn.Parameter(s["mtp.layers.0.mlp.shared_expert.up_proj.weight"])
        self.se_d = torch.nn.Parameter(s["mtp.layers.0.mlp.shared_expert.down_proj.weight"])
        self.se_gate = torch.nn.Parameter(s["mtp.layers.0.mlp.shared_expert_gate.weight"])
        self.out_norm = torch.nn.Parameter(s["mtp.norm.weight"])
        self.register_buffer("embed", embed_w.cuda().bfloat16(), persistent=False)
        self.register_buffer("lm_head", lm_head_w.cuda().bfloat16(), persistent=False)
        self.register_buffer("cos", rotary[0], persistent=False)
        self.register_buffer("sin", rotary[1], persistent=False)

    def forward(self, hidden, ids, mask):
        # hidden: [B,T,H] student pre-final-norm residual at t; ids: [B,T] tokens x[t]
        # draft at position t consumes follow token x[t+1] -> predicts x[t+2]
        B, T, _ = hidden.shape
        follow = torch.roll(ids, -1, 1)
        en = rms(F.embedding(follow, self.embed.float()), self.norm_e)
        hn = rms(hidden.float(), self.norm_h)
        x0 = en @ self.fc_e.T.to(en.dtype) + hn @ self.fc_h.T.to(hn.dtype)
        x0 = x0.to(torch.bfloat16)
        # attention (fp32 math; head is small, bf16 only at the lm_head matmul)
        x0f = x0.float()
        xn = rms(x0f, self.in_ln)
        qg = xn @ self.q.T                                        # rows of interleaved [q_head_h(256) | gate_head_h(256)] (kernel: G + h*(2*HD) + HD)
        pair = qg.view(B, T, QH, 2, HD)
        q, gate_q = pair[:, :, :, 0], pair[:, :, :, 1]
        k = (xn @ self.k.T).view(B, T, KVH, HD)
        v = (xn @ self.v.T).view(B, T, KVH, HD)
        q = rms(q, self.qn); k = rms(k, self.kn)     # per-head over HD
        pos = torch.arange(T, device=x0.device)
        c = self.cos[pos][:, :ROT].float()[None, :, None, :]; s = self.sin[pos][:, :ROT].float()[None, :, None, :]
        def rot(z):  # partial rotary on the first ROT dims, interleaved pairs
            z1, z2 = z[..., :ROT:2], z[..., 1:ROT:2]
            r1 = z1 * c - z2 * s; r2 = z2 * c + z1 * s
            return torch.cat([torch.stack([r1, r2], -1).flatten(-2), z[..., ROT:]], -1)
        q, k = rot(q).transpose(1, 2), rot(k).transpose(1, 2)   # B,H,T,D
        v = v.transpose(1, 2)
        k = k.repeat_interleave(QH // KVH, 1); v = v.repeat_interleave(QH // KVH, 1)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att * torch.sigmoid(gate_q.transpose(1, 2))       # gated attention (Qwen3-Next style)
        x1 = x0f + (att.transpose(1, 2).reshape(B, T, QH * HD) @ self.o.T)
        # MoE: norm_topk_prob=true is engine-enforced (config.zig:338 rejects
        # false); gate rows precede up rows (cuda_mtp_bf16.zig:54).
        xn = rms(x1, self.post_ln)
        rl = xn @ self.gate.T                                   # B,T,E
        w, idx = rl.topk(TOPK, -1)
        w = F.softmax(w, -1)
        out = torch.zeros_like(x1)
        xnf = xn
        for j in range(TOPK):
            e_flat = idx[..., j].reshape(-1)                      # N token slots -> expert id
            x_flat = xnf.reshape(-1, H)
            slot = torch.zeros_like(x_flat)
            ue, inv = torch.unique(e_flat, return_inverse=True)
            for k in range(ue.numel()):
                e_id = int(ue[k])
                m = (inv == k).nonzero().squeeze(1)
                xm = x_flat[m]
                g = xm @ self.gup[e_id, :EI].T
                u = xm @ self.gup[e_id, EI:].T
                slot[m] = (F.silu(g) * u) @ self.down[e_id].T     # N,H
            out = out + (slot * w[..., j].reshape(-1, 1)).view(B, T, H)
        sg = torch.sigmoid(xn @ self.se_gate.T)
        seg = F.silu(xn @ self.se_g.T) * (xn @ self.se_u.T)
        out = out + sg * (seg @ self.se_d.T)
        x2 = x1 + out
        return rms(x2, self.out_norm).bfloat16() @ self.lm_head.T


def rope_tables(max_len):
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float32, device="cuda") / ROT))
    ang = torch.arange(max_len, dtype=torch.float32, device="cuda")[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)


class Rows(Dataset):
    def __init__(self, path, tok, max_len, compose):
        self.samples = []
        for line in open(path):
            row = json.loads(line)
            text = compose(row)
            ids = tok(text, add_special_tokens=False)["input_ids"][:max_len]
            if len(ids) > 8:
                self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="gated student checkpoint dir (CausalLM-form config)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mtp-dir", default=PARENT_MTP, help="where mtp.* init tensors live")
    ap.add_argument("--smoke", action="store_true", help="10 steps only (pipeline check)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from safetensors.torch import load_file
    tok = AutoTokenizer.from_pretrained(args.student)
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16,
                                                   device_map="cuda")
    student.eval()
    for p_ in student.parameters():
        p_.requires_grad_(False)

    state = {}
    import glob
    for p in glob.glob(os.path.join(args.mtp_dir, "model*.safetensors")):
        try:
            st = load_file(p)
        except Exception:
            continue
        state.update({k: v for k, v in st.items() if k.startswith("mtp.")})
    assert len(state) == 19, f"expected 19 mtp tensors, got {len(state)}"

    embed_w = student.get_input_embeddings().weight.detach()
    lm_w = student.lm_head.weight.detach()
    head = MTPHead(state, embed_w, lm_w, rope_tables(args.max_len))

    def compose(row):
        msgs = [{"role": "user", "content": row["prompt"]}]
        asst = row["completion"]
        if row.get("thinking") and row.get("thinking_text"):
            asst = f"<think>{row['thinking_text']}</think>{asst}"
        msgs.append({"role": "assistant", "content": asst})
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=False,
                                           tokenize=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)

    ds = Rows(args.data, tok, args.max_len, compose)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def collate(batch):
        L = max(len(b) for b in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        mask = torch.zeros(len(batch), L, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, :len(b)] = torch.tensor(b)
            mask[i, :len(b)] = 1
        return ids, mask

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    step = 0
    for ep in range(args.epochs):
        for ids, mask in dl:
            ids, mask = ids.cuda(), mask.cuda()
            with torch.no_grad():
                hid = student.model(input_ids=ids, output_hidden_states=True).hidden_states[-1]
            logits = head(hid.bfloat16(), ids, mask)
            tgt = torch.roll(ids, -2, 1)                       # x[t+2]
            lm = torch.roll(mask, -2, 1)
            lm[:, -2:] = 0
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]).float(),
                                   tgt.view(-1), reduction="none")
            loss = (loss.view_as(tgt) * lm).sum() / lm.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); step += 1
            print(f"ep{ep} step{step} loss {loss.item():.3f}", flush=True)
            if args.smoke and step >= 10:
                break
        if args.smoke and step >= 10:
            break

    mapping = {
        "mtp.pre_fc_norm_embedding.weight": head.norm_e, "mtp.pre_fc_norm_hidden.weight": head.norm_h,
        "mtp.fc.weight": torch.cat([head.fc_e, head.fc_h], 1),
        "mtp.layers.0.input_layernorm.weight": head.in_ln,
        "mtp.layers.0.post_attention_layernorm.weight": head.post_ln,
        "mtp.layers.0.self_attn.q_proj.weight": head.q, "mtp.layers.0.self_attn.k_proj.weight": head.k,
        "mtp.layers.0.self_attn.v_proj.weight": head.v, "mtp.layers.0.self_attn.o_proj.weight": head.o,
        "mtp.layers.0.self_attn.q_norm.weight": head.qn, "mtp.layers.0.self_attn.k_norm.weight": head.kn,
        "mtp.layers.0.mlp.gate.weight": head.gate,
        "mtp.layers.0.mlp.experts.gate_up_proj": head.gup, "mtp.layers.0.mlp.experts.down_proj": head.down,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": head.se_g,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": head.se_u,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": head.se_d,
        "mtp.layers.0.mlp.shared_expert_gate.weight": head.se_gate,
        "mtp.norm.weight": head.out_norm,
    }
    from safetensors.torch import save_file
    tensors = {k: t.detach().to(torch.bfloat16).cpu().contiguous() for k, t in mapping.items()}
    save_file(tensors, os.path.join(args.out_dir, "model-mtp-trained.safetensors"),
              metadata={"format": "pt"})
    json.dump({"frozen_prefix_digest": "n/a (head only)", "tensors": len(tensors)},
              open(os.path.join(args.out_dir, "MTP_HEAD.json"), "w"))
    print(f"saved 19 mtp tensors -> {args.out_dir}/model-mtp-trained.safetensors")


if __name__ == "__main__":
    sys.exit(main())
