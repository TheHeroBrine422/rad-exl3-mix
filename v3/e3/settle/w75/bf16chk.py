#!/usr/bin/env python3
"""bf16chk.py CKPT PERMCKPT GOLDEN [NSEQ=8] [LEN=1024] -- w75 #106-01's bf16 gate (../w75.md): the permuted checkpoint
(perm06.py make) computes the same function as the original. Three bf16 runs on the same tokens (the first LEN tokens
of the first NSEQ GOLDEN documents with >= LEN tokens), one sequence at a time:
  A  the original;  B  the permuted checkpoint;
  C  the original with every attention layer's o_proj computed as linear(x[..., cols], W[:, cols]): mathematically
     identical, only o_proj's summation order changes (the float-reassociation scale of the permutation).
Reported: A == B bit for bit; max |B - A|, max |C - A|, B == C; mean KL(A || B) (f32) and top-1 agreement.
PASS: A == B bitwise, or (max |B - A| <= max |C - A|, mean KL <= 1e-6, top-1 agreement >= 0.9999). Last line:
"BF16 GATE PASS|FAIL {json}"."""
import json, sys, os
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b")
from common import load_model

ck, pc, gd = sys.argv[1:4]
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
LN = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
m = json.load(open(f"{gd}/kld.json")); toks = np.memmap(f"{gd}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
seqs = [torch.from_numpy(np.array(toks[off[i]:off[i] + LN], np.int64)) for i, d in enumerate(m["docs"]) if d["tokens"] >= LN][:NS]
assert len(seqs) == NS, len(seqs)
P = json.load(open(f"{pc}/perm06.json"))["layers"]

cap = float(os.environ.get("VRAM_CAP_GB", 7.2)) * 2 ** 30                 # the 4070's shared-card cap
torch.cuda.set_per_process_memory_fraction(min(1.0, cap / torch.cuda.get_device_properties(0).total_memory))
A, _, _ = load_model(ck); B, _, _ = load_model(pc)
mods = {int(l): A.get_decoder().layers[int(l)].self_attn.o_proj for l in P}
cols = {l: torch.tensor(P[str(l)]["cols"], device="cuda") for l in mods}
Wc = {l: mods[l].weight.data[:, cols[l]].contiguous() for l in mods}


def permuted_oproj(on):
    for l, mod in mods.items():
        if on: mod.forward = (lambda x, l=l: F.linear(x.index_select(-1, cols[l]), Wc[l]))
        elif "forward" in mod.__dict__: del mod.forward


st = dict(seqs=NS, len=LN, ab_bitwise=True, bc_bitwise=True, max_ab=0.0, max_ac=0.0, max_bc=0.0, kl_sum=0.0, agree=0, n=0)
with torch.no_grad():
    for s in seqs:
        x = s[None].cuda()
        la = A(x).logits[0]; lb = B(x).logits[0]
        permuted_oproj(True); lc = A(x).logits[0]; permuted_oproj(False)
        st["ab_bitwise"] &= bool(torch.equal(la, lb)); st["bc_bitwise"] &= bool(torch.equal(lb, lc))
        for i in range(0, la.shape[0], 128):                     # f32 math in row chunks (VRAM cap)
            a_, b_, c_ = la[i:i + 128].float(), lb[i:i + 128].float(), lc[i:i + 128].float()
            st["max_ab"] = max(st["max_ab"], float((a_ - b_).abs().max()))
            st["max_ac"] = max(st["max_ac"], float((a_ - c_).abs().max()))
            st["max_bc"] = max(st["max_bc"], float((b_ - c_).abs().max()))
            pa, pb = F.log_softmax(a_, -1), F.log_softmax(b_, -1)
            st["kl_sum"] += float((pa.exp() * (pa - pb)).sum(-1).clamp(min=0).sum())
            st["agree"] += int((a_.argmax(-1) == b_.argmax(-1)).sum()); st["n"] += a_.shape[0]
            del a_, b_, c_, pa, pb
        del la, lb, lc
st["kl_mean"] = st.pop("kl_sum") / st["n"]; st["top1_agree"] = st.pop("agree") / st["n"]
ok = st["ab_bitwise"] or (st["max_ab"] <= st["max_ac"] and st["kl_mean"] <= 1e-6 and st["top1_agree"] >= 0.9999)
print(f"BF16 GATE {'PASS' if ok else 'FAIL'} {json.dumps(st)}")
sys.exit(0 if ok else 1)
