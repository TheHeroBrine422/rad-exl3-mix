#!/usr/bin/env python3
"""embldlq.py HMETRIC K OUT.jsonl -- embedding rows quantized by LDLQ + trellis against a sensitivity metric (CPU).

Same 4,096 rows as #35 / #97 (seed 5), each divided by its RMS, fwht128 per 128 columns (orthonormal, no signs: the
shipped embedding's basis). Then trellis.ldlq_trellis (EXL3 16x16 tiles, 16-column block feedback, damping 0.01 x mean
diag) against H = Hdᵀ M Hd, M one of
  I    identity (control: no feedback; same tiles)
  M0   layer-0 reader metric (specan.py: diag(g)·ΣWᵀW·diag(g), input projections of layer 0's mixer + MLP)
  F    diagonal Fisher of the residual at layer 0 (Σdy² of layers.0.linear_attn.out_proj's output, 27B hdiag-dy, qad)
  MF   (M0 + F) / 2, each at mean diagonal 1 (the hedge; added after the first M0 / F results, amendment 2)
One bf16 scale per row: RMS × factor, factor ∈ {0.9, 1.05, 1.2} chosen per row by the row's error in the arm's own
metric (rows are independent under LDLQ). Error of the decoded rows (original basis, × RMS) in all three metrics:
rel_X = Σ eᵀXe / Σ wᵀXw.  Env SPEC (specan prefix), FDIAG (hdiag safetensors), E3_CKPT, NROWS."""
import json, math, os, sys, time
import numpy as np, torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import trellis as T
torch.set_grad_enabled(False)

HM, K, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
SPEC = os.environ.get("SPEC", "/data/work/w13/w83/out/specan")
FDIAG = os.environ.get("FDIAG", "/data/work/w13/w83/in/model.language_model.layers.0.linear_attn.out_proj.safetensors")
CK = os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16"); NROWS = int(os.environ.get("NROWS", 4096))
wm = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
NE = "model.language_model.embed_tokens.weight"
rows = np.sort(np.random.default_rng(5).choice(248320, NROWS, replace=False))
with safe_open(f"{CK}/{wm[NE]}", "pt") as f:
    sl = f.get_slice(NE); W = torch.cat([sl[int(i):int(i) + 1] for i in rows]).double()
N, C = W.shape
rms = W.pow(2).mean(1, keepdim=True).sqrt(); Wn = W / rms
Hd = torch.tensor([[1.0]], dtype=torch.float64)
while Hd.shape[0] < 128: Hd = torch.cat([torch.cat([Hd, Hd], 1), torch.cat([Hd, -Hd], 1)], 0)
Hd /= math.sqrt(128)
rot = lambda x: (x.reshape(-1, C // 128, 128) @ Hd).reshape(-1, C)
unrot = lambda x: (x.reshape(-1, C // 128, 128) @ Hd.T).reshape(-1, C)
M0 = torch.from_numpy(np.load(SPEC + "-M0.npy").astype(np.float64))
with safe_open(FDIAG, "pt") as f: fd = f.get_tensor("hout_diag").double()
M0 /= M0.diagonal().mean(); fd /= fd.mean()
BD = torch.block_diag(*[Hd] * (C // 128))
Hr = BD.T @ {"I": lambda: torch.eye(C, dtype=torch.float64), "M0": lambda: M0, "F": lambda: torch.diag(fd),
             "MF": lambda: (M0 + torch.diag(fd)) / 2}[HM]() @ BD   # metric, rotated basis
del BD
cb = np.ascontiguousarray(T.codebook("mul1"), dtype=np.float32)
bf = lambda x: x.float().bfloat16().double()
V = rot(Wn)
t0 = time.time(); best = None
for fac in (0.9, 1.05, 1.2):
    s = bf(V.pow(2).mean(1, keepdim=True).sqrt() * fac)
    Q = T.ldlq_trellis(V / s, Hr, cb, K) * s
    E = Q - V
    e = torch.einsum("ij,jk,ik->i", E, Hr, E)                  # per-row error in the arm's own metric
    if best is None: best_e, best = e, Q
    else: m = e < best_e; best_e = torch.where(m, e, best_e); best = torch.where(m[:, None], Q, best)
What = unrot(best) * rms; Err = W - What
o = {"H": HM, "K": K, "rows": N, "sec": round(time.time() - t0, 1)}
o["rel_I"] = float(Err.pow(2).sum() / W.pow(2).sum())
o["rel_M0"] = float(((Err @ M0) * Err).sum() / ((W @ M0) * W).sum())
o["rel_F"] = float((Err.pow(2) * fd).sum() / (W.pow(2) * fd).sum())
print(json.dumps(o), flush=True)
with open(OUT, "a") as fh: fh.write(json.dumps(o) + "\n")
