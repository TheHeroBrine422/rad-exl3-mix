#!/usr/bin/env python3
"""mkperm119.py HEADSENS.npz PERM.npy OUTDIR -- w119's stored row orders (RULES.md) and their hout_diag files.
  S.npy  = PERM.npy (today's Fisher-sorted order, headslab perm)
  R16.npy: 16-id runs strided across the 1,940 blocks (block j: runs j, j+1940, ..., j+7*1940)
  N.npy  = identity
  T.npy  : tiers 512 / 4,096 / 32,768 / rest by PERM's rank, each in vocab-id order as R16 inside the tier
OUTDIR/hd-<O>/lm_head.safetensors: hout_diag = fisher_qad_gen[O] (f32), as gpu94.sh writes it for S."""
import os, sys, numpy as np, torch
from safetensors.torch import save_file

HS, PF, OUT = sys.argv[1:4]; V = 248320
F = np.load(HS)["fisher_qad_gen"]; S = np.load(PF).astype(np.int64); assert sorted(S.tolist()) == list(range(V))


def r16(ids):
    s = np.sort(ids); n = len(s); assert n % 128 == 0
    return s.reshape(8, n // 128, 16).transpose(1, 0, 2).reshape(-1)


B = [0, 512, 4096, 32768, V]
O = {"S": S, "R16": r16(np.arange(V)), "N": np.arange(V, dtype=np.int64),
     "T": np.concatenate([r16(S[a:b]) for a, b in zip(B, B[1:])])}
os.makedirs(OUT, exist_ok=True)
for k, o in O.items():
    o = o.astype(np.int64); assert len(np.unique(o)) == V
    np.save(f"{OUT}/{k}.npy", o); os.makedirs(f"{OUT}/hd-{k}", exist_ok=True)
    save_file({"hout_diag": torch.from_numpy(F[o].astype(np.float32))}, f"{OUT}/hd-{k}/lm_head.safetensors")
    blk = o.reshape(-1, 128)
    print(k, "block 0 ids", blk[0][:4], "... runs contiguous:", bool((np.diff(blk[:, :16], axis=1) == 1).all()) if k in ("R16",) else "-",
          "top-512 Fisher share in blocks 0-3:", round(float(F[o[:512]].sum() / F.sum()), 3))
