#!/usr/bin/env python3
"""embspec.py OUT.npz -- one pass over the bf16 token embedding and LM head: second-moment matrices (CPU).

Accumulates in f64 over all 248,320 rows (chunks of CH rows):
  Gn  = Σ ê êᵀ, sn = Σ ê     ê = embedding row / its RMS (the per-row scale the coder applies first)
  Ge  = Σ e eᵀ, se = Σ e     raw embedding rows
  Gh  = Σ h hᵀ, sh = Σ h     raw head rows
  Geh = Σ e hᵀ               cross (embedding row v with head row v)
  rms_e, rms_h [V]           per-row RMS
  blk_cv [V]                 per row: coefficient of variation of the 40 per-128-block energies (what one scale per
                             row cannot follow; a 128-Hadamard keeps block energies)
Saved as f64 (≈ 1 GB). Env E3_CKPT, THREADS (default 8), CH (default 8192). Restartable: skips if OUT exists."""
import json, os, sys, time
import numpy as np, torch
from safetensors import safe_open

OUT = sys.argv[1]
if os.path.exists(OUT): sys.exit(f"{OUT} exists")
torch.set_num_threads(int(os.environ.get("THREADS", 8)))
CK = os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16"); CH = int(os.environ.get("CH", 8192))
wm = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
NE, NH = "model.language_model.embed_tokens.weight", "lm_head.weight"
fe, fh = safe_open(f"{CK}/{wm[NE]}", "pt"), safe_open(f"{CK}/{wm[NH]}", "pt")
se_, sh_ = fe.get_slice(NE), fh.get_slice(NH)
V, C = se_.get_shape(); assert tuple(sh_.get_shape()) == (V, C)
Z = lambda *s: torch.zeros(*s, dtype=torch.float64)
Gn, Ge, Gh, Geh = Z(C, C), Z(C, C), Z(C, C), Z(C, C)
sn, se, sh = Z(C), Z(C), Z(C)
rms_e, rms_h, blk_cv = np.zeros(V), np.zeros(V), np.zeros(V)
t0 = time.time()
for r0 in range(0, V, CH):
    e = se_[r0:r0 + CH].double(); h = sh_[r0:r0 + CH].double(); n = e.shape[0]
    re = e.pow(2).mean(1).sqrt(); rh = h.pow(2).mean(1).sqrt()
    rms_e[r0:r0 + n] = re.numpy(); rms_h[r0:r0 + n] = rh.numpy()
    en = e / re.clamp(min=1e-30)[:, None]
    be = en.view(n, C // 128, 128).pow(2).sum(2); blk_cv[r0:r0 + n] = (be.std(1) / be.mean(1)).numpy()
    Gn += en.T @ en; Ge += e.T @ e; Gh += h.T @ h; Geh += e.T @ h
    sn += en.sum(0); se += e.sum(0); sh += h.sum(0)
    print(f"{r0 + n}/{V} {time.time() - t0:.0f}s", flush=True)
np.savez(OUT + ".part.npz", Gn=Gn.numpy(), Ge=Ge.numpy(), Gh=Gh.numpy(), Geh=Geh.numpy(), sn=sn.numpy(),
         se=se.numpy(), sh=sh.numpy(), rms_e=rms_e, rms_h=rms_h, blk_cv=blk_cv, V=V)
os.replace(OUT + ".part.npz", OUT)
print("done", time.time() - t0)
