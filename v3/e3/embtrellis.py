#!/usr/bin/env python3
"""embtrellis.py ROWS OUT.jsonl -- trellis-coded token embedding vs V2's Lloyd tables (design/v3.md §1b.6), CPU.

ROWS random rows of the checkpoint's token table (seeded), quantized with H = I (plain reconstruction error):
  lloyd   V2's embedding: fwht128 per 128 columns, a 2^B-level Lloyd-Max codebook fitted to the amax-normalised
          values, a bf16 scale per (row, 128) searched for least squared error  (B + 0.125 bits)   = embq.py l<B>g128h
  trellis the same rotation and per-(row, 128) scale (RMS-based, factor searched), each row cut into 256-value tiles
          and coded by the mul1 bitshift trellis at K bits (fractional K via viterbi_tiles_pat), so a row decodes
          alone                                                                    (K + 0.125 bits)
One JSON line per (method, bits): rel = Σ (w - ŵ)² / Σ w² over the rows.  Env E3_CKPT, OMP_NUM_THREADS,
EMB_KS (trellis K list), EMB_G (trellis scale group, a multiple of 128 dividing 5120; 16-bit scale = 16/G bits),
EMB_LLOYD=0 (skip the Lloyd rows), EMB_FACTORS (scale = RMS x factor, searched per group; default 0.9,1.05,1.2),
EMB_ROWS_OUT=PREFIX (also save PREFIX-K<k>-G<g>.npz: row ids, row energy Σw², per-row error Σ(w-ŵ)², per-group chosen
factor index -- for tail rows, edge-factor checks and token-frequency reweighting later)."""
import json, math, os, sys, time
import numpy as np, torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trellis as T

CK = os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16")
R, OUT = int(sys.argv[1]), sys.argv[2]
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
name = next(n for n in idx if n.endswith("embed_tokens.weight"))
rows = np.sort(np.random.default_rng(5).choice(248320, R, replace=False))
with safe_open(f"{CK}/{idx[name]}", "pt") as f:
    sl = f.get_slice(name); W = torch.cat([sl[int(i):int(i) + 1] for i in rows]).double().numpy()
C = W.shape[1]
H = np.array([[1.0]])
while H.shape[0] < 128: H = np.block([[H, H], [H, -H]])
H /= math.sqrt(128)                                      # orthonormal
def rot(w): return (w.reshape(len(w), C // 128, 128) @ H).reshape(len(w), C)
def unrot(v): return (v.reshape(len(v), C // 128, 128) @ H.T).reshape(len(v), C)
def bf(x): return torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().float().double().numpy()
ref = (W ** 2).sum()
Vr = rot(W); G = Vr.reshape(R, C // 128, 128)

def emit(o):
    print(json.dumps(o), flush=True)
    with open(OUT, "a") as fh: fh.write(json.dumps(o) + "\n")

for B in ((1, 2, 3, 4, 5) if os.environ.get("EMB_LLOYD", "1") == "1" else ()):
    t0 = time.time(); amax = np.abs(G).max(2, keepdims=True)
    x = np.sort((G / np.where(amax > 0, amax, 1)).ravel())[::97]
    L = np.quantile(x, (np.arange(2 ** B) + 0.5) / 2 ** B)
    for _ in range(60):
        e = (L[1:] + L[:-1]) / 2; k = np.searchsorted(e, x)
        L = np.where(np.bincount(k, None, 2 ** B) > 0, np.bincount(k, x, 2 ** B) / np.maximum(np.bincount(k, None, 2 ** B), 1), L)
    e = (L[1:] + L[:-1]) / 2; best_e = best_v = None
    for t in np.linspace(0.6, 1.2, 25):
        s = bf(amax * t); s = np.where(s > 0, s, 1.0)
        v = L[np.searchsorted(e, G / s)] * s; er = ((v - G) ** 2).sum(2, keepdims=True)
        if best_e is None: best_e, best_v = er, v
        else: m = er < best_e; best_e = np.where(m, er, best_e); best_v = np.where(m, v, best_v)
    rel = ((unrot(best_v.reshape(R, C)) - W) ** 2).sum() / ref
    emit({"method": "lloyd", "bits": B, "bpv": B + 0.125, "rel": rel, "rows": R, "sec": round(time.time() - t0, 1)})

SG = int(os.environ.get("EMB_G", 128)); assert SG % 128 == 0 and C % SG == 0
Gs = Vr.reshape(R, C // SG, SG)
rms = np.sqrt((Gs ** 2).mean(2, keepdims=True)); rms = np.where(rms > 0, rms, 1.0)
for K in [float(k) for k in os.environ.get("EMB_KS", "1,1.5,2,2.5,3,3.5,4,5").split(",")]:
    t0 = time.time(); cb = np.ascontiguousarray(T.codebook("mul1"), dtype=np.float32); best_e = best_v = None
    FAC = [float(v) for v in os.environ.get("EMB_FACTORS", "0.9,1.05,1.2").split(",")]; best_f = None
    for fi, f in enumerate(FAC):
        s = bf(rms * f); x = (Gs / s).reshape(R * C // 256, 256)
        q = T.vit(x, cb, K).astype(np.float64).reshape(R, C // SG, SG) * s
        er = ((q - Gs) ** 2).sum(2, keepdims=True)
        if best_e is None: best_e, best_v, best_f = er, q, np.zeros(er.shape, np.int8)
        else: m = er < best_e; best_e = np.where(m, er, best_e); best_v = np.where(m, q, best_v); best_f = np.where(m, fi, best_f)
    rowerr = ((unrot(best_v.reshape(R, C)) - W) ** 2).sum(1)
    rel = rowerr.sum() / ref
    edge = float(((best_f == 0) | (best_f == len(FAC) - 1)).mean())
    if os.environ.get("EMB_ROWS_OUT"):
        np.savez(f"{os.environ['EMB_ROWS_OUT']}-K{K}-G{SG}.npz", rows=rows, energy=(W ** 2).sum(1), err=rowerr,
                 fac=best_f.reshape(R, -1), factors=np.array(FAC))
    emit({"method": "trellis", "bits": K, "group": SG, "bpv": K + 16 / SG, "rel": rel, "edge_factor_frac": round(edge, 4), "rows": R, "sec": round(time.time() - t0, 1)})
