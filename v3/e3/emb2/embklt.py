#!/usr/bin/env python3
"""embklt.py METHOD R OUT.jsonl -- trellis-coded embedding rows: the shipped menu vs a KLT transform coder (CPU).

Rows: 4,096 random rows of the bf16 token table (seed 5, as #35 / #97), each divided by its RMS first (the per-row
scale), then:
  base     fwht128 per 128 columns, one scale per row (factor 0.9 / 1.05 / 1.2, best per row), mul1 trellis on
           256-value tiles at K = R  (= embtrellis2.py's trellis method; R may be fractional / sub-1)
  kltu     the embedding's KLT (eigenvectors U of the normalized rows' second moment, descending; specan.py), every
           coefficient divided by sqrt(λ_j) (unit variance), uniform K = R on 256-value tiles: KLT without allocation
  kltwf    KLT + bit allocation: 20 bands of 256 coefficients in eigen order, coefficient j divided by sqrt(λ_j) inside
           its band, band b at K_b from the menu {0, 1/8 .. 1, 1.25 .. 6}; the K_b minimise Σ_b Σ_{j∈b} λ_j d(K_b)
           at Σ K_b = 20 R (DP in 1/8-bit units), d(K) = the trellis's measured error per unit variance (embmenu
           K ≥ 0.25 rows, log-linear between; d(0) = 1). K_b = 0: the band is dropped (decoded as 0).
           One scale per row (factor searched as base) on the coded coefficients.
  kltwf8   kltwf decoded through an int8 U (one f32 scale per column): what a stored rotation would give.
Every method's error is measured in the row domain: rel = Σ‖w - ŵ‖² / Σ‖w‖² (primary) and the layer-0 reader metric
rel_M0 = Σ (w-ŵ) M0 (w-ŵ)ᵀ / Σ w M0 wᵀ (secondary; M0 from specan.py). Side bytes (the int8 U; the band table) are
reported, not added: the comparison script charges them. Env SPEC (specan output prefix), E3_CKPT, OMP_NUM_THREADS."""
import json, math, os, sys, time
import numpy as np, torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import trellis as T

METHOD, R, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]
SPEC = os.environ.get("SPEC", "/data/work/w13/w83/out/specan")
CK = os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16")
NROWS = int(os.environ.get("NROWS", 4096))
wm = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
NE = "model.language_model.embed_tokens.weight"
rows = np.sort(np.random.default_rng(5).choice(248320, NROWS, replace=False))
with safe_open(f"{CK}/{wm[NE]}", "pt") as f:
    sl = f.get_slice(NE); W = torch.cat([sl[int(i):int(i) + 1] for i in rows]).double().numpy()
N, C = W.shape
rms = np.sqrt((W ** 2).mean(1, keepdims=True)); Wn = W / rms
M0 = np.load(SPEC + "-M0.npy").astype(np.float64)
cb = np.ascontiguousarray(T.codebook("mul1"), dtype=np.float32)
FAC = [0.9, 1.05, 1.2]
bf = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).bfloat16().double().numpy()


def code_rows(X, Ks):
    """X [N, C] (C = 20 bands x 256), Ks per band; one bf16 scale per row = RMS of the coded part x factor (best)."""
    coded = np.concatenate([np.arange(b * 256, b * 256 + 256) for b, k in enumerate(Ks) if k > 0])
    r = np.sqrt((X[:, coded] ** 2).mean(1, keepdims=True)); r = np.where(r > 0, r, 1.0)
    best_e = best_q = None
    for f in FAC:
        s = bf(r * f); Q = np.zeros_like(X)
        for k in sorted(set(k for k in Ks if k > 0)):
            bs = [b for b, kk in enumerate(Ks) if kk == k]
            cols = np.concatenate([np.arange(b * 256, b * 256 + 256) for b in bs])
            x = (X[:, cols] / s).reshape(-1, 256)
            Q[:, cols] = T.vit(x, cb, k).astype(np.float64).reshape(N, len(cols)) * s
        e = ((Q - X) ** 2).sum(1)
        if best_e is None: best_e, best_q = e, Q
        else: m = e < best_e; best_e = np.where(m, e, best_e); best_q = np.where(m[:, None], Q, best_q)
    return best_q


def dcurve():
    pts = {0.0: 1.0}
    for fn in ("embmenu.jsonl", "embmenu2.jsonl"):
        p = os.path.join(HERE, "data", fn)
        if not os.path.exists(p): p = os.path.join(HERE, "..", "data", fn)
        for l in open(p):
            d = json.loads(l)
            if d["method"] == "trellis" and d.get("group") == 5120: pts[float(d["bits"])] = d["rel"]
    ks = np.array(sorted(pts)); ls = np.log([pts[k] for k in ks])
    return lambda k: float(np.exp(np.interp(k, ks, ls)))


t0 = time.time(); info = {}
if METHOD == "base":
    H = np.array([[1.0]])
    while H.shape[0] < 128: H = np.block([[H, H], [H, -H]])
    H /= math.sqrt(128)
    V = (Wn.reshape(N, C // 128, 128) @ H).reshape(N, C)
    Q = code_rows(V, [R] * 20)
    What = (Q.reshape(N, C // 128, 128) @ H.T).reshape(N, C) * rms
else:
    U = np.load(SPEC + "-embU.npy").astype(np.float64)            # [C, C], columns = eigenvectors, descending
    G = np.load(SPEC.replace("specan", "spec") + ".npz")["Gn"] / 248320
    lam = np.einsum("ij,jk,ki->i", U.T, G, U); lam = np.maximum(lam, 1e-12 * lam.max())
    sd = np.sqrt(lam)
    X = (Wn @ U) / sd
    if METHOD == "kltu":
        Ks = [R] * 20
    else:
        d = dcurve(); menu = [i / 8 for i in range(9)] + [1 + i / 4 for i in range(1, 21)]
        units = int(round(20 * R * 8)); band_l = lam.reshape(20, 256).sum(1)
        # DP over bands: cost[b][u] = min distortion with u eighth-bits spent on bands < b
        INF = float("inf"); best = {0: (0.0, [])}
        for b in range(20):
            nb = {}
            for u, (c, ks) in best.items():
                for k in menu:
                    uu = u + int(round(k * 8))
                    if uu > units: continue
                    cc = c + band_l[b] * d(k)
                    if uu not in nb or cc < nb[uu][0]: nb[uu] = (cc, ks + [k])
            best = nb
        Ks = best[max(best)][1] if units not in best else best[units][1]
    info["Ks"] = Ks
    Q = code_rows(X, Ks)
    Uq = U
    if METHOD == "kltwf8":
        sc = np.abs(U).max(0, keepdims=True) / 127; Uq = np.round(U / sc).clip(-127, 127) * sc
    What = ((Q * sd) @ Uq.T) * rms
    info["side_bytes_U_int8"] = C * C + 4 * C
    info["coded_cols"] = int(256 * sum(1 for k in Ks if k > 0))
E = W - What
rel = float((E ** 2).sum() / (W ** 2).sum())
relM0 = float(np.einsum("ij,jk,ik->", E, M0, E) / np.einsum("ij,jk,ik->", W, M0, W))
rowrel = (E ** 2).sum(1) / (W ** 2).sum(1)
o = {"method": METHOD, "R": R, "rel": rel, "rel_M0": relM0, "rowrel_p90": float(np.percentile(rowrel, 90)),
     "rowrel_max": float(rowrel.max()), "rows": N, "sec": round(time.time() - t0, 1), **info}
print(json.dumps(o), flush=True)
with open(OUT, "a") as fh: fh.write(json.dumps(o) + "\n")
