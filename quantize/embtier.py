#!/usr/bin/env python3
"""embtier.py CONTAINER OFFSET COUNTS.npy PLAN [--write] -- fake-quantise a container's bf16 token table with a
vocabulary-tiered embedding: rows ranked by COUNTS (descending), PLAN = "N1:B1,N2:B2,...,B_rest" (e.g. "30000:3,1":
the 30k most frequent ids at 3 bits, the rest at 1). Each tier is embq.py's l<B>g128h: a 2^B-level Lloyd-Max codebook
(fitted per B on every 16th row of the whole table) under fwht128, a bf16 scale per 128 values found by search;
decoded values are written back as bf16 in place (run on a COPY). The engine then reads exactly what a tiered
embed_tq gather would produce; only the size on disk is not the tiered format's. Prints bits a value (incl. scales)
and the model's bpw for the table (248320 x 5120, 26.9B params)."""
import sys, time
import numpy as np

path, off, cntf, plan = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
write = "--write" in sys.argv[5:]
ROWS, COLS, G, CH, PARAMS = 248320, 5120, 128, 4096, 26895998464
parts = plan.split(",")
tiers = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in parts[:-1]] + [(None, int(parts[-1]))]
cnt = np.load(cntf); order = np.argsort(-cnt, kind="stable")
rowbits = np.empty(ROWS, np.int64); r0 = 0
for n, b in tiers:
    n = ROWS - r0 if n is None else n
    rowbits[order[r0:r0 + n]] = b; r0 += n
assert r0 == ROWS, plan

H128 = np.array([[1.0]])
while H128.shape[0] < 128: H128 = np.block([[H128, H128], [H128, -H128]])
def bf16_to_f32(u): return (u.astype(np.uint32) << 16).view(np.float32)
def f32_to_bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def round_bf16(x): return bf16_to_f32(f32_to_bf16(np.asarray(x, np.float32))).astype(np.float64)
def rot(w): return (w.reshape(w.shape[0], COLS // 128, 128) @ H128 / 128).reshape(w.shape[0], COLS)
def unrot(v): return (v.reshape(v.shape[0], COLS // 128, 128) @ H128).reshape(v.shape[0], COLS)


def fit_levels(x, B):            # Lloyd-Max on amax-normalised group values: 2^B levels in [-1, 1] (embq.py)
    x = np.sort(x)
    L = np.quantile(x, (np.arange(2 ** B) + 0.5) / 2 ** B)
    for _ in range(60):
        idx = np.searchsorted((L[1:] + L[:-1]) / 2, x)
        s = np.bincount(idx, x, 2 ** B); c = np.bincount(idx, None, 2 ** B)
        L = np.where(c > 0, s / np.maximum(c, 1), L)
    return L


def quant_l(w, L):               # embq.py quant_l
    n = w.shape[0]; g = rot(w).reshape(n, COLS // G, G)
    amax = np.abs(g).max(2, keepdims=True); edges = (L[1:] + L[:-1]) / 2
    be = bv = None
    for t in np.linspace(0.6, 1.2, 25):
        s = round_bf16(amax * t); s = np.where(s > 0, s, 1.0)
        v = L[np.searchsorted(edges, g / s)] * s; e = ((v - g) ** 2).sum(2, keepdims=True)
        if be is None: be, bv = e, v
        else: k = e < be; be = np.where(k, e, be); bv = np.where(k, v, bv)
    return unrot(bv.reshape(n, COLS))


mm = np.memmap(path, np.uint16, "r+" if write else "r", offset=off, shape=(ROWS, COLS))
t0 = time.time()
smp = []
for a in range(0, ROWS, CH * 16):
    g = rot(bf16_to_f32(np.asarray(mm[a:a + CH:16])).astype(np.float64)).reshape(-1, G)
    smp.append((g / np.maximum(np.abs(g).max(1, keepdims=True), 1e-30)).ravel())
smp = np.concatenate(smp)
LV = {b: fit_levels(smp, b) for b in sorted(set(rowbits.tolist()))}
se = sq = 0.0
for a in range(0, ROWS, CH):
    w = bf16_to_f32(np.asarray(mm[a:a + CH])).astype(np.float64); d = np.empty_like(w); rb = rowbits[a:a + CH]
    for b in np.unique(rb):
        k = rb == b; d[k] = quant_l(w[k], LV[b])
    se += float(((round_bf16(d) - w) ** 2).sum()); sq += float((w ** 2).sum())
    if write: mm[a:a + CH] = f32_to_bf16(d)
if write: mm.flush()
bits = float(np.mean(rowbits)) + 16 / G
print(f"{plan}: {bits:.4f} bits a value, table {bits * ROWS * COLS / PARAMS:.4f} bpw, relative RMS error "
      f"{np.sqrt(se / sq):.5f}, {time.time() - t0:.0f} s" + (" (written)" if write else ""))
