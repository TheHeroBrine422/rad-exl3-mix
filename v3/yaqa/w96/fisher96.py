#!/usr/bin/env python3
"""fisher96.py GOLDEN GROUPS.npy OUT.npz [PROCS] -- headsens.py's per-vocab-row head Fisher Σ p(1−p) (golden top-512),
per row group, plus psum Σ p, the row's effective count n = #positions with p > 1e-4, and the group's position count N.

GROUPS.npy: int8 [golden rows], the group of every golden row (−1 = skip); groups 0..G−1. OUT.npz: fisher, psum,
n [G, V] float64 and N [G]. The pass is split into PROCS × 4 row ranges; each range's sums are kept in OUT.parts/ so a
rerun (Thor may go away) skips finished ranges."""
import json, os, sys, numpy as np
from multiprocessing import Pool

G, GF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
m = json.load(open(f"{G}/kld.json")); V = m["n_vocab"]; TOP = m["top"]; R = m["rows"]
grp = np.load(GF, mmap_mode="r"); assert len(grp) == R, (len(grp), R)
NG = int(np.max(grp)) + 1
PD = OUT + ".parts"

def part(rng):
    a, b = rng; f = f"{PD}/{a}-{b}.npz"
    if os.path.exists(f): return f
    tb = np.memmap(f"{G}/top.bin", np.uint8, "r", shape=(R, TOP * 8))
    acc = np.zeros((3, NG * V)); N = np.zeros(NG)
    for c in range(a, b, 20_000):
        e = min(b, c + 20_000); g = np.asarray(grp[c:e]).astype(np.int64); keep = g >= 0
        if not keep.any(): continue
        x = np.asarray(tb[c:e][keep]).view(np.int32).reshape(-1, TOP, 2); g = g[keep]
        N += np.bincount(g, minlength=NG)
        p = np.exp(x[..., 1].view(np.float32).astype(np.float64))
        key = (g[:, None] * V + x[..., 0].astype(np.int64)).ravel(); p = p.ravel()
        acc[0] += np.bincount(key, p * (1 - p), NG * V)
        acc[1] += np.bincount(key, p, NG * V)
        acc[2] += np.bincount(key[p > 1e-4], None, NG * V)
    np.savez(f + ".tmp.npz", acc=acc, N=N); os.replace(f + ".tmp.npz", f); return f

if __name__ == "__main__":
    os.makedirs(PD, exist_ok=True)
    cuts = np.linspace(0, R, NP * 4 + 1).astype(int)
    with Pool(NP) as P: fs = P.map(part, list(zip(cuts[:-1], cuts[1:])), chunksize=1)
    acc = np.zeros((3, NG * V)); N = np.zeros(NG)
    for f in fs: z = np.load(f); acc += z["acc"]; N += z["N"]
    acc = acc.reshape(3, NG, V)
    np.savez(OUT, fisher=acc[0], psum=acc[1], n=acc[2], N=N)
    for i in range(NG):
        print(f"group {i}: N {int(N[i])}, Fisher sum {acc[0, i].sum():.1f}, rows F=0 {int((acc[0, i] == 0).sum())}, "
              f"n<10 {int((acc[2, i] < 10).sum())}, n<100 {int((acc[2, i] < 100).sum())}", flush=True)
