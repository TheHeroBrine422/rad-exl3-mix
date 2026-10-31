#!/usr/bin/env python3
"""headsens.py GOLDEN POOLS OUT.npz [PROCS] -- per-vocab-row sensitivity of the LM head from a bf16 golden.

The head's output-side Fisher diagonal: row v gets sum_t p_t(v) (1 - p_t(v)) (the exact expectation over sampled
labels of (p - onehot)^2, from the golden's top-512 probabilities per position; the tail beyond 512 is dropped).
Accumulated per calib pool (POOLS = pools.json) and for all positions / scored (gen) positions only. Also
sum_t p_t(v) and the count of v as the actual next token. OUT.npz: fisher_<pool>_<all|gen>, psum_..., count_...
(float64 [n_vocab]). The hidden-state-norm weighting of YAQA's sketch is left out (one factor per position)."""
import json, os, sys, numpy as np
from multiprocessing import Pool

G, PJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 6
m = json.load(open(f"{G}/kld.json")); V = m["n_vocab"]; TOP = m["top"]; R = m["rows"]
pools = json.load(open(PJ))["pool"]; names = ["fit", "search", "qad", "verify"]
assert len(pools) == len(m["docs"]), (len(pools), len(m["docs"]))
rowpool = np.zeros(R, np.int8)
for i, d in enumerate(m["docs"]):
    n = d["tokens"] - 1 - d["score_from"]; rowpool[d["row0"]:d["row0"] + n] = names.index(pools[i])
gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool); assert len(gen) == R
# next-token ids per row (row r of doc i predicts token score_from + 1 + k)
nxt = np.zeros(R, np.int64); toks = np.fromfile(f"{G}/tokens.i32", np.int32); off = 0
for d in m["docs"]:
    n = d["tokens"] - 1 - d["score_from"]
    nxt[d["row0"]:d["row0"] + n] = toks[off + d["score_from"] + 1: off + d["tokens"]]; off += d["tokens"]
assert off == len(toks)

def part(rng):
    a, b = rng
    tb = np.memmap(f"{G}/top.bin", np.uint8, "r", shape=(R, TOP * 8))
    acc = np.zeros((2, 2, 4 * V))                     # [fisher|psum][all|gen][pool*V + id]
    for c in range(a, b, 50_000):
        e = min(b, c + 50_000)
        x = np.asarray(tb[c:e]).view(np.int32).reshape(e - c, TOP, 2)
        ids = x[..., 0].astype(np.int64); p = np.exp(x[..., 1].view(np.float32).astype(np.float64))
        key = (rowpool[c:e, None].astype(np.int64) * V + ids).ravel()
        g = np.repeat(gen[c:e], TOP)
        for k, w in enumerate((p * (1 - p), p)):
            w = w.ravel()
            acc[k, 0] += np.bincount(key, w, 4 * V)
            acc[k, 1] += np.bincount(key[g], w[g], 4 * V)
    return acc

if __name__ == "__main__":
    cuts = np.linspace(0, R, NP * 4 + 1).astype(int)
    with Pool(NP) as P: acc = sum(P.map(part, list(zip(cuts[:-1], cuts[1:]))))
    out = {}
    for pi, pn in enumerate(names):
        for mi, mn in enumerate(("all", "gen")):
            out[f"fisher_{pn}_{mn}"] = acc[0, mi, pi * V:(pi + 1) * V]
            out[f"psum_{pn}_{mn}"] = acc[1, mi, pi * V:(pi + 1) * V]
            sel = (rowpool == pi) & (gen if mn == "gen" else True)
            out[f"count_{pn}_{mn}"] = np.bincount(nxt[sel], minlength=V).astype(np.float64)
            out[f"rows_{pn}_{mn}"] = np.array(sel.sum())
    np.savez(OUT, **out)
    for pn in names:
        f = out[f"fisher_{pn}_gen"]; s = np.sort(f)[::-1]; c = np.cumsum(s) / s.sum()
        print(f"{pn:6s} gen rows {int(out[f'rows_{pn}_gen']):>9d}: rows holding 50/90/99/99.9% of Fisher mass "
              + " / ".join(str(int(np.searchsorted(c, q)) + 1) for q in (.5, .9, .99, .999))
              + f", rows with zero mass {int((f == 0).sum())}", flush=True)
