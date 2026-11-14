#!/usr/bin/env python3
"""rep17703.py DIR REGISTRY [OUT.json] -- RULES-17703 stage B readout (w151, #177-03). DIR = a copy of ds8
/data/q27/mad/spotB's small outputs (docsums.json, check-K.json, proxy/{M,I}-K.jsonl). Per level K: R = mean KLD(M) /
mean KLD(I) per part (x, c) and pooled (geometric mean of the two part ratios), 95 % CIs from 2,000 paired document
bootstraps stratified by part; sanity (M within x1.5 of I, top-1 within 2 points); label (≈ / costs / better); the stop
rule; proxy MSE ratio M / I (geometric mean over the trunk tensors, and per family)."""
import json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "search", "v3"))
import slices as SL  # noqa: E402

D, REG = sys.argv[1:3]; OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(D, "rep17703.json")
S = json.load(open(f"{D}/docsums.json")); parts = np.array(SL.Registry(REG).parts_of("s173e")); P = sorted(set(parts.tolist()))
grp = [np.where(parts == p)[0] for p in P]
rng = np.random.default_rng(17703); draws = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grp]) for _ in range(2000)]


def fam(key):
    k = key.split(".")
    return ".".join(k[k.index("layers") + 2:]) if "layers" in k else key


def top1(e):
    v = (e.get("all") or {}).get("top1_agreement")       # radiance --kld-out: all.top1_agreement (fraction)
    return None if v is None else 100.0 * float(v)


out = {"slice": "golden-s173e", "parts": {p: int((parts == p).sum()) for p in P}, "levels": {}}
for K in ("2.5", "4", "6"):
    a, b = S.get(f"I-{K}"), S.get(f"M-{K}")
    r = {"check": json.load(open(f"{D}/check-{K}.json")).get("pass") if os.path.exists(f"{D}/check-{K}.json") else None}
    if a and b:
        A = np.array([d[0] for d in a["docs"]]); B = np.array([d[0] for d in b["docs"]]); N = np.array([d[1] for d in a["docs"]])
        assert len(A) == len(B) == len(parts) and np.array_equal(N, np.array([d[1] for d in b["docs"]]))
        def pr(ix): return {p: float(B[ix][parts[ix] == p].sum() / A[ix][parts[ix] == p].sum()) for p in P}
        def pooled(ix): return math.exp(np.mean([math.log(v) for v in pr(ix).values()]))
        full = np.arange(len(A)); bs = [pr(ix) for ix in draws]; bp = [pooled(ix) for ix in draws]
        r.update({"kld_I": a["kld"], "kld_M": b["kld"], "positions": a["positions"],
                  "per_part": {p: {"R": pr(full)[p], "ci95": [float(np.percentile([x[p] for x in bs], 2.5)), float(np.percentile([x[p] for x in bs], 97.5))],
                                   "kld_I": float(A[parts == p].sum() / N[parts == p].sum()), "kld_M": float(B[parts == p].sum() / N[parts == p].sum())} for p in P},
                  "R": pooled(full), "ci95": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))],
                  "top1_I": top1(a), "top1_M": top1(b)})
        sane = 1 / 1.5 <= b["kld"] / a["kld"] <= 1.5 and (r["top1_I"] is None or r["top1_M"] is None or abs(r["top1_M"] - r["top1_I"]) <= 2)
        lo, hi = r["ci95"]
        r["sanity"] = bool(sane)
        r["label"] = ("better" if hi < 1 else "costs" if lo > 1 and r["R"] > 1.03 else "≈") if sane else "not measured"
        r["off"] = (not sane) or r["R"] > 1.08 or lo > 1.05 or r["check"] is False
    pm, pi = f"{D}/proxy/M-{K}.jsonl", f"{D}/proxy/I-{K}.jsonl"
    if os.path.exists(pm) and os.path.exists(pi):
        em = {j["key"]: j["proxy_err"] for j in map(json.loads, open(pm))}; ei = {j["key"]: j["proxy_err"] for j in map(json.loads, open(pi))}
        ks = sorted(set(em) & set(ei)); lr = np.array([math.log(em[k] / ei[k]) for k in ks]); bf = defaultdict(list)
        for k, v in zip(ks, lr): bf[fam(k)].append(v)
        r["proxy"] = {"tensors": len(ks), "geo": float(math.exp(lr.mean())), "p10_p90": [float(math.exp(np.percentile(lr, 10))), float(math.exp(np.percentile(lr, 90)))],
                      "by_family": {f: round(math.exp(np.mean(v)), 4) for f, v in sorted(bf.items())}}
        r["proxy_off"] = r["proxy"]["geo"] > 1.03
    out["levels"][K] = r
ks = [k for k in out["levels"] if "kld_I" in out["levels"][k]]
if len(ks) >= 2:                       # the i8 ladder's local slope (per bit) from the twins, for a Δbpw equivalent
    kk = np.array([float(k) for k in ks]); kl = np.log([out["levels"][k]["kld_I"] for k in ks]); s = -np.polyfit(kk, kl, 1)[0]
    out["slope_per_bit"] = float(s)
    for k in ks: out["levels"][k]["dbpw"] = math.log(out["levels"][k]["R"]) / s
out["stop"] = any(v.get("off") or v.get("proxy_off") for v in out["levels"].values())
json.dump(out, open(OUT, "w"), indent=1)
for K, r in out["levels"].items():
    if "R" in r:
        pp = "  ".join(f"{p} {v['R']:.4f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" for p, v in r["per_part"].items())
        print(f"K{K}: KLD i8 {r['kld_I']:.5f} mad {r['kld_M']:.5f}  pooled R {r['R']:.4f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  {pp}  "
              f"top-1 {r['top1_I']} / {r['top1_M']}  dbpw {r.get('dbpw', float('nan')):+.4f}  {r['label']}  check {r['check']}")
    if "proxy" in r: print(f"     proxy M / I geo {r['proxy']['geo']:.4f} over {r['proxy']['tensors']} tensors (p10-p90 {r['proxy']['p10_p90'][0]:.3f}-{r['proxy']['p10_p90'][1]:.3f})")
print("STOP (off)" if out["stop"] else "continue to stage C")
