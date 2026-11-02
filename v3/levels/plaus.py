#!/usr/bin/env python3
"""plaus.py ROOT [OUT.json] -- per-level plausibility check of a ladder (missed-ideas review #9, 2026-11-02).

ROOT holds finished levels K<k>/proxy.jsonl (one codebook per root: the i8 ladder, or ROOT/mul1). For every finished
level and tensor, the proxy error is compared with the log-linear interpolation of the nearest finished levels below
and above (same tensor). Flags:
  off-curve  ratio outside 1 +- tol, tol = 3% when the neighbours are <= 0.5 K apart, 5% when further
             (old ladder K2-4.5: p1-p99 0.982-1.013 at quarter-step neighbours)
  order      error not between its neighbours' (a higher K must not be worse), or at an edge level worse than the
             next level inward
  bad        missing, non-finite or non-positive proxy
Levels below 1 (sub-1) get tolerances x2 (a new regime, no reference data). Prints a summary line per level with
flags and writes the full list to OUT.json. Exit code 1 when anything is flagged. Read-only."""
import glob, json, math, os, re, sys

root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else None
lv = {}
for d in glob.glob(f"{root}/K*"):
    m = re.fullmatch(r"K([0-9.]+)", os.path.basename(d))
    f = f"{d}/proxy.jsonl"
    if not m or not os.path.exists(f): continue
    if not os.environ.get("PLAUS_NOSHARDS") and (not os.path.exists(f"{d}/model.safetensors.index.json")
                                                 or len(glob.glob(f"{d}/q-L*.safetensors")) != 64): continue   # finished levels only
    rows = {}
    for l in open(f):
        r = json.loads(l); rows[r["key"]] = r.get("proxy_err")       # resumed layers repeat rows: keep the last
    lv[float(m.group(1))] = rows
Ks = sorted(lv)
flags = []
for i, K in enumerate(Ks):
    lo = Ks[i - 1] if i > 0 else None; hi = Ks[i + 1] if i + 1 < len(Ks) else None
    for key, e in lv[K].items():
        if e is None or not math.isfinite(e) or e <= 0:
            flags.append({"K": K, "key": key, "kind": "bad", "proxy": e}); continue
        e0 = lv[lo].get(key) if lo is not None else None; e1 = lv[hi].get(key) if hi is not None else None
        sub1 = 2 if K < 1 else 1
        if e0 and e1 and e0 > 0 and e1 > 0:
            t = (K - lo) / (hi - lo)
            interp = math.exp(math.log(e0) + (math.log(e1) - math.log(e0)) * t)
            tol = (0.03 if hi - lo <= 0.5 + 1e-9 else 0.05) * sub1
            r = e / interp
            if abs(r - 1) > tol:
                flags.append({"K": K, "key": key, "kind": "off-curve", "ratio": round(r, 4), "neighbours": [lo, hi]})
            if not (e1 <= e * (1 + 0.005) and e <= e0 * (1 + 0.005)):
                flags.append({"K": K, "key": key, "kind": "order", "proxy": e, "lo": e0, "hi": e1})
        elif e0 and e0 > 0 and e > e0 * 1.005:
            flags.append({"K": K, "key": key, "kind": "order", "proxy": e, "lo": e0})
        elif e1 and e1 > 0 and e < e1 / 1.005:
            flags.append({"K": K, "key": key, "kind": "order", "proxy": e, "hi": e1})
by = {}
for f in flags: by.setdefault(f["K"], []).append(f)
print(f"plaus {root}: {len(Ks)} finished levels {Ks}; {len(flags)} flags")
for K in Ks:
    fl = by.get(K, [])
    if fl:
        kinds = {}
        for f in fl: kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
        worst = max((f for f in fl if "ratio" in f), key=lambda f: abs(f["ratio"] - 1), default=None)
        print(f"  K{K:g}: {kinds}" + (f"; worst {worst['key']} x{worst['ratio']}" if worst else ""))
if out: json.dump(flags, open(out, "w"), indent=1)
sys.exit(1 if flags else 0)
