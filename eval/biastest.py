#!/usr/bin/env python3
"""biastest.py SCORES_DIR -- how much do V2's fit-slice numbers flatter the mixes picked on them? (no new evals)

turboderp's references were never tuned on our slices, so mix / reference is a slice-neutral measure. For every mix
scored on a fit-pool slice (f2- golden-fit2, f3- golden-fit3, a- golden-calA) AND on ver1 (w-, verify pool):
  r_fit = KLD(mix) / KLD(ref at the nearest bpw) on the fit slice, r_ver = the same on ver1,
  bias = r_fit / r_ver  (< 1: the mix looks better on the slice it was selected / fitted on).
The w- mixes were queued on ver1 BECAUSE they measured well on the fit slices, so this is the winner's-curse plus
fit-in-sample bias of V2's pipeline. Per-document bootstrap over both slices for the median bias's interval.
Refs: sc (self-calibrated) family, nearest bpw; a mix's bpw is the last -X.XX of its name."""
import glob, json, os, re, sys
import numpy as np
D = sys.argv[1]; W = os.environ.get("RXM_WORK", "/root")
G = {"f2": "golden-fit2", "f3": "golden-fit3", "a": "golden-calA", "w": "golden-ver1"}
meta = {}
def docsums(tag, name):
    """per-document (sum of KLD over gen rows, count) for score file tag-name on its golden"""
    g = G[tag]
    if g not in meta:
        m = json.load(open(f"{W}/{g}/kld.json")); gen = np.fromfile(f"{W}/{g}/gen.u8", np.uint8).astype(bool)
        meta[g] = (m, gen)
    m, gen = meta[g]; f = f"{D}/{tag}-{name}.f32"
    if not os.path.exists(f): return None
    x = np.fromfile(f, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    if len(x) != m["rows"]: return None
    s, n = [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); mk = gen[sl]
        s.append(x[sl][mk].sum()); n.append(mk.sum())
    return np.array(s), np.array(n)
bpw = lambda n: float(re.findall(r"(\d+\.\d+)", n)[-1])
REFS = {}
for tag in G:
    REFS[tag] = {bpw(os.path.basename(f)[len(tag) + 5:-4]): os.path.basename(f)[len(tag) + 1:-4]
                 for f in glob.glob(f"{D}/{tag}-ref-sc*.f32")}
def ref_for(tag, b):
    if not REFS[tag]: return None
    return REFS[tag][min(REFS[tag], key=lambda r: abs(r - b))]
rows = []; rng = np.random.default_rng(0)
for f in sorted(glob.glob(f"{D}/w-*.f32")):
    name = os.path.basename(f)[2:-4]
    if name.startswith("ref-") or not re.search(r"\d+\.\d+$", name): continue
    b = bpw(name)
    for tag in ("f3", "f2", "a"):
        mf = docsums(tag, name)
        if mf is None: continue
        rf, rw = ref_for(tag, b), ref_for("w", b)
        if rf is None or rw is None or abs(bpw(rf) - bpw(rw)) > 1e-9: continue
        RF, MW, RW = docsums(tag, rf), docsums("w", name), docsums("w", rw)
        if RF is None or MW is None or RW is None: continue
        def bias(i, j):
            r_fit = mf[0][i].sum() / RF[0][i].sum(); r_ver = MW[0][j].sum() / RW[0][j].sum()
            return r_fit, r_ver
        r_fit, r_ver = bias(slice(None), slice(None))
        bs = [np.log(np.divide(*bias(rng.integers(0, len(mf[0]), len(mf[0])), rng.integers(0, len(MW[0]), len(MW[0])))))
              for _ in range(300)]
        rows.append({"mix": name, "slice": tag, "bpw": b, "ref": rf, "r_fit": r_fit, "r_ver": r_ver,
                     "bias": r_fit / r_ver, "log_se": float(np.std(bs))})
        break                       # one fit slice per mix (f3 first: the stream's main fit slice)
for r in rows: print(json.dumps(r))
if rows:
    lb = np.array([np.log(r["bias"]) for r in rows]); se = np.array([r["log_se"] for r in rows])
    w = 1 / se ** 2; m = (lb * w).sum() / w.sum(); mse = 1 / np.sqrt(w.sum())
    print(f"\n{len(rows)} mixes: bias (r_fit / r_ver) median {np.exp(np.median(lb)):.3f}, inverse-variance mean "
          f"{np.exp(m):.3f} [{np.exp(m - 2 * mse):.3f}, {np.exp(m + 2 * mse):.3f}]; share < 1: {(lb < 0).mean():.0%}")
    for lo, hi in ((0, 2.5), (2.5, 4.5), (4.5, 99)):
        s = [x for x, r in zip(lb, rows) if lo <= r["bpw"] < hi]
        if s: print(f"  bpw {lo}-{hi}: n={len(s)} median bias {np.exp(np.median(s)):.3f}")
