#!/usr/bin/env python3
"""summ.py DIR[,DIR..] [--need-seeds] [--maxseed 4] -- w10-hout-attrib (ranked.md B3): where the diagonal-H_out gain
comes from, by family / depth third / K, on the 0.8B. Files: evalq.py --out score files <tag>.f32, looked up in the
DIRs in order; tags k<K><arm><sfx>, sfx "" (seed 0) or _s<n>. Arms: base; edy (full Σdy² diagonal); h-<sub> (the
diagonal on one subset only: gdn_in, attn_in, mlp_in, writers, d0 / d1 / d2 = layers 0-7 / 8-15 / 16-23).

Per arm and K, seed-paired with base: l_s = ln(KL_arm,s / KL_base,s); gain g = -mean_s l_s, SE = SD(l_s)/sqrt(n)
(n >= 2); share = g / G with G the full arm's gain at that K (delta-method SE). Decisions as README's draft pass rule.
--need-seeds: print "K arm seed" lines for the arms whose rule needs another seed (boundary within one SE), up to
--maxseed seeds; nothing else."""
import json, math, os, sys
import numpy as np

a = sys.argv[1:]
need = "--need-seeds" in a
maxseed = int(a[a.index("--maxseed") + 1]) if "--maxseed" in a else 4
minseed = int(os.environ.get("MINSEED", 2))
dirs = a[0].split(",")
SUBS = ["gdn_in", "attn_in", "mlp_in", "writers", "d0", "d1", "d2"]
FAMS, DEPTH = SUBS[:4], SUBS[4:]
SHARE_MIN, KHI_MIN = 0.15, 0.03

def f32(tag):
    for d in dirs:
        if os.path.exists(f"{d}/{tag}.f32"): return f"{d}/{tag}.f32"
_kl = {}
def kl(tag):
    if tag not in _kl:
        f = f32(tag)
        if f is None: _kl[tag] = None
        else:
            r = np.fromfile(f, np.float32).reshape(-1, 4); g = r[:, 2] > 0; _kl[tag] = float(r[g, 0].astype(np.float64).mean())
    return _kl[tag]
sfx = lambda s: "" if s == 0 else f"_s{s}"

def gain(K, arm, seeds=range(8)):
    """seed-paired gain vs base: dict(n, g, se, logs) or None"""
    ls = []
    for s in seeds:
        x, b = kl(f"k{K}{arm}{sfx(s)}"), kl(f"k{K}base{sfx(s)}")
        if x is not None and b is not None: ls.append((s, math.log(x / b)))
    if not ls: return None
    v = np.array([l for _, l in ls]); n = len(v)
    return {"n": n, "seeds": [s for s, _ in ls], "g": float(-v.mean()),
            "se": float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else None, "logs": [round(float(x), 5) for x in v]}

out, todo = {"subsets": {}, "khi": {}, "full": {}}, []
for K in (3, 2):
    F = gain(K, "edy"); out["full"][K] = F
    for sub in SUBS:
        r = gain(K, f"h-{sub}")
        if r is None or F is None: continue
        sh = r["g"] / F["g"]
        se = None if r["se"] is None else math.sqrt(r["se"] ** 2 + (sh * (F["se"] or 0)) ** 2) / abs(F["g"])
        r.update(share=sh, share_se=se); out["subsets"].setdefault(sub, {})[K] = r
        if r["n"] < maxseed and (r["n"] < minseed or se is None or abs(sh - SHARE_MIN) < se):
            todo.append((K, f"h-{sub}", r["n"]))
    for sub in SUBS:           # arms not started yet: their first seeds
        if sub not in out["subsets"] or K not in out["subsets"][sub]: todo.append((K, f"h-{sub}", 0))
for K in (4, 5):
    r = gain(K, "edy"); out["khi"][K] = r
    if r is None: todo.append((K, "edy", 0))
    elif r["n"] < maxseed and (r["n"] < minseed or r["se"] is None or abs(r["g"] - KHI_MIN) < r["se"]):
        todo.append((K, "edy", r["n"]))

if need:
    for K, arm, n in todo: print(K, arm, n)            # n = seeds done = the next seed offset
    sys.exit(0)

# ---- verdicts (README draft pass rule)
def harmful(r): return r["se"] is not None and r["g"] < 0 and r["g"] / max(r["se"], 1e-12) <= -2
V = {}
for sub, byK in out["subsets"].items():
    if 3 not in byK or 2 not in byK: V[sub] = "incomplete"; continue
    if any(harmful(byK[K]) for K in (3, 2)): V[sub] = "one-sided (harmful)"
    elif all(byK[K]["share"] < SHARE_MIN for K in (3, 2)): V[sub] = "one-sided (minor: share < 0.15 at K3 and K2)"
    else: V[sub] = "two-sided"
for K, r in out["khi"].items():
    if r is None: V[f"K{K}"] = "incomplete"
    else: V[f"K{K}"] = "skip two-sided at this K (gain < 3%)" if r["g"] < KHI_MIN else "keep two-sided"
add = {}
for K in (3, 2):
    F = out["full"].get(K)
    if not F: continue
    for nm, grp in (("families", FAMS), ("depth", DEPTH)):
        gs = [out["subsets"][s][K]["g"] for s in grp if s in out["subsets"] and K in out["subsets"][s]]
        if len(gs) == len(grp): add[f"K{K} {nm}"] = sum(gs) / F["g"]
out["verdicts"] = V; out["additivity"] = add; out["pending"] = todo

print(f"{'arm':10s} {'K':>2s} {'n':>2s} {'gain':>8s} {'se':>7s} {'share':>6s} {'±':>5s}")
for K in (3, 2):
    F = out["full"].get(K)
    if F: print(f"{'full edy':10s} {K:2d} {F['n']:2d} {F['g']:8.4f} {F['se'] or 0:7.4f} {1:6.2f}")
    for sub in SUBS:
        r = out["subsets"].get(sub, {}).get(K)
        if r: print(f"{sub:10s} {K:2d} {r['n']:2d} {r['g']:8.4f} {r['se'] or 0:7.4f} {r['share']:6.2f} {r['share_se'] or 0:5.2f}")
for K, r in out["khi"].items():
    if r: print(f"{'full edy':10s} {K:2d} {r['n']:2d} {r['g']:8.4f} {r['se'] or 0:7.4f}")
for k, v in add.items(): print(f"additivity {k}: sum of subset gains / full gain = {v:.2f}")
for k, v in V.items(): print(f"VERDICT {k}: {v}")
if todo: print(f"pending (need seeds / not run): {todo}")
print(json.dumps(out))
