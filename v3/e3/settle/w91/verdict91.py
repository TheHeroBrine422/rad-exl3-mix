#!/usr/bin/env python3
"""verdict91.py A -- ../w91.md item 1's rule over w91.sh's scores (A/b27/<pair>-{C,A}-{x,c}.json.rows): per pair the
control C -> arm A ratio on golden-search27b's two parts and pooled (w84's pair84.py: ratio of sums over scored
positions, stratified document bootstrap); R = geo over p0, p1 of the pooled r. POSITIVE iff R <= 0.995 and the geo of
r_x and of r_c are both < 1; else DROP. Also: G2 decode gate (p0: control mix -> C, pooled within 1 +- 0.005) and the
secondary depth arm D (report only). Writes A/verdict91.json."""
import json, math, os, sys
A = sys.argv[1]
sys.path.insert(0, os.path.join(A, "code/v3/e3/settle/w84"))
from pair84 import pair, fmt
GX, GC = f"{A}/golden-search27b-x", f"{A}/golden-search27b-c"
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
rows = lambda t, part: f"{A}/b27/{t}-{part}.json.rows"
have = lambda t: all(os.path.exists(rows(t, p)) for p in "xc")


def three(a, b):
    px = pair([(GX, rows(a, "x"), rows(b, "x"))]); pc = pair([(GC, rows(a, "c"), rows(b, "c"))])
    pp = pair([(GX, rows(a, "x"), rows(b, "x")), (GC, rows(a, "c"), rows(b, "c"))])
    return {"pooled": pp, "xcorp": px, "calib": pc}


out = {}
if have("base") and have("p0-C"):
    g2 = three("base", "p0-C"); ok = bool(abs(g2["pooled"]["r"] - 1) <= 0.005)
    out["G2"] = {**g2, "pass": ok}
    print(f"G2 decode (control mix -> dense control C, p0): {fmt(g2['pooled'])}  -> {'PASS' if ok else 'FAIL'}")
pairs = [p for p in ("p0", "p1") if have(f"{p}-C") and have(f"{p}-A")]
for p in pairs + (["D"] if have("D-C") and have("D-A") else []):
    r = three(f"{p}-C", f"{p}-A"); out[p] = r
    print(f"{p}: C -> A\n  pooled  {fmt(r['pooled'])}\n  xcorp   {fmt(r['xcorp'])}\n  calib   {fmt(r['calib'])}")
if len(pairs) == 2:
    R = geo([out[p]["pooled"]["r"] for p in pairs]); gx = geo([out[p]["xcorp"]["r"] for p in pairs]); gc = geo([out[p]["calib"]["r"] for p in pairs])
    zs = sum(out[p]["pooled"]["diff"] for p in pairs) / math.sqrt(sum(out[p]["pooled"]["se"] ** 2 for p in pairs))
    v = "POSITIVE (theme stays open)" if R <= 0.995 and gx < 1 and gc < 1 else "DROP"
    if not out.get("G2", {}).get("pass", False): v += " [G2 decode gate not passed: not read]"
    out["rule"] = {"R": R, "geo_x": gx, "geo_c": gc, "pooled_z": zs, "verdict": v}
    if "D" in out:
        d = out["D"]; dv = bool(d["pooled"]["r"] <= 0.995 and d["xcorp"]["r"] < 1 and d["calib"]["r"] < 1)
        out["rule"]["D"] = "open at depth (one seed)" if dv else "not open at depth"
    json.dump(out, open(f"{A}/verdict91.json", "w"), indent=1)
    print(f"W91 ITEM 1: R {R:.4f} (geo x {gx:.4f}, c {gc:.4f}, pooled z {zs:+.2f}) -> {v}" + (f"; D: {out['rule']['D']}" if "D" in out else ""))
else:
    json.dump(out, open(f"{A}/verdict91.json", "w"), indent=1)
    print(f"W91 ITEM 1: incomplete ({pairs})")
