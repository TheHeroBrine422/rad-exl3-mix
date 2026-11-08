#!/usr/bin/env python3
"""verdict90.py O P84 GX GC 34|99 -- ../HEADWM.md amendment 1 on headwm90.sh's rows (O/<arm>-{x,c}.json.rows):
pooled pairs over golden-search27b's two parts (P84/pair84.py: position-weighted, document bootstrap within each
part) and the part ratios. 34: validity (v4 -> h4-bf16) and #34-02 (u5 -> s5-3, u6 -> s6-3); 99: v8 -> h8-bf16 (report).
Writes O/verdict<which>.json."""
import json, math, os, sys

O, P84, GX, GC, which = sys.argv[1:6]
sys.path.insert(0, P84)
import pair84


def rows(arm, p): return f"{O}/{arm}-{p}.json.rows"


def pr(a, b):
    if not all(os.path.exists(rows(x, p)) for x in (a, b) for p in "xc"): return None
    pooled = pair84.pair([(GX, rows(a, "x"), rows(b, "x")), (GC, rows(a, "c"), rows(b, "c"))])
    px = pair84.pair([(GX, rows(a, "x"), rows(b, "x"))]); pc = pair84.pair([(GC, rows(a, "c"), rows(b, "c"))])
    return {"pooled": pooled, "x": px, "c": pc}


out = {}
pairs = [("v4", "h4-bf16"), ("h4-u5", "h4-s5-3"), ("h4-u6", "h4-s6-3"), ("h4-bf16", "h4-s6-3")] if which == "34" else [("v8", "h8-bf16")]
for a, b in pairs:
    p = pr(a, b); out[f"{a} -> {b}"] = p
    if p is None: print(f"pair {a} -> {b}: missing rows"); continue
    print(f"pair {a} -> {b} pooled: {pair84.fmt(p['pooled'])}")
    print(f"   xcorp: r {p['x']['r']:.4f} [{p['x']['ci95'][0]:.4f}, {p['x']['ci95'][1]:.4f}] z {p['x']['z']:+.2f} | "
          f"calib: r {p['c']['r']:.4f} [{p['c']['ci95'][0]:.4f}, {p['c']['ci95'][1]:.4f}] z {p['c']['z']:+.2f}")
if which == "34":
    v = out.get("v4 -> h4-bf16")
    valid = v is not None and all(math.isfinite(v[k]["kld_b"]) for k in ("x", "c")) and v["pooled"]["r"] < 1
    print(f"HEADWM90 VALIDITY: {'PASS' if valid else ('FAIL' if v else 'INCOMPLETE')} (h4-bf16 finite on both parts and pooled r(v4 -> h4-bf16) < 1)")
    res = {}
    for a, b in (("h4-u5", "h4-s5-3"), ("h4-u6", "h4-s6-3")):
        p = out.get(f"{a} -> {b}")
        if p is None: res[b] = None; continue
        pooled_ok = p["pooled"]["r"] < 1 and p["pooled"]["ci95"][1] < 1
        same = p["x"]["r"] < 1 and p["c"]["r"] < 1
        res[b] = "pass" if pooled_ok and same else ("flagged (pooled pass, a part >= 1)" if pooled_ok else "fail")
        print(f"  {b} vs {a}: {res[b]}")
    if not valid: verdict = "no verdict: the validity check did not pass" if v else "INCOMPLETE"
    elif any(x is None for x in res.values()): verdict = "INCOMPLETE"
    else:
        n = sum(x == "pass" for x in res.values())
        verdict = ("BUILD the slab head into the plugin (both pairs pass)" if n == 2 else
                   f"size-dependent menu option (only {[k for k, x in res.items() if x == 'pass'][0]} passes)" if n == 1 else
                   "NOT BUILT (no pair passes)")
    out["verdict"] = verdict
    print(f"HEADWM90 VERDICT #34-02: {verdict}")
json.dump(out, open(f"{O}/verdict{which}.json", "w"), indent=1)
