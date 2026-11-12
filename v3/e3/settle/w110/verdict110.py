#!/usr/bin/env python3
"""verdict110.py A -- #145 (../w110.md "Statistics" / "Rule"): every arm vs its mix's T on golden-search27d (both
parts), pooled (w84/pair84.py: stratified document bootstrap, 2000, seed 0), per part, per document (docstat.py's
arithmetic: per-document ratio geo / median, share better, top-document share); then pass / worse per the rule and the
C9 / swj decisions. Arms identical to another (A/mix/TAG.same) take that arm's numbers. Writes A/verdict.json; prints
the tables and, last, "VERDICT {json}"."""
import json, os, subprocess, sys
import numpy as np

A = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w84"))
from pair84 import pair, doc_sums  # noqa: E402

G = {p: f"{A}/golden-search27d-{p}" for p in ("x", "c")}
M1 = ["k1-cs", "k1-os", "k1-es", "k1-cn", "k1-on", "k1-en"]
M2 = ["sp-cc-ss", "sp-oo-ss", "sp-ee-ss", "sp-cc-nn", "sp-oo-nn", "sp-ee-nn", "sp-oc-ss", "sp-co-ss", "sp-ec-ss", "sp-ce-ss",
      "sp-cc-ns", "sp-cc-sn"]
TREF = {**{t: "k1-cs" for t in M1}, **{t: "sp-cc-ss" for t in M2}}


def rows(tag, p):
    s = f"{A}/mix/{tag}.same"
    if os.path.exists(s): tag = open(s).read().strip()
    return f"{A}/scores/{tag}-{p}.json.rows"


def perdoc(ref, tag):
    out = {}
    allr = []; sh = {}
    for p in ("x", "c"):
        sa = doc_sums(G[p], rows(ref, p))[0]; sb = doc_sums(G[p], rows(tag, p))[0]
        ok = sa > 0; rr = sb[ok] / sa[ok]; allr.append(rr); share = sa / sa.sum()
        big = share > 0.20
        trim = float(sb[~big].sum() / sa[~big].sum()) if big.any() else None
        out[p] = {"docs": int(len(sa)), "geo": float(np.exp(np.log(rr).mean())), "median": float(np.median(rr)),
                  "better": float(np.mean(rr < 1)), "top_share": float(share.max()), "r_without_gt20": trim}
    rr = np.concatenate(allr)
    out["both"] = {"docs": int(len(rr)), "geo": float(np.exp(np.log(rr).mean())), "median": float(np.median(rr)),
                   "better": float(np.mean(rr < 1))}
    return out


res = {}
for tag in M1 + M2:
    ref = TREF[tag]
    if tag == ref: continue
    if not all(os.path.exists(rows(t, p)) for t in (tag, ref) for p in ("x", "c")): print(f"{tag}: not scored"); continue
    pooled = pair([(G[p], rows(ref, p), rows(tag, p)) for p in ("x", "c")])
    parts = {p: pair([(G[p], rows(ref, p), rows(tag, p))]) for p in ("x", "c")}
    pd = perdoc(ref, tag)
    ps = (pooled["r"] <= 0.99 and pd["both"]["geo"] <= 0.99 and parts["x"]["r"] < 1 and parts["c"]["r"] < 1)
    wo = pooled["r"] >= 1.0 or pd["both"]["geo"] >= 1.0
    res[tag] = {"ref": ref, "same_as": open(f"{A}/mix/{tag}.same").read().strip() if os.path.exists(f"{A}/mix/{tag}.same") else None,
                "pooled": pooled, "x": parts["x"], "c": parts["c"], "perdoc": pd, "pass": bool(ps), "worse": bool(wo)}

# T's absolute KL per mix
absk = {}
for t in ("k1-cs", "sp-cc-ss"):
    if all(os.path.exists(rows(t, p)) for p in ("x", "c")):
        p = pair([(G[q], rows(t, q), rows(t, q)) for q in ("x", "c")]); absk[t] = {"kld": p["kld_a"], "top1": p["top1_a"]}

print("# #145 golden-search27d: arm vs T (k1-cs in M1, sp-cc-ss in M2), one seed (scale-only arms on the same codes)")
for t, v in absk.items(): print(f"T {t}: KLD {v['kld']:.5f}, top-1 {v['top1']:.4f}")
print(f"{'arm':10s} {'pooled r':>9s} {'95% CI':>17s} {'z':>6s} {'x r':>7s} {'c r':>7s} {'geo':>7s} {'med':>7s} {'geo x':>7s} {'geo c':>7s} {'top x':>6s} {'top c':>6s} pass worse")
for t, v in res.items():
    P = v["pooled"]; d = v["perdoc"]
    print(f"{t:10s} {P['r']:9.4f} [{P['ci95'][0]:.4f}, {P['ci95'][1]:.4f}] {P['z']:+6.2f} {v['x']['r']:7.4f} {v['c']['r']:7.4f} "
          f"{d['both']['geo']:7.4f} {d['both']['median']:7.4f} {d['x']['geo']:7.4f} {d['c']['geo']:7.4f} {d['x']['top_share']:6.3f} "
          f"{d['c']['top_share']:6.3f} {'PASS' if v['pass'] else '-':4s} {'WORSE' if v['worse'] else '-'}"
          + (f"  (= {v['same_as']})" if v["same_as"] else ""))


def g(t, k="pass"): return res.get(t, {}).get(k)


# 3 x 2 tables + interaction
def r_(t): return 1.0 if TREF.get(t) == t else res[t]["pooled"]["r"] if t in res else float("nan")


print("\n3 x 2 (pooled r vs T); interaction = r(X, off) / (r(X, on) r(cur, off))")
for name, tab in (("M1 K1", {c: (f"k1-{c}s", f"k1-{c}n") for c in "coe"}), ("M2 splice", {c: (f"sp-{c}{c}-ss", f"sp-{c}{c}-nn") for c in "coe"})):
    cn = r_(tab["c"][1])
    for c, (on, off) in tab.items():
        it = r_(off) / (r_(on) * cn) if c != "c" else 1.0
        print(f"  {name} C9 {dict(c='cur', o='off', e='e3')[c]:3s}: swj on {r_(on):.4f}  swj off {r_(off):.4f}  interaction {it:.4f}")

# rule: C9
dec = {"c9": None, "c9_band": None, "swj": None, "swj_band": None, "notes": []}
glob = [X for X in "oe" if g(f"k1-{X}s") and g(f"sp-{X}{X}-ss")]
if glob:
    X = min(glob, key=lambda X: np.sqrt(res[f"k1-{X}s"]["pooled"]["r"] * res[f"sp-{X}{X}-ss"]["pooled"]["r"]))
    dec["c9"] = {"o": "off", "e": "e3"}[X] + " at every tested K (K0.5 / 1 / 1.5)"
else:
    for X in "oe":
        nm = {"o": "off", "e": "e3"}[X]
        if g(f"sp-{X}c-ss") and not g(f"k1-{X}s") and g(f"sp-c{X}-ss", "worse"): dec["c9_band"] = f"{nm} below K1 (K <= 0.875; boundary unmeasured)"
        elif g(f"k1-{X}s") and g(f"sp-c{X}-ss") and g(f"sp-{X}c-ss", "worse"): dec["c9_band"] = f"{nm} from K1 up"
        else:
            pts = {k: g(t) for k, t in (("K0.5", f"sp-{X}c-ss"), ("K1", f"k1-{X}s"), ("K1.5", f"sp-c{X}-ss"))}
            if any(pts.values()): dec["notes"].append(f"C9 {nm}: passes at {[k for k, v in pts.items() if v]} without a rule pattern (report only)")
        if dec["c9_band"]: break
if not dec["c9"] and not dec["c9_band"]: dec["c9"] = "stays (cur)"
# rule: swj
if g("k1-cn") and g("sp-cc-nn"): dec["swj"] = "off at every tested K -> propose post.swj.k_min >= 2"
elif g("sp-cc-ns") and not g("k1-cn") and g("sp-cc-sn", "worse"): dec["swj_band"] = "off below K1 -> propose post.swj.k_min = 1"
else:
    dec["swj"] = "stays at every level <= K4.5"
    pts = {k: g(t) for k, t in (("K0.5", "sp-cc-ns"), ("K1", "k1-cn"), ("K1.5", "sp-cc-sn"))}
    if any(pts.values()): dec["notes"].append(f"swj off passes at {[k for k, v in pts.items() if v]} without a rule pattern (report only)")
# per mix, the task's form: best C9 (swj on) and swj on/off (C9 cur) by pooled r, both parts agreeing in sign
pm = {}
for name, c9arms, swoff in (("M1", {"cur": None, "off": "k1-os", "e3": "k1-es"}, "k1-cn"),
                            ("M2", {"cur": None, "off": "sp-oo-ss", "e3": "sp-ee-ss"}, "sp-cc-nn")):
    def agree(t): return t in res and (res[t]["x"]["r"] - 1) * (res[t]["c"]["r"] - 1) > 0
    best = min(c9arms, key=lambda a: 1.0 if c9arms[a] is None else res.get(c9arms[a], {}).get("pooled", {}).get("r", 9))
    pm[name] = {"best_c9": best, "best_c9_parts_agree": None if best == "cur" else bool(agree(c9arms[best])),
                "swj": "off" if swoff in res and res[swoff]["pooled"]["r"] < 1 else "on", "swj_parts_agree": bool(agree(swoff))}
dec["per_mix"] = pm
json.dump({"arms": res, "abs": absk, "decision": dec}, open(f"{A}/verdict.json", "w"), indent=1, default=float)
print("\nper mix:", json.dumps(pm))
for n in dec["notes"]: print("note:", n)
print("VERDICT " + json.dumps({k: v for k, v in dec.items() if k not in ("notes", "per_mix")}))
