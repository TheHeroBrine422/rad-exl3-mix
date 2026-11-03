#!/usr/bin/env python3
"""pstat.py -- w02's paired statistics and draft-rule verdicts (README.md), from evalq.py score files.
  pstat.py pick O BASE      C9: the alpha per family from the fit-slice arms O/BASE_a<fam><a>.fit.f32 vs O/BASE.fit.f32;
                            prints "fam=a,..." (refitq.py --alpha) and writes O/BASE.alpha.json
  pstat.py b1 O YB          B1 table + verdicts (search slice; sources in YB, arms in O)
  pstat.py c9 O YB          C9 table + verdicts
  pstat.py c10 O YB         C10 table + verdicts
pair() is pcmp.py's arithmetic (gen-masked means, per-document bootstrap SE, 2000 draws, seed 0)."""
import json, os, sys
import numpy as np
GS, GF = os.environ.get("GS", "/data/g08-search1"), os.environ.get("GF", "/data/g08-fit1")
FAMS = ("attn_in", "attn_o", "gdn_in", "gdn_out", "gate_up", "down")
AL = (-1, 1, 2)

def pair(g, fa, fb, N=2000):
    m = json.load(open(f"{g}/kld.json"))
    A = np.fromfile(fa, np.float32).reshape(-1, 4); B = np.fromfile(fb, np.float32).reshape(-1, 4)
    assert len(A) == len(B) == m["rows"], (fa, fb, len(A), len(B), m["rows"])
    gen = A[:, 2] > 0; sa, sb, n = [], [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); gg = gen[s]
        sa.append(A[s, 0][gg].sum()); sb.append(B[s, 0][gg].sum()); n.append(gg.sum())
    sa, sb, n = map(np.asarray, (sa, sb, n))
    ma, mb = sa.sum() / n.sum(), sb.sum() / n.sum()
    idx = np.random.default_rng(0).integers(0, len(n), (N, len(n)))
    se = ((sb[idx].sum(1) - sa[idx].sum(1)) / n[idx].sum(1)).std()
    return {"kld_A": float(ma), "kld_B": float(mb), "ratio": float(mb / ma), "diff": float(mb - ma), "se": float(se),
            "z": float((mb - ma) / se)}

def pick(O, base):
    ref = f"{O}/{base}.fit.f32"; out = {}; det = {}
    for f in FAMS:
        pts = [(0.0, 0.0, 0.0)]
        for a in AL:
            p = pair(GF, ref, f"{O}/{base}_a{f}{a}.fit.f32"); pts.append((float(a), p["diff"], p["se"]))
        x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        c2, c1, c0 = np.polyfit(x, y, 2)
        best = min(pts, key=lambda p: p[1])
        if c2 > 0: a_ = float(np.clip(-c1 / (2 * c2), -1, 2))
        else: a_ = best[0]
        if not (best[1] < -2 * best[2]): a_ = 0.0              # no fit-slice evidence (z > -2): leave the family alone
        out[f] = round(a_ * 20) / 20
        det[f] = {"points": pts, "quad": [float(c2), float(c1), float(c0)], "alpha": out[f]}
    s = ",".join(f"{f}={v:g}" for f, v in out.items())
    json.dump({"alpha": s, "families": det}, open(f"{O}/{base}.alpha.json", "w"), indent=1)
    return s

def row(name, p): return f"  {name:34s} ratio {p['ratio']:.4f}  diff {p['diff']:+.2e}  se {p['se']:.1e}  z {p['z']:+6.2f}"
def have(*fs): return all(os.path.exists(f) for f in fs)

def b1(O, YB):
    for k in (3, 2):
        print(f"K{k}")
        res = {}
        for arm, srcs in (("h6", ("bdiag", "base")), ("u6", ("bdiag",))):
            for b in srcs:
                ps = []
                for s in ("", "_s1", "_s2"):
                    src, t = f"k{k}{b}{s}", f"k{k}{b}{s}_{arm}"
                    if not have(f"{YB}/{src}.f32", f"{O}/{t}.f32"): continue
                    p = pair(GS, f"{YB}/{src}.f32", f"{O}/{t}.f32"); ps.append(p); print(row(f"{src} -> {arm}", p))
                if len(ps) == 1 and arm == "h6":                       # the default seed-0 screen
                    p = ps[0]; go = p["ratio"] <= 0.995 and p["z"] <= -2
                    print(f"  {b}/{arm} seed-0 screen: {'run B1_FULL=1 (3-seed rule)' if go else 'B1 closed here (ratio > 0.995 or z > -2)'}")
                if len(ps) < 3: print(f"  {b}/{arm}: {len(ps)}/3 seeds, no 3-seed verdict"); continue
                rs = [p["ratio"] for p in ps]; mr = float(np.mean(rs))
                ok = all(r < 1 for r in rs) and mr <= 0.99 and sum(p["z"] <= -3 for p in ps) >= 2
                res[(b, arm)] = (mr, ok)
                print(f"  {b}/{arm}: mean ratio {mr:.4f}, worst seed {max(rs):.4f} -> {'PASS' if ok else 'fail'} (draft rule B1)")
                if any(p["ratio"] > 1.01 and p["z"] >= 3 for p in ps): print(f"  {b}/{arm}: HARM on a seed (> +1%, z >= 3)")
        if ("bdiag", "h6") in res and ("bdiag", "u6") in res:
            gh, gu = 1 - res[("bdiag", "h6")][0], 1 - res[("bdiag", "u6")][0]
            print(f"  u6 share of h6's gain on bdiag: {gu / gh if gh > 0 else float('nan'):.2f} (>= 0.67: the gain is undamping / rounds, not h)")

def c9(O, YB):
    for k in (3, 2):
        b = f"k{k}base"; print(f"K{k}")
        if os.path.exists(f"{O}/{b}.alpha.json"): print(f"  alpha* {json.load(open(f'{O}/{b}.alpha.json'))['alpha']}")
        for a in AL:
            if have(f"{O}/{b}.fit.f32", f"{O}/{b}_aall{a}.fit.f32"): print(row(f"fit: all={a}", pair(GF, f"{O}/{b}.fit.f32", f"{O}/{b}_aall{a}.fit.f32")))
        if have(f"{O}/{b}.fit.f32", f"{O}/{b}_astar.fit.f32"): print(row("fit: alpha*", pair(GF, f"{O}/{b}.fit.f32", f"{O}/{b}_astar.fit.f32")))
        if not have(f"{YB}/{b}.f32", f"{O}/{b}_astar.f32", f"{YB}/{b}_s1.f32", f"{O}/{b}_s1_astar.f32"): print("  incomplete"); continue
        p = pair(GS, f"{YB}/{b}.f32", f"{O}/{b}_astar.f32"); q = pair(GS, f"{YB}/{b}_s1.f32", f"{O}/{b}_s1_astar.f32")
        print(row("search: base -> alpha*", p)); print(row("search: base_s1 -> alpha* (transfer)", q))
        ok = p["z"] <= -3 and p["ratio"] <= 0.995 and q["ratio"] < 1
        print(f"  -> {'PASS' if ok else 'fail'} (draft rule C9)")

def c10(O, YB):
    for k in (3, 2):
        print(f"K{k}")
        for b in ("base", "bdiag"):
            s = f"k{k}{b}"
            if not have(f"{YB}/{s}.f32", f"{O}/{s}_swj.f32"): print(f"  {s}: incomplete"); continue
            pj = pair(GS, f"{YB}/{s}.f32", f"{O}/{s}_swj.f32"); print(row(f"{s} -> swj", pj))
            ok = pj["ratio"] <= 0.99 and pj["z"] <= -3
            if have(f"{O}/{s}_swu.f32"):
                pu = pair(GS, f"{YB}/{s}.f32", f"{O}/{s}_swu.f32"); pq = pair(GS, f"{O}/{s}_swu.f32", f"{O}/{s}_swj.f32")
                print(row(f"{s} -> swu", pu)); print(row(f"{s}_swu -> swj", pq))
                print(f"  joint beyond up-only: {'yes' if pq['z'] <= -2 else 'no'}")
            print(f"  {s}: swj {'PASS' if ok else 'fail'} at K{k} (draft rule C10 needs both K on base)")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pick": print(pick(sys.argv[2], sys.argv[3]))
    else: {"b1": b1, "c9": c9, "c10": c10}[cmd](*sys.argv[2:4])
