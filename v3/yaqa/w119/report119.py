#!/usr/bin/env python3
"""report119.py CMD -- w119 readouts and verdicts (RULES.md + amendment 1) on score119's outputs.
Env: SC (scores root: SC/<set>/ref.npz, SC/<set>/<option>.npz), DOCS (docs119.json), DOCS94 (w94's docs94.json, for the
qad set), SHARE (calibshare119.json), PERMS (dir with S.npy, T.npy), OUT (output dir), SLAB (headslab plan.json of the
S donors, for S-slab plans).
  e2      E2 verdict (order, sigma_out) + the a8 arm, on search; fit / qad / unweighted reported
  e3      E3 (c): T plans and S-slab plans chosen on fit, judged on search; the fit-gen pricing curve
  e9      E9 hull on fit-gen (calib-weighted) over every option and plan
  verify  the final check (once): winner's plans vs S-slab on verify (U = UFAM), E9's top options report-only"""
import itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibshare119 import stratum

SC, OUT = os.environ.get("SC", "scores"), os.environ.get("OUT", "rep")
DOCS, DOCS94, SHARE = os.environ.get("DOCS", "docs119.json"), os.environ.get("DOCS94", "docs94.json"), os.environ.get("SHARE", "calibshare119.json")
PERMS = os.environ.get("PERMS", "perms")
NB, RB, V = 1940, 128, 248320
RARE = {"fineweb2-hq/arb_Arab", "fineweb2-hq/rus_Cyrl", "fineweb2-hq/ell_Grek", "fineweb2-hq/fas_Arab", "fineweb2-hq/jpn_Jpan",
        "fineweb2-hq/cmn_Hani", "wikipedia/hi", "wikipedia/he", "wikipedia/ko", "wikipedia/bn", "wikipedia/uk", "wikipedia/th"}
TIERS = [(0, 4), (4, 32), (32, 256), (256, NB)]          # blocks of tiers 512 / 4,096 / 32,768 / rest
TROWS = [(b - a) * RB for a, b in TIERS]
os.makedirs(OUT, exist_ok=True)
LOGF = open(f"{OUT}/report119.log", "a")


def say(*a):
    print(*a, flush=True); print(*a, file=LOGF, flush=True)


class Set:
    def __init__(self, name):
        self.name = name; d = f"{SC}/{name}"; r = np.load(f"{d}/ref.npz")
        self.doc, self.top = r["doc"], r["top"]; self.N = len(self.doc)
        sets = json.load(open(DOCS94 if name == "qad" else DOCS))["sets"]
        src = {x["doc"]: x["source"] for x in sets[name]}
        self.udocs, self.di = np.unique(self.doc, return_inverse=True)
        self.src = np.array([src[int(d)] for d in self.udocs])
        # calib-composition weights per document (RULES.md)
        share = json.load(open(SHARE)); st = np.array([stratum(s) for s in self.src])
        cnt = {s: int((st == s).sum()) for s in set(st)}
        st = np.array([s if cnt[s] >= 2 else "other" for s in st])
        rows_doc = np.bincount(self.di, minlength=len(self.udocs)).astype(float)
        strata = sorted(set(st)); cs = {}
        for s in strata:
            cs[s] = sum(v for k, v in share.items() if (k == s) or (s == "other" and cnt.get(k, 0) < 2 and k in cnt))
        tot = sum(cs.values()); w = np.zeros(len(self.udocs))
        for s in strata:
            m = st == s; dump_share = rows_doc[m].sum() / rows_doc.sum()
            w[m] = (cs[s] / tot) / dump_share if dump_share > 0 else 0
        w = np.minimum(w, 5.0); w *= rows_doc.sum() / (w * rows_doc).sum()
        self.wdoc = w; self.rare_doc = np.array([s in RARE for s in self.src])
        self.rank = None
        self._c = {}

    def rare_token(self):
        if self.rank is None:
            S = np.load(f"{PERMS}/S.npy"); self.rank = np.empty(V, np.int64); self.rank[S] = np.arange(V)
        return self.rank[self.top] >= 20000

    def kl(self, opt):
        if opt not in self._c:
            self._c[opt] = np.load(f"{SC}/{self.name}/{opt}.npz")["kl"].astype(np.float64)
        return self._c[opt]

    def has(self, opt): return os.path.exists(f"{SC}/{self.name}/{opt}.npz")

    def docsum(self, k): return np.bincount(self.di, weights=k, minlength=len(self.udocs))

    def mean(self, k, weighted=True):
        s = self.docsum(k); n = np.bincount(self.di, minlength=len(self.udocs))
        w = self.wdoc if weighted else np.ones(len(s))
        return float((w * s).sum() / (w * n).sum())

    def ratio(self, ka, kb, weighted=True, B=2000):
        """kb / ka (calib-weighted ratio of sums), document bootstrap (seed 0)"""
        sa, sb = self.docsum(ka), self.docsum(kb); w = self.wdoc if weighted else np.ones(len(sa))
        r = (w * sb).sum() / (w * sa).sum(); rng = np.random.default_rng(0); bs = []
        for _ in range(B):
            i = rng.integers(0, len(sa), len(sa)); bs.append((w[i] * sb[i]).sum() / (w[i] * sa[i]).sum())
        return float(r), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    def guard(self, ka, kb):
        """rare-script and rare-token KL ratios kb / ka (unweighted, pooled rows)"""
        mr = self.rare_doc[self.di]; mt = self.rare_token()
        f = lambda m: float(kb[m].sum() / ka[m].sum()) if m.any() and ka[m].sum() > 0 else float("nan")
        return f(mr), f(mt)


def fmt(r): return f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"


# ------------------------------------------------------------------ E2
def e2():
    S = Set("search"); res = {"search": {}}
    def cmp(x, k):
        a, b = S.kl(f"S-K{k:g}"), S.kl(f"{x}-K{k:g}")
        r = S.ratio(a, b); g = S.guard(a, b); return {"ratio": r, "unweighted": S.ratio(a, b, False), "rare_script": g[0], "rare_token": g[1],
                                                      "kl_S": S.mean(a), "kl_X": S.mean(b)}
    out = {}
    for x, ks in (("R16", (3, 4, 5)), ("N", (3, 4))):
        out[x] = {k: cmp(x, k) for k in ks if S.has(f"{x}-K{k:g}")}
        for k, v in out[x].items():
            say(f"E2 search {x}/S K{k}: {fmt(v['ratio'])} (unweighted {fmt(v['unweighted'])}); rare-script {v['rare_script']:.3f} "
                f"rare-token {v['rare_token']:.3f}; KL S {v['kl_S']:.3e} X {v['kl_X']:.3e}")
    def passes(x):
        o = out.get(x, {})
        if 3 not in o or 4 not in o: return False
        ok = all(o[k]["ratio"][0] <= 0.90 and o[k]["ratio"][2] < 0.97 for k in (3, 4))
        if x == "R16": ok = ok and 5 in o and o[5]["ratio"][0] <= 0.95
        ok = ok and all(v["rare_script"] <= 1.10 and v["rare_token"] <= 1.30 for v in o.values())
        return ok
    pR, pN = passes("R16"), passes("N")
    nr = {k: S.ratio(S.kl(f"R16-K{k}"), S.kl(f"N-K{k}")) for k in (3, 4) if S.has(f"N-K{k}") and S.has(f"R16-K{k}")}
    for k, v in nr.items(): say(f"E2 search N/R16 K{k}: {fmt(v)}")
    if pN: pN = all(nr[k][0] <= 1.05 for k in (3, 4))
    order = "N" if (pN and pR) or (pN and not pR) else ("R16" if pR else "S")
    say(f"E2 ORDER: R16 {'PASS' if pR else 'fail'}, N {'PASS' if pN else 'fail'} -> donor order {order}")
    # sigma_out
    sig = {"0.1": None}
    if S.has("R16-K4"):
        for s_ in ("s003", "s03"):
            if S.has(f"R16{s_}-K4"):
                a, b = S.kl("R16-K4"), S.kl(f"R16{s_}-K4"); r = S.ratio(a, b); g = S.guard(S.kl("S-K4"), b)
                sig[s_] = {"ratio_vs_0.1": r, "guard_vs_S": g}
                say(f"E2 search sigma_out {s_} / 0.1 (R16 K4): {fmt(r)}; guards vs S {g[0]:.3f} / {g[1]:.3f}")
    best = [k for k, v in sig.items() if v and v["ratio_vs_0.1"][0] <= 0.97 and v["ratio_vs_0.1"][2] < 1 and v["guard_vs_S"][0] <= 1.10 and v["guard_vs_S"][1] <= 1.30]
    sigma = min(best, key=lambda k: sig[k]["ratio_vs_0.1"][0]) if best else "0.1"
    say(f"E2 SIGMA_OUT: {sigma}")
    # optimism / other sets
    for sn in ("fit", "qad"):
        if not os.path.exists(f"{SC}/{sn}/ref.npz"): continue
        T = Set(sn)
        for x, ks in (("R16", (3, 4, 5)), ("N", (3, 4))):
            for k in ks:
                if T.has(f"{x}-K{k}") and T.has(f"S-K{k}"):
                    say(f"E2 {sn} {x}/S K{k}: {fmt(T.ratio(T.kl(f'S-K{k}'), T.kl(f'{x}-K{k}')))}")
    # a8 arm
    a8 = {}
    for o in sorted(o[:-7] for o in os.listdir(f"{SC}/search") if o.endswith("-a8.npz")):
        a, b = S.kl(o), S.kl(o + "-a8"); r = S.ratio(a, b); d = S.mean(b) - S.mean(a)
        a8[o] = {"ratio": r, "dkl": d, "kl_f16": S.mean(a)}
        say(f"E2 a8/f16 {o}: {fmt(r)}  dKL {d:+.3e} (f16 {S.mean(a):.3e})")
    a8ok = all(o in a8 and a8[o]["ratio"][0] <= 1.02 and a8[o]["ratio"][2] < 1.05 for o in ("S-K4", "S-K6"))
    say(f"E2 A8 HEAD: KL condition {'met' if a8ok else 'not met'}; no A8 head kernel timing -> head stays f16 (amendment 1)")
    json.dump({"order": order, "R16": out.get("R16"), "N": out.get("N"), "N_over_R16": nr, "sigma": sig, "sigma_out": sigma,
               "a8": a8, "a8_kl_ok": a8ok}, open(f"{OUT}/e2.json", "w"), indent=1, default=str)


# ------------------------------------------------------------------ block-sum plans
def tier_sums(st, fam, ks):
    """per donor K: per-row sums of d / dl over each tier's blocks [len(ks), 4, N]"""
    D = np.zeros((len(ks), 4, st.N)); DL = np.zeros_like(D)
    for i, k in enumerate(ks):
        z = np.load(f"{SC}/{st.name}/{fam}-K{k:g}.npz" if k != 16 else f"{SC}/{st.name}/T-bf16.npz")   # bf16 rows: any T family
        for t, (a, b) in enumerate(TIERS):
            D[i, t] = z["d"][:, a:b].sum(1, dtype=np.float64); DL[i, t] = z["dl"][:, a:b].sum(1, dtype=np.float64)
    return D, DL


def plan_kl(D, DL, idx):
    t = np.arange(4); return D[idx, t].sum(0) + np.log1p(DL[idx, t].sum(0))


TFAM, UFAM = os.environ.get("TFAM", "T"), os.environ.get("UFAM", "R16")      # amendment 3: T03 / R16s003 decide


def tplans(st, ks):
    D, DL = tier_sums(st, TFAM, ks); out = []
    for idx in itertools.product(range(len(ks)), repeat=4):
        bits = sum(ks[i] * n for i, n in zip(idx, TROWS)) / V
        out.append((idx, bits, plan_kl(D, DL, np.array(idx))))
    return out


def block_plan_kl(st, fam, kb):
    """a plan = donor K per stored block (fam's order) -> per-row KL"""
    ks = sorted(set(kb)); d = np.zeros(st.N); dl = np.zeros(st.N); kb = np.asarray(kb)
    for k in ks:
        z = np.load(f"{SC}/{st.name}/{fam}-K{k:g}.npz"); m = kb == k
        d += z["d"][:, m].sum(1, dtype=np.float64); dl += z["dl"][:, m].sum(1, dtype=np.float64)
    return d + np.log1p(dl)


TK = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]


def choose_t(fit_plans, fit, B, ks):
    best = None
    for idx, bits, kl in fit_plans:
        if bits > B + 1e-9: continue
        v = fit.mean(kl)
        if best is None or v < best[0]: best = (v, idx, bits)
    return best


def e3():
    fit, se = Set("fit"), Set("search")
    ks = [k for k in TK if fit.has(f"{TFAM}-K{k:g}")]
    say(f"E3 T donors ({TFAM}, U = {UFAM}): {ks}")
    fp = tplans(fit, ks); sp = {idx: kl for idx, _, kl in tplans(se, ks)}
    slab = json.load(open(os.environ["SLAB"]))["plans"] if os.environ.get("SLAB") else {}
    res = {}; ratios = {"T": [], "S": []}; gu = {"T": [], "S": []}; rows = []
    for B in (3, 4, 5, 6):
        v, idx, bits = choose_t(fp, fit, B, ks)
        kt = sp[idx]; ku = se.kl(f"{UFAM}-K{B}") if se.has(f"{UFAM}-K{B}") else None
        sname = f"s{B}-3"
        ksl = block_plan_kl(se, "S", slab[sname]["K_blocks"]) if sname in slab else se.kl(f"S-K{B}")
        sbits = slab[sname]["avg_bits"] if sname in slab else B
        r_tu = se.ratio(ku, kt); r_su = se.ratio(ku, ksl)
        g_t = se.guard(ksl, kt); g_u = se.guard(ksl, ku)
        res[B] = {"T": {"Ks": [ks[i] for i in idx], "bits": bits, "kl": se.mean(kt), "fit_kl": v}, "U_kl": se.mean(ku),
                  "S": {"plan": sname if sname in slab else f"S-K{B}", "bits": sbits, "kl": se.mean(ksl)},
                  "T_over_U": r_tu, "S_over_U": r_su, "guard_T_vs_S": g_t, "guard_U_vs_S": g_u}
        ratios["T"].append(kt); ratios["S"].append(ksl); rows.append(ku)
        say(f"E3 B{B}: T {[ks[i] for i in idx]} ({bits:.3f} b) {se.mean(kt):.3e} | U {se.mean(ku):.3e} | S-slab "
            f"{res[B]['S']['plan']} ({sbits:.3f} b) {se.mean(ksl):.3e} | T/U {fmt(r_tu)} S/U {fmt(r_su)} | guards vs S: "
            f"T {g_t[0]:.3f}/{g_t[1]:.3f} U {g_u[0]:.3f}/{g_u[1]:.3f}")
    # geo means over budgets with a joint document bootstrap
    rng = np.random.default_rng(0); nd = len(se.udocs); w = se.wdoc
    def geo(lay):
        sa = [se.docsum(k) for k in rows]; sb = [se.docsum(k) for k in ratios[lay]]
        g0 = np.exp(np.mean([np.log((w * b).sum() / (w * a).sum()) for a, b in zip(sa, sb)])); bs = []
        for _ in range(2000):
            i = rng.integers(0, nd, nd)
            bs.append(np.exp(np.mean([np.log((w[i] * b[i]).sum() / (w[i] * a[i]).sum()) for a, b in zip(sa, sb)])))
        return float(g0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    gT, gS = geo("T"), geo("S")
    say(f"E3 geo over budgets vs U: T {fmt(gT)}, S-slab {fmt(gS)}")
    guard_ok = {lay: all((res[B]["guard_T_vs_S"] if lay == "T" else (1, 1))[0] <= 1.10 and
                         (res[B]["guard_T_vs_S"] if lay == "T" else (1, 1))[1] <= 1.30 for B in res) for lay in ("T", "S")}
    guard_ok["U"] = all(res[B]["guard_U_vs_S"][0] <= 1.10 and res[B]["guard_U_vs_S"][1] <= 1.30 for B in res)
    cand = {"U": 1.0}
    if guard_ok["T"] and gT[0] <= 0.90 and gT[2] < 1: cand["T"] = gT[0]
    if gS[0] <= 0.90 and gS[2] < 1: cand["S-slab"] = gS[0]
    if not guard_ok["U"]: cand.pop("U")
    win = min(cand, key=cand.get) if cand else "S-slab"
    say(f"E3 LAYOUT: {win} (candidates {cand}; guards {guard_ok})")
    json.dump({"budgets": res, "geo_T_over_U": gT, "geo_S_over_U": gS, "guards_ok": guard_ok, "winner": win, "T_donors": ks},
              open(f"{OUT}/e3-{TFAM}-{UFAM}.json", "w"), indent=1, default=str)


# ------------------------------------------------------------------ E9 hull
def lower_hull(pts):
    pts = sorted(pts); h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0: h.pop()
        h.append(p)
    return h


def e9():
    fit = Set(os.environ.get("E9SET", "fit")); opts = []
    for f in sorted(os.listdir(f"{SC}/{fit.name}")):
        o = f[:-4]
        if o == "ref" or not f.endswith(".npz"): continue
        if o.startswith(("S-K", "Sm1-K", "R16-K", "R16s003-K", "R16s003m1-K", "R16s03-K", "N-K")):
            if o.endswith("-a8"): continue
            opts.append((float(o.split("-K")[1]), fit.mean(fit.kl(o)), o))
        elif o.startswith("i8-"):
            opts.append((8.125, fit.mean(fit.kl(o)), o))
        elif o == "bf16-full":
            opts.append((16.0, fit.mean(fit.kl(o)), o))
    ks = [k for k in TK if fit.has(f"{TFAM}-K{k:g}")] + ([16] if fit.has("T-bf16") else [])
    tp = tplans(fit, ks)
    best = {}
    for idx, bits, kl in tp:
        key = round(bits * 8) / 8
        v = fit.mean(kl)
        if key not in best or v < best[key][0]: best[key] = (v, idx, bits)
    for key, (v, idx, bits) in best.items():
        opts.append((bits, v, "T[" + ",".join("bf16" if ks[i] == 16 else f"{ks[i]:g}" for i in idx) + "]"))
    if os.environ.get("SLAB"):
        slab = json.load(open(os.environ["SLAB"]))["plans"]
        for n, p in slab.items():
            if n.startswith("s"): opts.append((p["avg_bits"], fit.mean(block_plan_kl(fit, "S", p["K_blocks"])), f"S-slab {n}"))
    pts = [(b, v, n) for b, v, n in opts]
    h = lower_hull([(b, v) for b, v, _ in pts]); hs = {(b, v) for b, v in h}
    out = []
    for b, v, n in sorted(pts):
        on = (b, v) in hs
        # hull without this option, at its bits (interpolated)
        h2 = lower_hull([(b2, v2) for b2, v2, n2 in pts if n2 != n]); ref = None
        for (x0, y0), (x1, y1) in zip(h2, h2[1:]):
            if x0 <= b <= x1: ref = y0 + (y1 - y0) * (b - x0) / (x1 - x0)
        out.append({"opt": n, "bits": b, "kl": v, "on_hull": on, "vs_hull_without": (v / ref) if ref else None})
    for o in out:
        if o["on_hull"] or o["opt"].startswith(("i8", "bf16", "T[bf16")) or o["opt"] in ("S-K7", "S-K6"):
            vh = "" if o["vs_hull_without"] is None else f"  x{o['vs_hull_without']:.3f} of the hull without it"
            say(f"E9 {fit.name} {o['opt']:28s} {o['bits']:.3f} b  KL {o['kl']:.3e}  {'HULL' if o['on_hull'] else ''}{vh}")
    json.dump(out, open(f"{OUT}/e9-{fit.name}-{TFAM}.json", "w"), indent=1)


# ------------------------------------------------------------------ final check (once)
def verify():
    """E3's winner (U = UFAM uniform) vs S-slab at budgets 3-6 on verify-gen; E9's top options report-only"""
    ve = Set("verify"); slab = json.load(open(os.environ["SLAB"]))["plans"]; res = {}; su, ss = [], []
    for B in (3, 4, 5, 6):
        ku = ve.kl(f"{UFAM}-K{B}"); sname = f"s{B}-3"
        ks = block_plan_kl(ve, "S", slab[sname]["K_blocks"]) if sname in slab else ve.kl(f"S-K{B}")
        r = ve.ratio(ks, ku); g = ve.guard(ks, ku); su.append(ve.docsum(ku)); ss.append(ve.docsum(ks))
        res[B] = {"U": ve.mean(ku), "S_slab": ve.mean(ks), "U_over_S": r, "unweighted": ve.ratio(ks, ku, False), "guard_U_vs_S": g}
        say(f"VERIFY B{B}: U {ve.mean(ku):.3e} S-slab {sname if sname in slab else f'S-K{B}'} {ve.mean(ks):.3e} U/S {fmt(r)} "
            f"(unweighted {fmt(res[B]['unweighted'])}); rare-script {g[0]:.3f} rare-token {g[1]:.3f}")
    w = ve.wdoc; g0 = float(np.exp(np.mean([np.log((w * a).sum() / (w * b).sum()) for a, b in zip(su, ss)])))
    say(f"VERIFY geo U / S-slab over budgets: {g0:.3f}")
    rep = {}
    for o in (f"{UFAM}-K6.5", f"{UFAM}m1-K7", f"{UFAM}-K7", "i8-gptq"):
        if ve.has(o): rep[o] = ve.mean(ve.kl(o)); say(f"VERIFY report {o}: {rep[o]:.3e}")
    if ve.has(f"{UFAM}m1-K7") and ve.has("i8-gptq"):
        r = ve.ratio(ve.kl("i8-gptq"), ve.kl(f"{UFAM}m1-K7")); rep["m1K7_over_i8gptq"] = r; say(f"VERIFY report K7 mul1 / int8 GPTQ: {fmt(r)}")
    json.dump({"budgets": res, "geo_U_over_S": g0, "report": rep}, open(f"{OUT}/verify.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    {"e2": e2, "e3": e3, "e9": e9, "verify": verify}[sys.argv[1]]()
