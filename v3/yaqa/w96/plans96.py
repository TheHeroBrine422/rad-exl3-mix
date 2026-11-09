#!/usr/bin/env python3
"""plans96.py CMD -- w96 (#127) plans and readouts on score96's block sums (RULES.md). Paths from env W (default
/data/work/w96): W/in/hs (headsens.npz, perm.npy, K<k>/wq/lm_head.rowerr.npz), W/in/x (fisher-calib.npz: groups
0 A-gen, 1 B-gen, 2 unused, 3 A-nongen, 4 B-nongen; uni-*.npy; w94's sim-search.npz + plans), W/fisher-xc.npz,
W/sums/<set> (score96), W/in/dump/<set> (doc.i32, gen.u8). Outputs into W/out.

  g1       scorer check vs w94's sim-search.npz (every plan of sim-search-plans.json) + base plans vs headslab3's plan.json
  census   item 1
  tune     the qad split grid (built from half A, scored on the qad-dump gen rows)
  final    the chosen settings on the full qad pool, scored on search-gen; verdict, rare-token KL, borderline sizes
  curve A  arm A's 1/8-bit curve (headslab curve mode) scored on search-gen, written as w94's sim npz for simjson94
"""
import json, os, subprocess, sys
import numpy as np

W = os.environ.get("W", "/data/work/w96"); HSD = f"{W}/in/hs"; X = f"{W}/in/x"; O = f"{W}/out"
CODE = os.environ.get("CODE", f"{W}/code/v3/yaqa")
sys.path.insert(0, CODE); import headslab as hsl                       # the planner, unchanged
KS = [3.0, 4.0, 5.0, 6.0, 7.0]; BLK = 128
V = 248320; NB = V // BLK
os.makedirs(O, exist_ok=True)
perm = np.load(f"{HSD}/perm.npy")
E = np.stack([np.load(f"{HSD}/K{k:g}/wq/lm_head.rowerr.npz")["err"] for k in KS])      # [K, V] perm order

def say(*a):
    print(*a, flush=True)
    with open(f"{O}/plans96.log", "a") as f: print(*a, file=f)

# ---------------- block sums -> per-row KL of any plan
class Sums:
    def __init__(self, name):
        d = f"{W}/sums/{name}"; m = json.load(open(f"{d}/meta.json")); nb = (len(m["rows"]) + m["rb"] - 1) // m["rb"]
        Z = [np.load(f"{d}/blk-{i}.npz") for i in range(nb)]
        for k in ("d", "dl", "mx", "zt", "lse", "top"): setattr(self, k, np.concatenate([z[k] for z in Z]))
        self.fisher = sum(z["fisher"] for z in Z); self.psum = sum(z["psum"] for z in Z); self.n1e4 = sum(z["n1e4"].astype(np.int64) for z in Z)
        self.rows = np.array(m["rows"]); assert m["Ks"] == KS
        self.doc = np.fromfile(f"{W}/in/dump/{name}/doc.i32", np.int32)[self.rows]
        self.tb = self.top // BLK; self.N = len(self.rows)
    def kl(self, a):                                  # a: donor index per head block [NB]
        ix = np.broadcast_to(a[None, None, :], (1, 1, NB))
        ds = np.take_along_axis(self.d, np.broadcast_to(ix, (self.N, 1, NB)), 1)[:, 0].sum(1, dtype=np.float64)
        dl = np.take_along_axis(self.dl, np.broadcast_to(ix, (self.N, 1, NB)), 1)[:, 0].sum(1, dtype=np.float64)
        return ds + np.log1p(dl)
    def hit(self, a):
        m = np.take_along_axis(self.mx, np.broadcast_to(a[None, None, :], (self.N, 1, NB)), 1)[:, 0].max(1)
        return self.zt[np.arange(self.N), a[self.tb]] >= m

def kidx(kb): return np.array([KS.index(float(k)) for k in kb])

# ---------------- plans with headslab's planner
def costs(Fv):                                        # Fv: Fisher in vocab order -> block costs [K, NB]
    return (E * Fv[perm][None, :]).reshape(len(KS), NB, BLK).sum(2)
def splans(Fv, budgets=(4, 5, 6)):
    C = costs(Fv); out = {}
    for B in budgets:
        kmin = max(hsl.FLOOR, round(B - 1.5 + 1e-9))
        out[f"s{B}-3"] = hsl.slabs(C, KS, kmin, B * NB, NB, 3)
    return out
UNI = {f"u{k:g}": np.full(NB, i) for i, k in enumerate(KS)}

# ---------------- Fisher arms (rates; scale is irrelevant to the planner)
FC = np.load(f"{X}/fisher-calib.npz")
def grp(*g, key="fisher"): return sum(FC[key][i] for i in g)
def Ng(*g): return float(sum(FC["N"][i] for i in g))
_xc = None
def xc():
    global _xc
    if _xc is None: z = np.load(f"{W}/fisher-xc.npz"); _xc = (z["fisher"][0], z["n"][0], float(z["N"][0]))
    return _xc
uni = np.load(f"{X}/uni-calib.npy") + np.load(f"{X}/uni-xcorp.npy")
def bins_q(val, nb):                                  # equal-count bins, ties kept together
    e = np.unique(np.quantile(val, np.linspace(0, 1, nb + 1)[1:-1]))
    return np.searchsorted(e, val, side="right")
def headnorm():
    f = f"{O}/headnorm.npy"
    if not os.path.exists(f):
        from safetensors import safe_open
        ck = os.environ.get("CK", "/data/models/qwen38-27b-bf16")
        ix = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
        with safe_open(f"{ck}/{ix}", "pt") as h: w = h.get_tensor("lm_head.weight").float()
        np.save(f, w.norm(dim=1).numpy().astype(np.float64))
    return np.load(f)
BINS = {"U": bins_q(np.log(uni + 0.5), 64), "E": bins_q(headnorm(), 10)}

def fisher_set(halves, more=None, w=0.0):
    """(F, n, N) for qad gen of the given halves (0 = A, 1 = B), plus w × extra positions: more = 'q' (the halves'
    non-gen rows) or 'x' (xc8k-b1)."""
    F, n, N = grp(*halves), grp(*halves, key="n"), Ng(*halves)
    if more == "q":
        g = [h + 3 for h in halves]; F, n, N = F + w * grp(*g), n + w * grp(*g, key="n"), N + w * Ng(*g)
    elif more == "x":
        Fx, nx, Nx = xc(); F, n, N = F + w * Fx, n + w * nx, N + w * Nx
    return F, n, N
def smooth(F, n, N, prior, kappa):
    r = F / N; b = BINS[prior]; m = (np.bincount(b, r) / np.bincount(b))[b]
    return (n * r + kappa * m) / (n + kappa)

KAPPAS = [1, 3, 10, 30, 100, 300, 1000]; WS = [0.25, 0.5, 1.0]; PLANS3 = ("s4-3", "s5-3", "s6-3")

def geo(S, P, ref):
    k = {n: S.kl(P[n]).mean() for n in PLANS3}
    return float(np.exp(np.mean([np.log(k[n] / ref[n]) for n in PLANS3]))), k

def proxyB(P):                                        # Fisher-cost proxy on half B (report only)
    C = costs(grp(1)); return {n: float(C[P[n], np.arange(NB)].sum()) for n in PLANS3}

def cmd_tune():
    S = Sums("qad"); res = {}
    def run(name, Fv):
        P = splans(Fv); g, k = geo(S, P, ref) if name != "baseA" else (1.0, {n: S.kl(P[n]).mean() for n in PLANS3})
        pr = proxyB(P); res[name] = {"geo": g, "kl": k, "proxyB": pr, "bits": {n: float(np.array(KS)[P[n]].mean()) for n in P}}
        say(f"{name:22s} geo {g:.4f}  " + "  ".join(f"{n} {k[n]:.3e}" for n in PLANS3)
            + "  proxyB/base " + " ".join(f"{pr[n] / res['baseA']['proxyB'][n]:.3f}" if 'baseA' in res else "1" for n in PLANS3))
        return g
    F, n, N = fisher_set([0]); ref = {}
    run("baseA", F / N); ref.update(res["baseA"]["kl"])
    for pri in ("U", "E"):
        for k in KAPPAS: run(f"smooth-{pri}-k{k}", smooth(F, n, N, pri, k))
    for mo in ("q", "x"):
        for w in WS:
            Fm, nm, Nm = fisher_set([0], mo, w); run(f"more-{mo}-w{w:g}", Fm / Nm)
    def pick(prefix, key):
        c = {k: v["geo"] for k, v in res.items() if k.startswith(prefix)}; best = min(c.values())
        ok = [k for k in c if c[k] <= best * 1.005]; return min(ok, key=key)
    ksm = lambda s: (float(s.split("-k")[1]), s)
    sm = pick("smooth-", ksm)
    mq = pick("more-q-", lambda s: float(s.split("-w")[1])); mx = pick("more-x-", lambda s: float(s.split("-w")[1]))
    mo = min((mq, mx), key=lambda s: res[s]["geo"]); mw = float(mo.split("-w")[1]); msrc = mo.split("-")[1]
    Fm, nm, Nm = fisher_set([0], msrc, mw)
    for pri in ("U", "E"):
        for k in KAPPAS: run(f"both-{msrc}-w{mw:g}-{pri}-k{k}", smooth(Fm, nm, Nm, pri, k))
    bo = pick("both-", ksm)
    choice = {"smooth": sm, "more-q": mq, "more-x": mx, "both": bo}
    say("chosen:", choice)
    json.dump({"choice": choice, "grid": res}, open(f"{O}/tune.json", "w"), indent=1)

def arm_fisher(spec):                                 # tune.json names -> full-qad Fisher (vocab order)
    full = [0, 1]
    if spec == "base": return np.load(f"{HSD}/headsens.npz")["fisher_qad_gen"]
    p = spec.split("-")
    if p[0] == "smooth":
        F, n, N = fisher_set(full); return smooth(F, n, N, p[1], float(p[2][1:]))
    if p[0] == "more":
        F, n, N = fisher_set(full, p[1], float(p[2][1:])); return F / N
    if p[0] == "both":
        F, n, N = fisher_set(full, p[1], float(p[2][1:])); return smooth(F, n, N, p[3], float(p[4][1:]))
    raise KeyError(spec)

def boot(S, a, b, nboot=2000):                        # paired ratio of means a / b − 1, document bootstrap
    u, inv = np.unique(S.doc, return_inverse=True); nd = len(u)
    sa, sb = np.bincount(inv, a), np.bincount(inv, b)
    ix = np.random.default_rng(0).integers(0, nd, (nboot, nd))
    r = sa[ix].sum(1) / sb[ix].sum(1) - 1
    return float(a.sum() / b.sum() - 1), float(r.std()), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], nd

def cmd_final():
    T = json.load(open(f"{O}/tune.json"))["choice"]; S = Sums("search")
    arms = {"base": "base", **T}; rare = S.top >= 20000
    KL, P, out = {}, {}, {"arms": arms, "rows": S.N, "rare_rows": int(rare.sum()), "res": {}}
    for n, a in UNI.items(): KL[n] = S.kl(a)
    for arm, spec in arms.items():
        P[arm] = splans(arm_fisher(spec))
        for n in PLANS3: KL[(arm, n)] = S.kl(P[arm][n])
    for k, a in json.load(open(f"{HSD}/plan.json"))["plans"].items():
        if k in PLANS3: assert (kidx(a["K_blocks"]) == P["base"][k]).all(), f"base {k} differs from headslab3 plan.json"
    useful = {}
    for arm in arms:
        r = {}
        for n in PLANS3:
            u = "u" + n[1]; a = KL[(arm, n)]; bits = float(np.array(KS)[P[arm][n]].mean())
            x = {"bits": bits, "kl": float(a.mean()), "top1": float(S.hit(P[arm][n]).mean()),
                 "rare_kl": float(a[rare].mean()), "vs_uniform": boot(S, a, KL[u])[:3]}
            if arm != "base":
                m, se, ci, nd = boot(S, a, KL[("base", n)]); x["vs_base"] = [m, se, ci]
                x["bits_ok"] = bits <= float(np.array(KS)[P["base"][n]].mean()) + 0.002
                x["pass"] = m <= -0.03 and ci[1] < 0 and x["bits_ok"]
                x["rare_vs_base"] = float(a[rare].mean() / KL[("base", n)][rare].mean() - 1)
                if ci[0] < 0 < ci[1] and m != 0: x["docs_to_settle"] = int(np.ceil(nd * (1.96 * se / abs(m)) ** 2))
            r[n] = x
        if arm != "base":
            r["useful"] = sum(r[n]["pass"] for n in PLANS3) >= 2
            r["geo"] = float(np.exp(np.mean([np.log1p(r[n]["vs_base"][0]) for n in PLANS3])))
            useful[arm] = r["useful"]
        out["res"][arm] = r
    for n, a in UNI.items(): out["res"][n] = {"kl": float(KL[n].mean()), "rare_kl": float(KL[n][rare].mean())}
    base_more = "more-" + T["both"].split("-")[1]               # the more arm that "both" is built on
    out["both_report_only"] = not (useful["smooth"] and useful[base_more])
    cand = [a for a in useful if useful[a] and not (a == "both" and out["both_report_only"])]
    if not cand: cand = [a for a in useful if not (a == "both" and out["both_report_only"])]
    out["best"] = min(cand, key=lambda a: out["res"][a]["geo"]); out["best_useful"] = bool(useful[out["best"]])
    json.dump(out, open(f"{O}/final.json", "w"), indent=1)
    np.savez(f"{O}/final-plans.npz", **{f"{a}:{n}": P[a][n] for a in P for n in PLANS3})
    for arm in arms:
        r = out["res"][arm]
        say(f"{arm:7s} {arms[arm]:24s} " + "  ".join(
            f"{n} {r[n]['kl']:.3e} (vs u {r[n]['vs_uniform'][0]:+.1%}" + (f", vs base {r[n]['vs_base'][0]:+.1%} [{r[n]['vs_base'][2][0]:+.1%}, {r[n]['vs_base'][2][1]:+.1%}]" if arm != "base" else "") + ")"
            for n in PLANS3) + (f"  useful {r['useful']}" if arm != "base" else ""))
    say("best:", out["best"], "useful" if out["best_useful"] else "NOT useful")
    # report-only: the best arm with blocks re-ordered by its block Fisher before the DP
    Fv = arm_fisher(arms[out["best"]]); bf = Fv[perm].reshape(NB, BLK).sum(1); o = np.argsort(-bf, kind="stable")
    C = costs(Fv)[:, o]; ro = {}
    for B in (4, 5, 6):
        kmin = max(hsl.FLOOR, round(B - 1.5 + 1e-9)); a = np.empty(NB, int); a[o] = hsl.slabs(C, KS, kmin, B * NB, NB, 3)
        k = S.kl(a); m, se, ci, _ = boot(S, k, KL[("base", f"s{B}-3")]); ro[f"s{B}-3"] = [m, ci]
        say(f"reorder {out['best']} s{B}-3: {k.mean():.3e} vs base {m:+.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}] (report only)")
    out["reorder_report_only"] = ro; json.dump(out, open(f"{O}/final.json", "w"), indent=1)

def cmd_g1():
    S = Sums("search"); z = np.load(f"{X}/sim-search.npz"); pl = json.load(open(f"{X}/sim-search-plans.json"))
    gen = np.fromfile(f"{W}/in/dump/search/gen.u8", np.uint8).astype(bool); assert (np.nonzero(gen)[0] == S.rows).all()
    names = [str(n) for n in z["names"]]; worst = (1.0, 1.0, "")
    for i, n in enumerate(names):
        a = kidx(pl[n]["K_blocks"]); mine = S.kl(a); ref = z["kl"][i][gen].astype(np.float64)
        ratio = mine.mean() / ref.mean(); cor = np.corrcoef(mine, ref)[0, 1]; hm = (S.hit(a) == z["hit"][i][gen].astype(bool)).mean()
        if abs(ratio - 1) > abs(worst[0] - 1) or cor < worst[1]: worst = (ratio, min(cor, worst[1]), n)
        if n in ("u3", "u5", "u7", "s4-3", "s5-3", "s6-3"): say(f"G1 {n}: ratio {ratio:.5f} corr {cor:.6f} top-1 agree {hm:.4f}")
    lse_ok = np.abs(S.lse - z["lse"][gen]).max(); top_ok = (S.top == z["top"][gen]).mean()
    ok = abs(worst[0] - 1) <= 0.01 and worst[1] >= 0.99
    say(f"G1 over {len(names)} plans: worst ratio {worst[0]:.5f}, worst corr {worst[1]:.6f} ({worst[2]}); lse max diff {lse_ok:.2e}, argmax agree {top_ok:.5f} -> {'PASS' if ok else 'FAIL'}")
    json.dump({"pass": bool(ok), "worst_ratio": worst[0], "worst_corr": worst[1]}, open(f"{O}/g1.json", "w"))

def cmd_census():
    hs = np.load(f"{HSD}/headsens.npz"); F = hs["fisher_qad_gen"]; n = grp(0, 1, key="n")
    S = Sums("search"); Fh = np.zeros(V); Fh[perm] = S.fisher              # exact held-out Fisher, vocab order
    Fg = hs["fisher_search_gen"]
    sets = {"F=0": F == 0, "n<10": n < 10, "n<100": n < 100}
    out = {"qad_gen_positions": int(hs["rows_qad_gen"]), "heldout_rows": S.N}
    for k, m in sets.items():
        out[k] = {"rows": int(m.sum()), "heldout_exact_share": float(Fh[m].sum() / Fh.sum()),
                  "search_pool_golden_share": float(Fg[m].sum() / Fg.sum()), "qad_share": float(F[m].sum() / F.sum())}
        say(f"census {k}: {int(m.sum())} rows; held-out Fisher share {out[k]['heldout_exact_share']:.2%} (exact, search-gen dump), "
            f"{out[k]['search_pool_golden_share']:.2%} (search-pool golden gen), qad share {out[k]['qad_share']:.2%}")
    # rare-token rows of the held-out set: how many have an argmax row with n<10 / n<100
    am = perm[S.top]
    for k in ("n<10", "n<100"): out[f"heldout_argmax_in_{k}"] = float(sets[k][am].mean())
    xF, xn, xN = xc()
    out["xc8k_b1_positions"] = int(xN); out["xc_n<10_rows"] = int((xn < 10).sum()); out["qadx_n<10_rows"] = int((n + xn < 10).sum())
    say(json.dumps(out)); json.dump(out, open(f"{O}/census.json", "w"), indent=1)

def cmd_curve(arm):
    T = json.load(open(f"{O}/tune.json"))["choice"]; spec = "base" if arm == "base" else T[arm]
    hs = np.load(f"{HSD}/headsens.npz"); npz = f"{O}/hs-{arm}.npz"
    np.savez(npz, fisher_arm_gen=arm_fisher(spec), fisher_verify_gen=hs["fisher_verify_gen"], psum_arm_gen=hs["psum_qad_gen"])
    B = ",".join(f"{3 + i / 8:g}" for i in range(1, 33)); C = {}
    for mode, extra in (("curve", {"HS_FLOOR_ALL": "1", "HS_BUDGETS": B}), ("plan", {})):   # w94's set: curve + plan.json's
        cj = f"{O}/{mode}-{arm}.json"
        if not os.path.exists(cj):
            env = {k: v for k, v in os.environ.items() if not k.startswith("HS_")}
            env.update(HS_TRAIN="arm", HS_FLOOR="3", **extra)
            subprocess.run([sys.executable, "-I", f"{CODE}/headslab.py", "plan", npz, f"{HSD}/perm.npy", HSD, cj + ".part"],
                           env=env, check=True, stdout=open(f"{O}/{mode}-{arm}.log", "w"))
            os.replace(cj + ".part", cj)
        C.update(json.load(open(cj))["plans"])
    # w94's sim plan set: its plan.json plans (u4-u6, s*-3/-4, b*) + the curve; here the arm's s*-3 / s*-4 / b* too
    P = {**{n: {"K_blocks": [KS[i] for i in a], "avg_bits": float(np.array(KS)[a].mean())} for n, a in UNI.items()}, **C}
    S = Sums("search"); gen = np.fromfile(f"{W}/in/dump/search/gen.u8", np.uint8).astype(bool); N = len(gen)
    names = sorted(P); kl = np.zeros((len(names), N), np.float32); hit = np.zeros((len(names), N), np.uint8)
    for i, n in enumerate(names):
        a = kidx(P[n]["K_blocks"]); kl[i, gen] = S.kl(a); hit[i, gen] = S.hit(a)
    top = np.zeros(N, np.int64); top[gen] = S.top; lse = np.zeros(N, np.float32); lse[gen] = S.lse
    d = f"{O}/sim-{arm}"; os.makedirs(f"{d}/meta/search", exist_ok=True)
    np.savez(f"{d}/sim-search.npz", names=np.array(names), kl=kl, hit=hit, top=top, lse=lse,
             avg_bits=np.array([P[n]["avg_bits"] for n in names]))
    json.dump({"Ks": KS, "plans": P}, open(f"{d}/plans.json", "w"))
    for f in ("doc.i32", "gen.u8"):
        b = open(f"{W}/in/dump/search/{f}", "rb").read(); open(f"{d}/meta/search/{f}", "wb").write(b)
    say(f"curve {arm}: {len(names)} plans scored on {int(gen.sum())} search-gen rows -> {d}")

if __name__ == "__main__":
    c = sys.argv[1]
    {"g1": cmd_g1, "census": cmd_census, "tune": cmd_tune, "final": cmd_final,
     "curve": lambda: cmd_curve(sys.argv[2])}[c]()
