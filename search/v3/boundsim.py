#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""boundsim.py ROWS.npz TIMING.json V2RESULTS OUTDIR [M] -- continuous early-stop boundary simulation (#166, w126;
rules: RULES-boundary.md, committed before this ran).

Rules: F (full slice), S3 (w123: stages 1/8 -> 1/2 -> 1 as separate runs, reject z >= 0 / 0.5), S1 (the same
checkpoints in one --kld-progress run), B(p, w) (reject once t >= w and z > 2 ((1 - t) / 0.9)^p, every document).
z = paired delta-method statistic over the scored documents (seqrule.py). Two evaluations:
 1 normal model: per-document contributions e = delta + sigma_p eps (sigma_x = X sqrt(Nx), sigma_c = C sqrt(Nc), part
   SEs as protocol.py), pooled estimate w mean_x + (1 - w) mean_c, SE from the scored documents' sample variances;
   common random numbers across rules and deltas; priors as protocol.py (4 normal x s 0.01 / 0.02 / 0.04 + V2's
   candidates). Cost per candidate in card seconds: mix + one run + tau x tokens scored (+ 2 s kill latency; S3: one
   run per stage reached) + the incumbent's full eval / 25.
 2 real documents: 27d-f pairs scaled to delta (protocol.py section 4), drawn with replacement at their own sizes, a
   fresh interleaved order per draw: P(accept) and the fraction scored per rule.
Outputs: OUTDIR/{normal.json, real.json, pick.json}; tables on stdout."""
import json, math, os, sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "perf"))
import seqrule as SR          # noqa: E402
import protocol as PR         # noqa: E402
import slicese as SS          # noqa: E402

KILL_S = 2.0
DPS = 25                      # decisions per slice (rotation)
RULES = [("F", {"kind": "full"}), ("S3", {"kind": "stages", "at": [0.125, 0.5], "z": [0.0, 0.5], "runs": True}),
         ("S1", {"kind": "stages", "at": [0.125, 0.5], "z": [0.0, 0.5]})]
for p in (0.5, 1.0, 2.0):
    for w in (0.05, 0.10, 0.20):
        RULES.append((f"B p{p:g} w{int(round(w * 100))}", {"kind": "curve", "p": p, "warmup": w}))


def bounds(rule, N):
    """reject bound on z after document k = 1..N (INF where no check); stages: checks at ceil(F N)"""
    b = np.full(N, np.inf)
    if rule["kind"] == "curve":
        for k in range(1, N + 1): b[k - 1] = SR.boundary(rule, k / N)
    elif rule["kind"] == "stages":
        for f, z in zip(rule["at"], rule["z"]):
            k = max(1, math.ceil(f * N - 1e-9)); b[k - 1] = min(b[k - 1], z - 1e-12)    # reject iff z >= Z
    return b


def apply(rule, z, fin_ok):
    """z: (M, N) paths; fin_ok: (M,) the whole-slice accept condition. -> accept (M,), docs scored (M,), stopped (M,)"""
    M, N = z.shape; b = bounds(rule, N)
    hit = z > b[None, :]
    any_ = hit.any(1); first = np.where(any_, hit.argmax(1) + 1, N)
    return (~any_) & fin_ok, first, any_


def extra_runs(rule, docs, N):
    if not rule.get("runs"): return np.zeros(len(docs))
    ks = [max(1, math.ceil(f * N - 1e-9)) for f in rule["at"]]
    return sum((docs > k).astype(float) for k in ks)


class Normal:
    def __init__(self, Nx, Nc, M, seed=11):
        self.Nx, self.Nc, self.N = Nx, Nc, Nx + Nc
        self.order = np.array([o == "x" for o in SR.interleave(Nx, Nc)])
        rng = np.random.default_rng(seed); eps = rng.standard_normal((M, self.N))
        ex = np.where(self.order[None, :], eps, 0.0); ec = np.where(~self.order[None, :], eps, 0.0)
        self.kx = np.cumsum(self.order); self.kc = np.cumsum(~self.order)
        kx = np.maximum(self.kx, 1)[None, :]; kc = np.maximum(self.kc, 1)[None, :]
        self.Ex = np.cumsum(ex, 1) / kx; self.Ec = np.cumsum(ec, 1) / kc
        vx = (np.cumsum(ex * ex, 1) - kx * self.Ex ** 2) / np.maximum(kx - 1, 1)
        vc = (np.cumsum(ec * ec, 1) - kc * self.Ec ** 2) / np.maximum(kc - 1, 1)
        self.Vx, self.Vc = vx / kx, vc / kc
        self.valid = (self.kx >= 2) & (self.kc >= 2)

    def paths(self, d, X, C, w):
        sx, sc = X * math.sqrt(self.Nx), C * math.sqrt(self.Nc)
        mx = d + sx * self.Ex; mc = d + sc * self.Ec
        p = w * mx + (1 - w) * mc
        se = np.sqrt(w * w * sx * sx * self.Vx + (1 - w) ** 2 * sc * sc * self.Vc)
        z = np.where(self.valid[None, :], p / np.maximum(se, 1e-15), 0.0)
        fin = (p[:, -1] < 0) & (mx[:, -1] < 0) & (mc[:, -1] < 0)
        return z, fin


def normal_eval(T, SE, w, tok_x, tok_c, priors, M):
    Nx, Nc = int(T["x_docs"]), int(T["calib_docs"]); NM = Normal(Nx, Nc, M)
    tau = T["s_per_tok"]; N = Nx + Nc
    tokcum = np.cumsum(np.where(NM.order, tok_x, tok_c))
    full_cost = T["mix_s"] + T["run_s"] + tau * tokcum[-1]
    out = []
    for pr in priors:
        if pr[0] == "emp":
            d = np.quantile(pr[2], (np.arange(200) + 0.5) / 200); pd = np.full(len(d), 1 / len(d))
            s0, q = pr[3]; svec = np.sqrt(s0 * s0 + (q * d) ** 2); label = pr[1]; mu, sig, sname = pr[1], None, f"{s0:g}+{q:g}|d|"
            cfgs = [(sname, svec)]
        else:
            mu, sig = pr; d = np.linspace(mu - 5 * sig, mu + 5 * sig, 161); pd = norm.pdf(d, mu, sig); pd /= pd.sum()
            cfgs = [(s, np.full(len(d), s)) for s in (0.01, 0.02, 0.04)]; label = f"N({100*mu:g}%,{100*sig:g}%)"
        for sname, svec in cfgs:
            acc = {r: np.zeros(len(d)) for r, _ in RULES}; cost = {r: np.zeros(len(d)) for r, _ in RULES}
            frac = {r: np.zeros(len(d)) for r, _ in RULES}
            for i, (di, si) in enumerate(zip(d, svec)):
                z, fin = NM.paths(di, SE["x"] * si, SE["c"] * si, w)
                for rn, rule in RULES:
                    a, docs, stop = apply(rule, z, fin)
                    c = T["mix_s"] + T["run_s"] * (1 + extra_runs(rule, docs, N)) + tau * tokcum[docs - 1] + KILL_S * (stop & (rule["kind"] != "stages" or not rule.get("runs")))
                    acc[rn][i] = a.mean(); cost[rn][i] = c.mean() + full_cost / DPS; frac[rn][i] = (docs / N).mean()
            E = lambda v: float(np.sum(pd * v))
            oracle = E(np.maximum(-d, 0)); gF = E(acc["F"] * -d)
            for rn, _ in RULES:
                g = E(acc[rn] * -d); c = E(cost[rn])
                out.append({"prior": label, "s": sname if isinstance(sname, str) else float(sname), "rule": rn, "cost_s": c,
                            "p_accept": E(acc[rn]), "gain": g, "gain_frac": g / oracle, "gain_per_h": 100 * g * 3600 / c,
                            "wrong_accept": E(np.where(d > 0, acc[rn], 0)), "good_loss": 1 - g / gF if gF > 0 else None,
                            "p_rej_good": E(np.where(d < 0, 1 - acc[rn], 0)) / max(E((d < 0).astype(float)), 1e-12),
                            "frac": E(frac[rn])})
    return out


def real_eval(P, nd=400, targets=(-0.01, -0.005, 0.005, 0.01)):
    rng = np.random.default_rng(7); out = []
    for (g, a, b), rec in P.items():
        lr = PR.pooled(rec)[0]
        if not 0.01 <= abs(lr) <= 0.15: continue
        Nx, Nc = len(rec["x"][0]), len(rec["c"][0]); N = Nx + Nc
        isx = np.array([o == "x" for o in SR.interleave(Nx, Nc)])
        for tg in targets:
            k = tg / lr
            R = {p: (A, np.where(A > 0, A * (B / np.where(A > 0, A, 1)) ** k, B)) for p, (A, B, _, _) in rec.items()}
            (Ax, Bx), (Ac, Bc) = R["x"], R["c"]
            true = math.log((Bx.sum() + Bc.sum()) / (Ax.sum() + Ac.sum()))
            ix = rng.integers(0, Nx, (nd, Nx)); ic = rng.integers(0, Nc, (nd, Nc))
            A = np.empty((nd, N)); B = np.empty((nd, N))
            A[:, isx] = Ax[ix]; B[:, isx] = Bx[ix]; A[:, ~isx] = Ac[ic]; B[:, ~isx] = Bc[ic]
            kk = np.arange(1, N + 1)[None, :]
            SA, SB = np.cumsum(A, 1), np.cumsum(B, 1)
            SAA, SBB, SAB = np.cumsum(A * A, 1), np.cumsum(B * B, 1), np.cumsum(A * B, 1)
            mA, mB = SA / kk, SB / kk
            su2 = SBB / mB ** 2 - 2 * SAB / (mA * mB) + SAA / mA ** 2
            se = np.sqrt(np.maximum(su2, 0) / np.maximum(kk - 1, 1)) / np.sqrt(kk)
            dd = np.log(SB / SA)
            z = np.where(kk >= 3, dd / np.maximum(se, 1e-15), 0.0)
            fin = (dd[:, -1] < 0) & (Bx[ix].sum(1) < Ax[ix].sum(1)) & (Bc[ic].sum(1) < Ac[ic].sum(1))
            for rn, rule in RULES:
                acc, docs, stop = apply(rule, z, fin)
                out.append({"golden": g, "a": a, "b": b, "target": tg, "true": true, "rule": rn, "p_accept": float(acc.mean()),
                            "frac": float((docs / N).mean())})
    return out


def pick(NR, RR):
    """RULES-boundary.md 'Pick'"""
    rules = [r for r, _ in RULES]
    key = lambda r: (r["prior"], str(r["s"]))
    base = {key(r): r for r in NR if r["rule"] == "F"}
    sel = lambda r: (r["s"] == "0.01+2|d|") or (isinstance(r["s"], float) and r["s"] <= 0.04)
    stat = {}
    for rn in rules:
        v = [r for r in NR if r["rule"] == rn and sel(r)]
        ratio = np.array([r["gain_per_h"] / base[key(r)]["gain_per_h"] for r in v])
        wa_ok = all(r["wrong_accept"] <= base[key(r)]["wrong_accept"] + 1e-9 for r in v)
        gl = np.array([r["good_loss"] for r in v])
        real = {tg: float(np.mean([r["p_accept"] for r in RR if r["rule"] == rn and r["target"] == tg])) for tg in (-0.01, -0.005, 0.005, 0.01)}
        stat[rn] = {"median": float(np.median(ratio)), "min": float(ratio.min()), "n_priors": len(v), "wa_ok": wa_ok,
                    "good_loss_median": float(np.median(gl)), "real_p_accept": real,
                    "frac_median": float(np.median([r["frac"] for r in v]))}
    s3 = stat["S3"]
    elig = [rn for rn in rules if stat[rn]["wa_ok"] and all(stat[rn]["real_p_accept"][tg] <= s3["real_p_accept"][tg] + 0.005 for tg in (0.005, 0.01))]
    order = sorted(elig, key=lambda rn: -stat[rn]["median"])
    guarded = [rn for rn in order if stat[rn]["min"] >= s3["min"] - 0.05]
    if not guarded: return stat, None, elig, []
    best = stat[guarded[0]]["median"]
    tied = [rn for rn in guarded if stat[rn]["median"] >= best * 0.98]
    pk = min(tied, key=lambda rn: stat[rn]["good_loss_median"])
    return stat, pk, elig, tied


def main():
    npz, timing, v2, od = sys.argv[1:5]; M = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
    os.makedirs(od, exist_ok=True)
    G = SS.load(npz); T = json.load(open(timing)); T["runs_per_eval"] = 1
    P = PR.two_part(G)
    rx, W, tx = [], [], []
    for (g, a, b), rec in P.items():
        lr = PR.pooled(rec)[0]
        W.append(rec["x"][0].sum() / (rec["x"][0].sum() + rec["c"][0].sum())); tx.append(rec["x"][2].mean())
        if not 0.005 <= abs(lr) <= 0.15: continue
        sx = PR.boot_se(*rec["x"][:2]) * math.sqrt(len(rec["x"][0])); sc = PR.boot_se(*rec["c"][:2]) * math.sqrt(len(rec["c"][0]))
        rx.append(sx / sc)
    RX = float(np.median(rx)); w = float(np.median(W)); tok_x = min(float(np.median(tx)), T.get("x_tok_cap", 1e9))
    tok_c = T["calib_tok"] / T["calib_docs"]
    SE = {"x": RX / math.sqrt(T["x_docs"]), "c": 1 / math.sqrt(T["calib_docs"])}
    print(f"boundsim: {len(P)} two-part pairs; RX {RX:.2f}, x KL share w {w:.2f}, tokens / doc x {tok_x:.0f} c {tok_c:.0f}; M {M}")
    priors = [(0.005, 0.01), (0.0, 0.01), (0.005, 0.005), (0.01, 0.02)]
    for name, smp in PR.v2_moves(v2).items():
        if name == "V2 all": priors.append(("emp", name, smp, (0.01, 2.0)))
    NR = normal_eval(T, SE, w, tok_x, tok_c, priors, M)
    json.dump(NR, open(f"{od}/normal.json", "w"), indent=0)
    RR = real_eval(P)
    json.dump(RR, open(f"{od}/real.json", "w"), indent=0)
    stat, pk, elig, tied = pick(NR, RR)
    json.dump({"stat": stat, "pick": pk, "eligible": elig, "tied": tied, "RX": RX, "w": w, "SE": SE, "M": M}, open(f"{od}/pick.json", "w"), indent=1)
    print(f"== per rule over {stat['F']['n_priors']} priors (normal s <= 0.04 x 4 + V2): gain/h vs F (median, min), good-move loss, "
          "slice fraction scored; real documents P(accept) at delta -1 / -0.5 / +0.5 / +1 %")
    for rn, _ in RULES:
        s = stat[rn]; ra = s["real_p_accept"]
        print(f"  {rn:12s} gain/h x{s['median']:.3f} (min {s['min']:.3f})  good-loss {s['good_loss_median']:+.3f}  frac {s['frac_median']:.3f}"
              f"  real P(acc) {ra[-0.01]:.3f} {ra[-0.005]:.3f} {ra[0.005]:.4f} {ra[0.01]:.4f}  {'eligible' if rn in elig else 'NOT eligible'}")
    print(f"== tied (within 2% of the best guarded median): {tied}")
    print(f"== PICK: {pk}")
    print("== per prior (gain/h vs F) for F / S3 / S1 / pick")
    key = lambda r: (r["prior"], str(r["s"]))
    base = {key(r): r for r in NR if r["rule"] == "F"}
    for r in NR:
        if r["rule"] in ("S3", "S1", pk):
            b = base[key(r)]
            print(f"  {r['prior']:12s} s {str(r['s']):10s} {r['rule']:12s} x{r['gain_per_h']/b['gain_per_h']:.3f}  wrong acc {r['wrong_accept']:.4f} (F {b['wrong_accept']:.4f})"
                  f"  good-loss {r['good_loss']:+.3f}  P(rej|good) {r['p_rej_good']:.3f} (F {b['p_rej_good']:.3f})  cost {r['cost_s']:.0f} s (F {b['cost_s']:.0f})")


if __name__ == "__main__":
    main()
