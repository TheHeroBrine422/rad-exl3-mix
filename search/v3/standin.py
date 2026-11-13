# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""standin.py WORKDIR [--layers 16] [--results 400] [--seed 0] -- CPU stand-in end-to-end run of the search driver
(w126, #166, task "Tests"): stub evals with synthetic per-document KL from a KNOWN model; the driver must climb, its fit
must recover the model, early stops must fire. Exit 0 iff every check passes.

Truth: the menu's tensors (menu.synth) with kfit3's model form -- cost_w = s_w err_w^q, s_w = exp(group x depth curve +
0.5 log hbar_w + N(0, 0.15)) (the per-tensor part is partly visible to the fit through hbar, partly not), depth
compounding kappa, head cost, embedding offset. Slices: 130 x + 50 c documents, fresh per slice; a document's KL under a
mix = a_d n_d sum_b v_db share_b(mix) KL(mix), share_b = the block's share of the cost (block = group x 8-layer band,
head), v_db lognormal (x 0.30, c 0.15): moves that shift cost between blocks get document-level noise, as real ones do.
The fake executor reads the driver's queue (evals.jsonl), scores up to --par evals per step (in queue order), applies
the early-stop block exactly as the hosts will (seqrule.Paired on the documents in order, stop at the boundary),
writes rxrun.multi's results.jsonl + docs/<tag>-<slice>.json, and books card seconds (mix + run + tokens x tau).
Checks: (1) every size's true KLD falls (driver's incumbent vs its first); (2) wrong accepts (truly worse by > 0.5%)
<= 5% of accepts; (3) early stops fired (>= 30% of the rejects stopped early) and saved card time; (4) the fit recovers
the model: on 300 random one-step moves around the final incumbents, corr(predicted, true d log KL) >= 0.8 and the
fitted log-ratio error RMSE <= 0.03; (5) a rotation happened and every incumbent was re-scored on the new slice;
(6) a restarted driver (state replay) sees the same incumbents and continues.
--real (w127, #167; realistic byte jitter): the real ladder's K grid (0.125 steps to K5, 0.25 above; from K1), the real
menu's per-tensor overhead, the base container's own head / embedding (H4 / E4: donor-free mixes, so the driver
calibrates its byte offset as on box2) and every container w126's measured offset larger than its menu bytes
(8,255,936 B on the 27B = 0.00246 bpw, kept in bpw); --size-adjust 0|1 = the driver's CONF size_adjust (the A/B arms);
--traj FILE: the incumbents' TRUE KL / container bpw per eval + card-hours, the true optimum's slope b_true and log KL
at each size (TruthPred: the truth through moves.mckp) for abreport.py."""
import argparse, json, math, os, random, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import driver as DR          # noqa: E402
import menu as MN            # noqa: E402
import moves as MV           # noqa: E402
import seqrule as SR         # noqa: E402
import slices as SL          # noqa: E402

TAU, MIX_S, RUN_S, KILL_S = 0.000778, 11.2, 7.7, 2.0


class Truth:
    def __init__(self, M, seed):
        rng = np.random.default_rng(seed); self.M = M
        G = MN.GROUPS; NL = M.NL
        gl = {g: rng.normal(0, 0.4) for g in G}; dep = {g: rng.normal(0, 0.3, 3) for g in G}
        self.s, self.L, self.blk = {}, {}, {}
        for t in M.T:
            w = t["w"]
            if w == MN.EMB: continue
            if w == MN.HEAD: ls = math.log(0.5); b = ("head", 0)
            else:
                g = t["group"]; x = t["layer"] / max(NL - 1, 1); d = np.interp(x, [0, 0.5, 1], dep[g])
                ls = gl[g] + d + 0.5 * math.log(t["hbar"]) + rng.normal(0, 0.15); b = (g, t["layer"] // 8)
            self.s[w] = math.exp(ls); self.L[w] = t["layer"]; self.blk[w] = b
        self.blocks = sorted(set(self.blk.values()), key=str)
        self.q, self.kap, self.c0 = 1.1, 0.4, 0.004
        self.eb = {t["w"]: t["opts"][len(t["opts"]) // 2]["err"] for t in M.T if t["w"] != MN.EMB}
        k0 = sum(self.s[w] * self.eb[w] for w in self.s)
        self.scale = 0.05 / k0                                         # a mid-menu (K3.5) mix ~ KL 0.06
        self.emb = {o["o"]: 0.004 * 2 ** (-(o["K"] - 3)) for o in M.T[M.WI[MN.EMB]]["opts"]}

    def costs(self, a):
        return {w: self.scale * self.s[w] * self.eb[w] * (self.M.opt(w, a[w])["err"] / self.eb[w]) ** self.q for w in self.s}

    def kl(self, a, parts=False):
        c = self.costs(a); NL = self.M.NL; X = np.zeros(NL + 1)
        for w, v in c.items(): X[self.L[w]] += v
        P = self.c0 + self.emb[a[MN.EMB]]
        for l in range(NL + 1): P = P * (1 + self.kap * X[l]) + X[l]
        if not parts: return P
        tot = sum(c.values()); sh = {b: 0.0 for b in self.blocks}
        for w, v in c.items(): sh[self.blk[w]] += v / tot
        return P, sh


def make_slice(name, rng, truth, nx=130, nc=50):
    docs = []
    for p, n, tok, sv in [("x", nx, 1400, 0.30), ("c", nc, 900, 0.15)]:
        for i in range(n):
            docs.append({"part": p, "src": i, "orig": f"{name}-{p}{i}", "source": "synth", "tokens": tok, "score_from": 0,
                         "a": float(rng.lognormal(0, 0.8)), "v": {str(b): float(rng.lognormal(-sv * sv / 2, sv)) for b in truth.blocks}})
    xs = [d for d in docs if d["part"] == "x"]; cs = [d for d in docs if d["part"] == "c"]
    rng.shuffle(xs); rng.shuffle(cs)
    pat = SL.interleave(len(xs), len(cs)); ix = iter(xs); ic = iter(cs)
    return [next(ix) if p == "x" else next(ic) for p in pat]


class TruthPred:
    """the truth with moves.Pred's interface (mckp / repair / lin): the true knapsack optimum per size"""

    def __init__(self, T):
        self.T = T; M = T.M; self.M = M
        self.W = [t["w"] for t in M.T if t["w"] != MN.EMB]
        self.D = type("D", (), {"W": self.W})()
        self.C, self.B, self.names = [], [], []
        for w in self.W:
            t = M.T[M.WI[w]]; eb = T.eb[w]
            self.C.append(np.array([T.scale * T.s[w] * eb * (o["err"] / eb) ** T.q for o in t["opts"]]))
            self.B.append(np.array([o["bytes"] for o in t["opts"]], float)); self.names.append([o["o"] for o in t["opts"]])
        self.L = [T.L[w] for w in self.W]; self.NX = M.NL + 1
        self.emb = M.T[M.WI[MN.EMB]]

    def idx(self, a): return [self.M.oi[self.M.WI[w]][a[w]] for w in self.W]

    def X(self, J):
        X = np.zeros(self.NX)
        for i, j in enumerate(J): X[self.L[i]] += self.C[i][j]
        return X

    def P(self, X, a):
        P = [self.T.c0 + self.T.emb[a[MN.EMB]]]
        for l in range(self.NX): P.append(P[-1] * (1 + self.T.kap * X[l]) + X[l])
        return P

    def kl(self, a): return self.P(self.X(self.idx(a)), a)[-1]

    def lin(self, a):
        J = self.idx(a); X = self.X(J); P = self.P(X, a); g = np.zeros(self.NX); acc = 1.0
        for l in range(self.NX - 1, -1, -1):
            g[l] = acc * (P[l] * self.T.kap + 1); acc *= (1 + self.T.kap * X[l])
        return J, [g[self.L[i]] * (self.C[i] - self.C[i][J[i]]) / P[-1] for i in range(len(J))]

    def assign(self, J, a0):
        a = dict(a0)
        for i, w in enumerate(self.W): a[w] = self.names[i][J[i]]
        return a


def true_frontier(T, sizes, tol, off_bytes, delta=0.1):
    """per size: the truth's knapsack optimum under the container cap (menu bytes <= size - offset): log KL*, and
    b_true = -d log KL* / d bpw (central difference at size -/+ delta, container bpw)"""
    TP = TruthPred(T); M = T.M; lv = M.levels(); out = {}

    def opt(s):
        hi = s * M.P / 8 - off_bytes; lo = (s - tol) * M.P / 8 - off_bytes
        best = None
        for n in lv[::6] + lv[-1:]:                                     # several starts: mckp is a local Lagrangian
            m = MV.mckp(TP, M.uniform(n), hi, lo)
            if m is None or M.bytes(m) > hi: continue
            k = TP.kl(m)
            if best is None or k < best[0]: best = (k, m)
        return best

    for s in sizes:
        o = opt(s); lo_ = opt(s - delta); hi_ = opt(s + delta)
        pts = [(M.bpw(x[1]) + off_bytes * 8 / M.P, math.log(x[0])) for x in (lo_, o, hi_) if x is not None]
        b = -(pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]) if len(pts) >= 2 and pts[-1][0] > pts[0][0] else None
        out[s] = {"logkl_opt": math.log(o[0]) if o else None, "bpw_opt": (M.bpw(o[1]) + off_bytes * 8 / M.P) if o else None, "b_true": b}
    return out


class FakeExec:
    def __init__(self, root, truth, slices, par, off_bytes=0, base=None):
        self.root, self.T, self.S, self.par = root, truth, slices, par; self.off_bytes = off_bytes; self.base = base or {}
        self.off = 0; self.card_s = 0.0; self.card_s_full = 0.0; self.n = 0; self.stopped = 0
        os.makedirs(f"{root}/docs", exist_ok=True)

    def step(self, M):
        p = f"{self.root}/evals.jsonl"
        if not os.path.exists(p): return 0
        done = 0
        with open(p) as f:
            f.seek(self.off)
            while done < self.par:
                l = f.readline()
                if not l: break
                self.off = f.tell(); e = json.loads(l); self.score(e, M); done += 1
        return done

    def score(self, e, M):
        a = {}
        for w, o in e["mix"]["levels"].items(): a[w] = o
        for w, path in e["mix"]["donors"].items():
            t = M.T[M.WI[w]]; a[w] = next(o["o"] for o in t["opts"] if (o.get("src") or {}).get("path") == path)
        for w, o in self.base.items(): a.setdefault(w, o)                # no donor: the base container's own tensor
        sl = e["slices"][0]["name"]; docs = self.S[sl]
        P, sh = self.T.kl(a, parts=True)
        sums = [d["a"] * (d["tokens"] - 1) * P * sum(d["v"][str(b)] * s for b, s in sh.items()) for d in docs]
        npos = [d["tokens"] - 1 for d in docs]
        out = [[s, n] for s, n in zip(sums, npos)]; stopped = None
        es = e.get("early_stop")
        if es:
            Pd = SR.Paired(len(docs), es["rule"])
            for i, ((ra, rn), b) in enumerate(zip(es["ref"], sums)):
                Pd.add(ra, b, es["parts"][i])
                if Pd.check():
                    stopped = {"k": Pd.k, "z": Pd.z(), "d": Pd.d()}; out = out[:Pd.k] + [None] * (len(docs) - Pd.k); break
        k = sum(1 for x in out if x is not None)
        tok = sum(d["tokens"] for d, x in zip(docs, out) if x is not None)
        self.card_s += MIX_S + RUN_S + TAU * tok + (KILL_S if stopped else 0)
        self.card_s_full += MIX_S + RUN_S + TAU * sum(d["tokens"] for d in docs)
        self.n += 1; self.stopped += stopped is not None
        json.dump({"m": out}, open(f"{self.root}/docs/{e['tag']}-{sl}.json", "w"))
        b = M.bytes(a) + self.off_bytes
        kl = sum(x[0] for x in out if x) / sum(x[1] for x in out if x)
        r = {"tag": e["tag"], "slice": sl, "bytes": b, "bpw": b * 8 / M.P, "positions": sum(x[1] for x in out if x), "kld": kl}
        if stopped: r["stopped"] = stopped
        with open(f"{self.root}/results.jsonl", "a") as f: f.write(json.dumps(r) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="standin.py")
    ap.add_argument("work"); ap.add_argument("--layers", type=int, default=16); ap.add_argument("--results", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--par", type=int, default=2)
    ap.add_argument("--sizes", default="2.0,3.0,4.5"); ap.add_argument("--rotate-after", type=int, default=25)
    ap.add_argument("--groups", default=None, help="fit groups NAME:SIZE,SIZE:all|own;... (driver CONF fit_groups)")
    ap.add_argument("--flat-n", type=int, default=6)
    ap.add_argument("--real", action="store_true", help="realistic byte jitter (w127): real K grid + overheads, base head / emb, container offset")
    ap.add_argument("--size-adjust", type=int, default=1); ap.add_argument("--traj"); ap.add_argument("--slices", type=int, default=6)
    ap.add_argument("--size-tol", type=float, default=None); ap.add_argument("--offset-bpw", type=float, default=8255936 * 8 / MN.PARAMS)
    a = ap.parse_args(argv)
    W = a.work; shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    if a.real:
        levels = tuple(float(k) for k in np.r_[np.arange(1.0, 5.0, 0.125), np.arange(5.0, 7.51, 0.25)])
        md = MN.synth(n_layers=a.layers, levels=levels, seed=a.seed, heads=(3.0, 4.0, 5.0, 6.0, 7.0), embs=(3.0, 4.0, 5.0, 6.0),
                      real_bytes=True, base_head=4.0, base_emb=4.0)
    else:
        levels = tuple(np.round(np.arange(1.0, 6.01, 0.25), 3))
        md = MN.synth(n_layers=a.layers, levels=levels, seed=a.seed)
    json.dump(md, open(f"{W}/menu.json", "w"))
    M = MN.Menu(md); T = Truth(M, a.seed + 1)
    off_bytes = int(round(a.offset_bpw * M.P / 8)) if a.real else 0
    base = {MN.HEAD: "H4", MN.EMB: "E4"} if a.real else {}
    rng = np.random.default_rng(a.seed + 2)
    reg = SL.Registry(f"{W}/registry.json"); S = {}
    for i in range(a.slices):
        name = f"syn{i}"; docs = make_slice(name, rng, T); S[name] = docs
        man = {"docs": [dict(d, row0=0) for d in docs]}
        reg.add(name, f"/synthetic/{name}", parts={"x": "synx", "c": "sync"}, man=man)
        reg.d["slices"][name]["built"] = f"2026-11-12T00:{i // 60:02d}:{i % 60:02d}Z"; reg.save()
    S["ver"] = make_slice("ver", rng, T)                               # the held-out monitor (not in the registry: never rotated)
    sizes = [float(x) for x in a.sizes.split(",")]
    conf = {"root": f"{W}/run", "menu": f"{W}/menu.json", "registry": f"{W}/registry.json", "base": "BASE", "sizes": sizes,
            "size_tol": a.size_tol if a.size_tol is not None else (0.02 if a.real else 0.03), "size_adjust": bool(a.size_adjust), "evals": f"{W}/evals.jsonl", "results": f"{W}/results.jsonl", "docs": f"{W}/docs",
            "queue_depth": 4, "refit_every": 15, "rotate_after": a.rotate_after, "audit_frac": 0.1, "seed": a.seed, "dry": True,
            "init_head": "H4", "init_emb": "E4", "ndraw": 8, "poll_s": 0, "fit_opts": {"beta": "fit"},
            "monitor": {"name": "ver", "golden": "/synthetic/ver", "flat_n": a.flat_n}}
    if a.groups:
        conf["fit_groups"] = [{"name": n, "sizes": [float(x) for x in z.split(",")], "rows": r} for n, z, r in (g.split(":") for g in a.groups.split(";"))]
    json.dump(conf, open(f"{W}/conf.json", "w"), indent=1)
    D = DR.Driver(f"{W}/conf.json"); X = FakeExec(W, T, S, a.par, off_bytes, base)
    first = {}; steps = 0; restarted = False
    TF = true_frontier(T, sizes, conf["size_tol"], off_bytes) if a.traj else None
    tj = open(a.traj, "w") if a.traj else None
    if tj: tj.write(json.dumps({"ev": "truth", "seed": a.seed, "size_adjust": bool(a.size_adjust), "off_bytes": off_bytes, "P": M.P,
                                "frontier": {str(k): v for k, v in TF.items()}}) + "\n")
    last = {}
    while X.n < a.results and steps < 20 * a.results:
        D.step(); X.step(M); steps += 1
        if tj:                                                         # the incumbents' truth after every executor step
            for s in sizes:
                t = D.inc.get(s)
                if t not in D.res or D.sub[t]["kind"] == "monitor": continue
                ai = D.assign(t); key = (t, X.n)
                if last.get(s) == key: continue
                last[s] = key
                tj.write(json.dumps({"ev": "inc", "n": X.n, "card_h": X.card_s / 3600, "size": s, "tag": t, "kl": T.kl(ai),
                                     "bpw": (M.bytes(ai) + off_bytes) * 8 / M.P, "b": D.b_of(s), "dec": D.dec[s]}) + "\n")
        for fp in D.fitprocs.values():                                 # stand-in: fits finish before the next step
            if fp: fp[0].wait()
        for s in sizes:
            t = D.inc.get(s)
            if s not in first and t in D.res: first[s] = T.kl(D.assign(t))
        if not restarted and X.n >= a.results // 2:                   # (6) restart mid-run
            inc0 = dict(D.inc); D = DR.Driver(f"{W}/conf.json"); restarted = D.inc == inc0
            if not restarted: print(f"RESTART mismatch {inc0} vs {D.inc}")
    D.ingest()
    for fp in D.fitprocs.values():
        if fp: fp[0].wait()
    D.maybe_refit()
    print(D.report())
    ok = True

    def chk(name, cond, msg):
        nonlocal ok
        ok &= bool(cond); print(f"  {'PASS' if cond else 'FAIL'} {name}: {msg}")
    # (1) climb
    gains = {}
    sizes = [s for s in sizes if D.feasible(s)]                        # a size below the menu is skipped by the driver
    for s in sizes:
        t = D.inc[s]; a1 = D.assign(t); gains[s] = math.log(T.kl(a1) / first[s])
    chk("climb", all(g < -0.02 for g in gains.values()), " ".join(f"{s}: {100 * g:+.1f}%" for s, g in gains.items()))
    # (2) wrong accepts
    acc = [json.loads(l) for l in open(f"{W}/run/state.jsonl") if '"ev": "incumbent"' in l]
    wrong = n_acc = 0
    for e in acc:
        if not e.get("prev") or D.sub.get(e["tag"], {}).get("kind") in ("incumbent", None): continue
        n_acc += 1; d = math.log(T.kl(D.assign(e["tag"])) / T.kl(D.assign(e["prev"])))
        wrong += d > 0.005
    chk("wrong accepts", n_acc > 0 and wrong <= max(1, 0.05 * n_acc), f"{wrong} of {n_acc} accepts truly worse by > 0.5%")
    # (3) early stops
    dec = [json.loads(l) for l in open(f"{W}/run/state.jsonl") if '"ev": "decision"' in l]
    rej = [e for e in dec if e["decision"] == "reject"]; st = sum(1 for e in rej if D.res.get(e["tag"], {}).get("stopped"))
    chk("early stops", len(rej) > 0 and st >= 0.3 * len(rej) and X.card_s < X.card_s_full,
        f"{st} of {len(rej)} rejects stopped early; card time {X.card_s / 3600:.2f} h vs {X.card_s_full / 3600:.2f} h all-full ({X.n} evals)")
    # (4) fit recovery
    rr = random.Random(5); dp, dt = [], []
    for _ in range(300):
        s = rr.choice(sizes); a0 = D.assign(D.inc[s]); a1 = dict(a0); pr = D.fits[D.gname[s]]["pred"]
        w = rr.choice([w for w in a0 if w != MN.EMB]); t = M.T[M.WI[w]]; j = M.oi[M.WI[w]][a0[w]]
        j2 = min(len(t["opts"]) - 1, max(0, j + rr.choice([-2, -1, 1, 2])))
        if j2 == j: continue
        a1[w] = t["opts"][j2]["o"]
        dp.append(math.log(pr.kl(a1) / pr.kl(a0))); dt.append(math.log(T.kl(a1) / T.kl(a0)))
    dp, dt = np.array(dp), np.array(dt); corr = float(np.corrcoef(dp, dt)[0, 1]); rm = float(np.sqrt(np.mean((dp - dt) ** 2)))
    chk("fit recovers the model", corr >= 0.8 and rm <= 0.03, f"one-step moves: corr {corr:.3f}, RMSE {rm:.4f} (true sd {dt.std():.4f}); fits " + str({g: f["report"]["rows"] for g, f in D.fits.items()}))
    # (5) rotation
    rot = [json.loads(l) for l in open(f"{W}/run/state.jsonl") if '"ev": "rotate"' in l and '"from": null' not in l]
    rescored = all(D.res.get(D.inc[s], {}).get("slice") == D.reg.active() or D.inc[s] in D.outstanding() for s in sizes)
    chk("rotation", len(rot) >= 1 and rescored, f"{len(rot)} rotations, active {D.reg.active()}, incumbents on it or in flight: {rescored}")
    chk("restart", restarted, "driver rebuilt from state.jsonl mid-run: same incumbents")
    # (7) held-out monitor: every accept read once on "ver"; the curve follows the true KL; never a fit row
    mons = [json.loads(l) for l in open(f"{W}/run/state.jsonl") if '"ev": "monitor"' in l]
    nonfit = all(not o["tag"].split("@")[0].split("-")[2] == "mon" for o in D.obs.values())
    tr = [(math.log(m["kld"]), math.log(T.kl(D.assign(m["tag"])))) for m in mons]
    cm = float(np.corrcoef(*zip(*tr))[0, 1]) if len(tr) > 2 else float("nan")
    chk("monitor", len(mons) >= n_acc - 2 and nonfit and cm > 0.9, f"{len(mons)} monitor reads for {n_acc} accepts, corr with true KL {cm:.3f}, "
        f"none fitted: {nonfit}; sizes stopped by the flat rule: {sorted(D.stopped_sizes)}")
    bias = D.bias
    print(f"  info: audit runs stopped-after-the-fact {len(bias)}; mean prefix - full log r {np.mean([b['d_prefix'] - b['d_full'] for b in bias]) if bias else float('nan'):+.4f}")
    if tj:
        for e in acc:                                                  # every accept: its true change (raw and size-adjusted)
            if not e.get("prev") or D.sub.get(e["tag"], {}).get("kind") in ("incumbent", None): continue
            a1, a0 = D.assign(e["tag"]), D.assign(e["prev"])
            tj.write(json.dumps({"ev": "accept", "size": e["size"], "tag": e["tag"], "dlogkl": math.log(T.kl(a1) / T.kl(a0)),
                                 "dbpw": (M.bytes(a1) - M.bytes(a0)) * 8 / M.P}) + "\n")
        tj.write(json.dumps({"ev": "end", "evals": X.n, "card_h": X.card_s / 3600, "pass": bool(ok), "b": {str(s): D.b_of(s) for s in sizes},
                             "bytes_off": D.bytes_off, "accepts": n_acc, "rejects": len(rej), "stopped": st}) + "\n"); tj.close()
    print("STANDIN " + ("PASS" if ok else "FAIL"))
    json.dump({"gains": gains, "wrong": wrong, "accepts": n_acc, "rejects": len(rej), "stopped": st, "card_h": X.card_s / 3600,
               "card_h_full": X.card_s_full / 3600, "corr": corr, "rmse": rm, "rotations": len(rot), "evals": X.n, "pass": ok},
              open(f"{W}/standin.json", "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
