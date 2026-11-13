# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""moves.py -- candidate generation for one size of the V3 search (w126, #166; V2's kinds: search/auto/streamq.py).

Given the size's incumbent assignment, its byte window [lo, hi] and the fitted model (fit.SModel + parameters), each
kind proposes one new assignment inside the window:
  solve     the model's optimum at hi (Lagrangian multiple-choice knapsack on the costs linearised at the current
            solution, re-linearised 3x; exact compounding for the result) -- V2's mean mix
  partial   the incumbent moved part of the way to `solve`: the changes with the best predicted gain per byte, a random
            25-75% of them, bytes repaired to the window (a trust region: the model's optimum is extrapolation)
  pair      local search: one one-step upgrade (one tensor, by predicted gain per byte among the top 12, random) paid for
            by the cheapest one-step downgrades elsewhere -- V2's ls
  block     the same with a whole group x 8-layer band moved one step up and bands moved down (measurable, V2's swaps)
  thompson  `solve` under a posterior draw of the parameters (Laplace) -- V2's ts
  explore   uncertainty: of 24 random pair / block moves, the one with the best UCB (predicted gain + 1 sd over 12
            posterior draws); or a random band move with no model ranking (half the time) -- V2's tl / rd. While some
            embedding option has no row yet, half the explore moves try the unmeasured one nearest the incumbent's,
            bytes repaired (w127: the fit gives an unseen embedding no gain, so no other kind would ever measure it;
            the stand-in's 6.0 search stayed on E4 at +17% true KL vs the optimum's E6)
Bytes come from the menu (heads: the quantized head's bytes). Candidates equal to the incumbent or to any assignment
already evaluated (hash) are skipped; generate() returns None after a few tries."""
import hashlib, json, math, random

import numpy as np

import menu as MN

KINDS = ["solve", "partial", "pair", "block", "thompson", "explore"]
SHARE0 = {"solve": 0.12, "partial": 0.18, "pair": 0.25, "block": 0.18, "thompson": 0.15, "explore": 0.12}


def ahash(a): return hashlib.sha1(json.dumps(sorted(a.items())).encode()).hexdigest()[:16]


class Pred:
    """fast predictions of one parameter vector over the menu: cost table C[i][j] per (tensor, option)"""

    def __init__(self, Mo, x):
        self.Mo, self.x = Mo, x; self.tb = Mo.tables(x); D = Mo.D; self.D = D; M = D.M; self.M = M
        self.C = []; self.B = []; self.names = []
        for i, w in enumerate(D.W):
            t = M.T[M.WI[w]]
            self.C.append(np.array([Mo.cost(self.tb, i, o["err"], o.get("fam", "v3")) for o in t["opts"]]))
            self.B.append(np.array([o["bytes"] for o in t["opts"]], float)); self.names.append([o["o"] for o in t["opts"]])
        self.L = D.layer; self.NX = D.NL + 1
        self.emb = M.T[M.WI[MN.EMB]] if MN.EMB in M.WI else None

    def idx(self, a): return [self.M.oi[self.M.WI[w]][a[w]] for w in self.D.W]

    def X(self, J):
        X = np.zeros(self.NX)
        for i, j in enumerate(J): X[self.L[i]] += self.C[i][j]
        return X

    def kl(self, a):
        return self.Mo.kl_from_X(self.tb, self.X(self.idx(a)), self.D.emb_label(a))

    def lin(self, a):
        """per tensor: linearised d log KL of every option vs the current one"""
        J = self.idx(a); X = self.X(J); g, P = self.Mo.grad_X(self.tb, X, self.D.emb_label(a))
        return J, [g[self.L[i]] * (self.C[i] - self.C[i][J[i]]) / P for i in range(len(J))]

    def assign(self, J, a0):
        a = dict(a0)
        for i, w in enumerate(self.D.W): a[w] = self.names[i][J[i]]
        return a


def mckp(pred, a, hi, lo, iters=3):
    """the model's optimum at bytes <= hi: Lagrangian multiple-choice knapsack over trunk + head on the costs
    linearised at the current solution (re-linearised `iters` times), for every embedding option (exact prediction
    picks the best), bytes repaired into [lo, hi]"""
    embs = [o["o"] for o in pred.emb["opts"]] if pred.emb and len(pred.emb["opts"]) > 1 else [a.get(MN.EMB)]
    best = None
    for eo in embs:
        cur = dict(a)
        if eo is not None: cur[MN.EMB] = eo
        cur = _mckp1(pred, cur, hi, lo, iters)
        if cur is None: continue
        k = pred.kl(cur); inw = lo <= pred.M.bytes(cur) <= hi
        if best is None or (inw, -k) > (best[2], -best[0]): best = (k, cur, inw)      # in the window first, then predicted KL
    return best[1] if best else None


def _mckp1(pred, a, hi, lo, iters):
    M = pred.M; fixed = M.bytes(a) - sum(pred.B[i][j] for i, j in enumerate(pred.idx(a)))
    cur = dict(a)
    for _ in range(iters):
        J0, dl = pred.lin(cur)
        def pick(lam):
            J = [int(np.argmin(dl[i] + lam * pred.B[i])) for i in range(len(dl))]
            return J, fixed + sum(pred.B[i][j] for i, j in enumerate(J))
        J, b = pick(0.0)
        if b > hi:
            sc = np.median([np.ptp(d) for d in dl if np.ptp(d) > 0] or [1.0]) / np.median([np.ptp(B) for B in pred.B if np.ptp(B) > 0] or [1.0])
            lo_l, hi_l = 0.0, max(sc, 1e-30)
            while pick(hi_l)[1] > hi and hi_l < sc * 1e12: lo_l = hi_l; hi_l *= 4
            for _ in range(50):
                m = 0.5 * (lo_l + hi_l)
                if pick(m)[1] > hi: lo_l = m
                else: hi_l = m
            J, b = pick(hi_l)
        cur = repair(pred, pred.assign(J, cur), lo, hi)
    return cur


def repair(pred, a, lo, hi):
    """bytes into [lo, hi]: downgrade the cheapest losses while > hi, upgrade the best gains while it fits"""
    M = pred.M; b = M.bytes(a)
    for _ in range(4000):
        if b <= hi: break
        J, dl = pred.lin(a); best = None
        for i, j in enumerate(J):
            if j == 0: continue
            db = pred.B[i][j - 1] - pred.B[i][j]; v = dl[i][j - 1] / -db
            if best is None or v < best[0]: best = (v, i, j - 1, db)
        if best is None: break
        a = pred.assign([best[2] if k == best[1] else j for k, j in enumerate(J)], a); b += best[3]
    for _ in range(4000):
        J, dl = pred.lin(a); best = None
        for i, j in enumerate(J):
            if j + 1 >= len(pred.B[i]): continue
            db = pred.B[i][j + 1] - pred.B[i][j]
            if b + db > hi or dl[i][j + 1] >= 0: continue
            v = dl[i][j + 1] / db
            if best is None or v < best[0]: best = (v, i, j + 1, db)
        if best is None: break
        a = pred.assign([best[2] if k == best[1] else j for k, j in enumerate(J)], a); b += best[3]
    return a


def bands(D, width=8):
    """group x depth band -> tensor indices"""
    out = {}
    for i in range(len(D.W)):
        if D.grp[i] < 0: continue
        out.setdefault((int(D.grp[i]), int(D.layer[i]) // width), []).append(i)
    return out


def step_move(pred, a, up_set, lo, hi, rng):
    """move tensors up_set one option up, pay with one-step downgrades of other tensors (cheapest predicted loss per
    byte) until bytes <= hi"""
    J, dl = pred.lin(a); J = list(J); b = pred.M.bytes(a)
    moved = set()
    for i in up_set:
        if J[i] + 1 < len(pred.B[i]): b += pred.B[i][J[i] + 1] - pred.B[i][J[i]]; J[i] += 1; moved.add(i)
    if not moved: return None
    cand = sorted(((dl[i][J[i] - 1] / (pred.B[i][J[i]] - pred.B[i][J[i] - 1]), i) for i in range(len(J)) if i not in moved and J[i] > 0))
    for _, i in cand:
        if b <= hi: break
        b -= pred.B[i][J[i]] - pred.B[i][J[i] - 1]; J[i] -= 1
    if b > hi: return None
    return repair(pred, pred.assign(J, a), lo, hi)


def block_move(pred, a, key, BD, lo, hi):
    J, dl = pred.lin(a); J = list(J); b = pred.M.bytes(a); idx = BD[key]
    for i in idx:
        if J[i] + 1 < len(pred.B[i]): b += pred.B[i][J[i] + 1] - pred.B[i][J[i]]; J[i] += 1
    # pay with whole bands down (cheapest predicted loss per byte), then single tensors
    opts = []
    for k2, ix in BD.items():
        if k2 == key: continue
        db = sum(pred.B[i][J[i]] - pred.B[i][J[i] - 1] for i in ix if J[i] > 0)
        dk = sum(dl[i][J[i] - 1] for i in ix if J[i] > 0)
        if db > 0: opts.append((dk / db, k2))
    opts.sort()
    for _, k2 in opts:
        if b <= hi: break
        for i in BD[k2]:
            if J[i] > 0: b -= pred.B[i][J[i]] - pred.B[i][J[i] - 1]; J[i] -= 1
    if b > hi: return None
    return repair(pred, pred.assign(J, a), lo, hi)


def generate(kind, pred, draws, a, lo, hi, seen, rng, tries=6):
    """one candidate of `kind` (None if none new); draws: list of Pred for posterior samples"""
    inc = ahash(a)
    for t in range(tries):
        c = _gen(kind, pred, draws, a, lo, hi, rng)
        if c is None: continue
        h = ahash(c)
        if h != inc and h not in seen and lo <= pred.M.bytes(c) <= hi: return c
    return None


def _gen(kind, pred, draws, a, lo, hi, rng):
    D = pred.D
    if kind == "solve": return mckp(pred, a, hi, lo)
    if kind == "thompson":
        if not draws: return None
        return mckp(rng.choice(draws), a, hi, lo)
    if kind == "partial":
        s = mckp(pred, a, hi, lo); J0 = pred.idx(a); J1 = pred.idx(s)
        diff = [i for i in range(len(J0)) if J0[i] != J1[i]]
        if not diff: return None
        _, dl = pred.lin(a)
        gain = sorted(diff, key=lambda i: dl[i][J1[i]] / max(abs(pred.B[i][J1[i]] - pred.B[i][J0[i]]), 1.0))
        k = max(1, int(len(gain) * rng.uniform(0.25, 0.75)))
        J = list(J0)
        for i in gain[:k]: J[i] = J1[i]
        return repair(pred, pred.assign(J, a), lo, hi)
    if kind == "pair":
        J, dl = pred.lin(a)
        ups = sorted(((dl[i][J[i] + 1] / (pred.B[i][J[i] + 1] - pred.B[i][J[i]]), i) for i in range(len(J)) if J[i] + 1 < len(pred.B[i])))
        if not ups: return None
        i = rng.choice(ups[:12])[1]
        return step_move(pred, a, [i], lo, hi, rng)
    if kind == "block":
        BD = bands(D); J, dl = pred.lin(a)
        sc = []
        for k, ix in BD.items():
            db = sum(pred.B[i][J[i] + 1] - pred.B[i][J[i]] for i in ix if J[i] + 1 < len(pred.B[i]))
            dk = sum(dl[i][J[i] + 1] for i in ix if J[i] + 1 < len(pred.B[i]))
            if db > 0: sc.append((dk / db, k))
        if not sc: return None
        sc.sort(); k = rng.choice(sc[:6])[1]
        return block_move(pred, a, k, BD, lo, hi)
    if kind == "explore":
        un = unseen_emb(pred, a)
        if un and rng.random() < 0.5:
            c = dict(a); c[MN.EMB] = un[0]
            return repair(pred, c, lo, hi)
        BD = bands(D); keys = sorted(BD)
        if rng.random() < 0.5 or not draws:
            return block_move(pred, a, rng.choice(keys), BD, lo, hi)
        cands = []
        for _ in range(24):
            if rng.random() < 0.5: c = block_move(pred, a, rng.choice(keys), BD, lo, hi)
            else: c = step_move(pred, a, [rng.randrange(len(D.W))], lo, hi, rng)
            if c is not None: cands.append(c)
        if not cands: return None
        base = [math.log(p.kl(a)) for p in draws[:12]]
        best, bv = None, None
        for c in cands:
            v = np.array([math.log(p.kl(c)) - b0 for p, b0 in zip(draws[:12], base)])
            ucb = -(v.mean() - 1.0 * v.std())                         # predicted gain + 1 sd
            if bv is None or ucb > bv: best, bv = c, ucb
        return best
    raise ValueError(kind)


def unseen_emb(pred, a):
    """embedding options no fitted row has used, nearest the incumbent's first"""
    if pred.emb is None or MN.EMB not in a or len(pred.emb["opts"]) < 2: return []
    seen = set(getattr(pred.Mo, "levels", []) or []); names = [o["o"] for o in pred.emb["opts"]]
    cur = names.index(a[MN.EMB])
    return [names[j] for j in sorted(range(len(names)), key=lambda j: (abs(j - cur), -j)) if j != cur and f"e{j}" not in seen]


def kind_shares(stats, share0=SHARE0, lo=0.05, hi=0.40):
    """V2's kind rewards: share ~ the smoothed accept rate x the share0 prior, clamped, renormalised.
    stats: kind -> (accepted, decided)"""
    w = {}
    for k, s0 in share0.items():
        acc, n = stats.get(k, (0, 0))
        w[k] = s0 * ((acc + 1) / (n + 4)) / 0.25
    tot = sum(w.values()); w = {k: min(hi, max(lo, v / tot)) for k, v in w.items()}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def pick_kind(shares, counts, rng):
    """deficit round robin over kinds by share"""
    n = sum(counts.get(k, 0) for k in shares) + 1
    return max(shares, key=lambda k: shares[k] * n - counts.get(k, 0) + 1e-6 * rng.random())
