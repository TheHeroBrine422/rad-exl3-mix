# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""seqrule.py -- the search's accept / reject rule on per-document KL sums, paired with the incumbent (w126, #166).

One module for the boundary simulation (boundsim.py), the driver (driver.py) and the hosts' early-stop watcher
(hostd patch: a byte-identical copy as pipeline/rxrun/earlystop.py; test_v3.py checks the copy). Pure Python.

A candidate is scored document by document (radiance --kld-progress) in a fixed order with x / c interleaved. After
document k the paired statistic over the scored documents is (w131, #171: rule key "stat", default "pmean")
    d_p  = log(sum_p B / sum_p A)                            (B candidate, A incumbent: KL sums of part p's documents)
    se_p = sd(u) / sqrt(k_p),  u_d = B_d / mean B - A_d / mean A over part p's documents (delta method)
    d    = mean_p d_p,  se = sqrt(sum_p se_p^2) / n_parts    (equal weight per part; se infinite until every part has
                                                             3 documents)
    z    = (d + off) / se.
"pooled" = the old statistic (w126): d = log(sum B / sum A) over both parts, se from u over all scored documents. With
one part the two are the same.
off = the size adjustment (w127, #167): b (bpw_candidate - bpw_incumbent), b = the fit's local slope -d log KLD / d bpw
at the size, so the compared score is log KLD + b bpw (rule key "off", default 0 = raw KLD; the driver puts it in the
early-stop block's rule, so the hosts stop on the same adjusted statistic).
Capped statistic (w145, RULES-picks.md (b), user 11-13): rule key "w" = one weight per document of the slice (in its
order; cap_weights below: each document's incumbent KL capped at `cap` (0.2) of its part's whole-slice incumbent total,
heavier documents scaled down to the cap); A_d and B_d are both multiplied by w_d before every sum above, so d_p is
log(sum_p w B / sum_p w A) and the early stop runs on it too (the hosts get "w" in the early-stop block's rule; no
hostd change). Absent = every weight 1 = the old statistic.
Per-document guard (RULES-picks (c)): rule key "guard": "docmean" -- at the end of a whole slice, every part's
c_p = mean_d log(B_d / A_d) (unweighted, documents with A_d, B_d > 0) must have the same sign as its d_p, both
size-adjusted (c_p + off vs d_p + off); otherwise the candidate is rejected (Paired.why = "guard"). No CI requirement.
Stratified slices (RULES-strata.md, #186, w164): a document may carry a stratum ("L" or "N"; add(a, b, part, stratum)).
Then every sum above is kept per (part, stratum): the cap is within (part, stratum) (cap_weights(strata=...)), the part's
log ratio is d_p = sum_s w_s d_ps / sum_s w_s over the strata present (w_L = 0.961, w_N = 0.039, STRATUM_W; a stratum
absent from a part: the other alone), its se is sqrt(sum_s (w_s / W)^2 se_ps^2), the guard's c_p is the same weighted
mean of the per-stratum document means, and d = mean_p d_p as before. The rule must say "strata": true (the driver and
the hosts' early-stop block do); without it the code is the old one, number for number. d_pooled() of a stratified
candidate is the weighted pooled ratio log(sum_{p,s} w_s B / sum_{p,s} w_s A) (a diagnostic only; the decision is "pmean").
Rules (reject early only; accept only on the whole slice: d + off < 0 and d_p + off < 0 on every part, and the guard):
    {"kind": "curve", "p": P, "warmup": W, "b0": 2, "t0": 0.1}   reject once t >= W and z > b0 ((1 - t) / (1 - t0))^P,
                                                                 t = documents scored / documents in the slice
    {"kind": "stages", "at": [F1, ..], "z": [Z1, ..]}            reject at the document completing fraction F_i if z >= Z_i
    {"kind": "full"}                                             never early
"""
import math

INF = float("inf")


def capped_weights(a, cap):
    """weights w <= 1 so that w_i a_i <= cap * sum(w a) for every i: the over-cap documents (a set that only grows) are
    fixed at exactly cap of the new total T = rest / (1 - n_fixed cap), the others keep 1 (exact water-filling;
    eval/docstat/readout.py's loop gives the same weights except in rare cases where it stops slightly above the cap)"""
    n = len(a); w = [1.0] * n
    if sum(a) <= 0 or n * cap < 1: return w
    fixed = set()
    while True:
        rest = sum(x for i, x in enumerate(a) if i not in fixed); T = rest / (1 - len(fixed) * cap)
        new = {i for i, x in enumerate(a) if i not in fixed and x > (cap + 1e-12) * T}
        if not new: break
        fixed |= new
    for i in fixed: w[i] = cap * T / a[i]
    return w


def cap_weights(ref, parts, cap=0.2, strata=None):
    """the capped statistic's per-document weights: per part, capped_weights of the incumbent's (ref's) KL sums over the
    WHOLE slice (ref: one number per document, slice order; parts: the documents' parts). strata (RULES-strata §4): one
    stratum per document -> the cap is per (part, stratum) instead. None = no cap"""
    if not cap: return None
    w = [1.0] * len(ref)
    keys = parts if strata is None else [(p, s) for p, s in zip(parts, strata)]
    for p in dict.fromkeys(keys):
        ix = [i for i, q in enumerate(keys) if q == p]
        for i, x in zip(ix, capped_weights([max(float(ref[i]), 0.0) for i in ix], cap)): w[i] = x
    return w


def boundary(rule, t):
    """the reject bound on z at slice fraction t (INF = no early reject there)"""
    k = rule.get("kind", "full")
    if k == "curve":
        if t < rule.get("warmup", 0.1) or t >= 1.0: return INF
        return rule.get("b0", 2.0) * ((1.0 - t) / (1.0 - rule.get("t0", 0.1))) ** rule["p"]
    return INF


STRATUM_W = {"L": 0.961, "N": 0.039}          # RULES-strata §2 (fixed for V3; the rule may override it with "sw")


class Paired:
    """running paired statistic of one candidate vs the incumbent over the documents scored so far"""

    def __init__(self, n_total, rule):
        self.N, self.rule = n_total, rule
        self.k = 0; self.SA = self.SB = self.SAA = self.SBB = self.SAB = 0.0
        self.part = {}                                   # part -> [sum A, sum B, docs, sum AA, sum BB, sum AB]
        self.stage = 0; self.stopped = None; self.why = None; self.off = float(rule.get("off", 0.0) or 0.0)
        self.stat = rule.get("stat", "pmean"); self.w = rule.get("w"); self.guard = rule.get("guard")
        self.lr = {}                                     # part -> [sum log(B / A), documents] (the (c) guard)
        self.raw = {}                                    # part -> [sum A, sum B] unweighted ((a), reported only)
        self.strat = bool(rule.get("strata"))            # RULES-strata: the per-(part, stratum) sums below
        self.sw = dict(STRATUM_W, **(rule.get("sw") or {}))
        self.ps = {}                                     # (part, stratum) -> the part's six sums, as above
        self.lps = {}                                    # (part, stratum) -> [sum log(B / A), documents]

    def add(self, a, b, part="all", stratum=None):
        if self.strat != (stratum is not None):
            raise ValueError("stratified rule: every document needs its stratum" if self.strat else "stratum given to an unstratified rule (no \"strata\": true)")
        if stratum is not None and stratum not in self.sw: raise ValueError(f"unknown stratum {stratum!r}")
        if a > 0 and b > 0: q = self.lr.setdefault(part, [0.0, 0]); q[0] += math.log(b / a); q[1] += 1
        r = self.raw.setdefault(part, [0.0, 0.0]); r[0] += a; r[1] += b
        if self.strat and a > 0 and b > 0: q = self.lps.setdefault((part, stratum), [0.0, 0]); q[0] += math.log(b / a); q[1] += 1
        if self.w is not None: wk = self.w[self.k]; a, b = a * wk, b * wk
        self.k += 1; self.SA += a; self.SB += b; self.SAA += a * a; self.SBB += b * b; self.SAB += a * b
        p = self.part.setdefault(part, [0.0, 0.0, 0, 0.0, 0.0, 0.0]); p[0] += a; p[1] += b; p[2] += 1
        p[3] += a * a; p[4] += b * b; p[5] += a * b
        if self.strat:
            x = self.ps.setdefault((part, stratum), [0.0, 0.0, 0, 0.0, 0.0, 0.0]); x[0] += a; x[1] += b; x[2] += 1
            x[3] += a * a; x[4] += b * b; x[5] += a * b

    def d_pooled(self):
        if self.strat:
            SA = sum(self.sw[s] * v[0] for (_, s), v in self.ps.items()); SB = sum(self.sw[s] * v[1] for (_, s), v in self.ps.items())
            return math.log(SB / SA) if SA > 0 and SB > 0 else 0.0
        return math.log(self.SB / self.SA) if self.SA > 0 and self.SB > 0 else 0.0

    def part_logr(self, p):
        """d_p: the part's log ratio, None without positive sums; stratified: sum_s w_s d_ps / sum_s w_s (the strata present)"""
        if not self.strat:
            v = self.part.get(p)
            return math.log(v[1] / v[0]) if v and v[0] > 0 and v[1] > 0 else None
        ks = [(s, v) for (q, s), v in self.ps.items() if q == p and v[0] > 0 and v[1] > 0]
        if not ks: return None
        W = sum(self.sw[s] for s, _ in ks)
        return sum(self.sw[s] * math.log(v[1] / v[0]) for s, v in ks) / W

    def d(self):
        """the statistic's log ratio: the mean of the parts' log ratios ("pmean") or the pooled one"""
        if self.stat == "pooled": return self.d_pooled()
        if self.strat:
            lr = [x for x in (self.part_logr(p) for p in self.part) if x is not None]
        else:
            lr = [math.log(p[1] / p[0]) for p in self.part.values() if p[0] > 0 and p[1] > 0]
        return sum(lr) / len(lr) if lr else 0.0

    def da(self):
        """the size-adjusted log ratio d + off"""
        return self.d() + self.off

    @staticmethod
    def _se(k, SA, SB, SAA, SBB, SAB):
        if k < 3 or SA <= 0 or SB <= 0: return INF
        mA, mB = SA / k, SB / k
        su2 = SBB / (mB * mB) - 2 * SAB / (mA * mB) + SAA / (mA * mA)                  # sum of u^2 (mean u is 0)
        return math.sqrt(max(su2, 0.0) / (k - 1)) / math.sqrt(k)

    def _se_part(self, p):
        """the stratified se of part p: sqrt(sum_s (w_s / W)^2 se_ps^2) over the strata present (independent documents)"""
        ks = [(s, v) for (q, s), v in self.ps.items() if q == p and v[0] > 0 and v[1] > 0]
        if not ks: return INF
        W = sum(self.sw[s] for s, _ in ks)
        ss = [self._se(v[2], v[0], v[1], v[3], v[4], v[5]) for _, v in ks]
        if any(x == INF for x in ss): return INF
        return math.sqrt(sum((self.sw[s] / W * x) ** 2 for (s, _), x in zip(ks, ss)))

    def se(self):
        if self.stat == "pooled" or not self.part: return self._se(self.k, self.SA, self.SB, self.SAA, self.SBB, self.SAB)
        if self.strat: ss = [self._se_part(p) for p in self.part]
        else: ss = [self._se(p[2], p[0], p[1], p[3], p[4], p[5]) for p in self.part.values()]
        if any(x == INF for x in ss): return INF
        return math.sqrt(sum(x * x for x in ss)) / len(ss)

    def z(self):
        s = self.se()
        return self.da() / s if s not in (0.0, INF) else 0.0

    def check(self):
        """after a document: "reject" (stop the run) or None"""
        if self.stopped: return self.stopped
        t = self.k / self.N; r = self.rule
        if r.get("kind") == "stages":
            while self.stage < len(r["at"]) and self.k >= math.ceil(r["at"][self.stage] * self.N - 1e-9):
                zb = r["z"][self.stage]; self.stage += 1
                if self.z() >= zb: self.stopped = "reject"; return self.stopped
            return None
        if self.z() > boundary(r, t): self.stopped = "reject"
        return self.stopped

    def docmean(self, part):
        """(c): the part's mean per-document log ratio (unweighted; None without documents). Stratified (RULES-strata §5):
        the w_s-weighted mean of its strata's document means"""
        if self.strat:
            ks = [(s, v) for (q, s), v in self.lps.items() if q == part and v[1]]
            if not ks: return None
            W = sum(self.sw[s] for s, _ in ks)
            return sum(self.sw[s] * v[0] / v[1] for s, v in ks) / W
        q = self.lr.get(part)
        return q[0] / q[1] if q and q[1] else None

    def guard_ok(self):
        """the (c) guard: every part's size-adjusted per-document mean has the sign of its size-adjusted (b) log ratio"""
        if self.guard != "docmean": return True
        if self.strat:
            for p in self.part:
                c, dp = self.docmean(p), self.part_logr(p)
                if c is None or dp is None or (c + self.off < 0) != (dp + self.off < 0): return False
            return True
        for p, v in self.part.items():
            c = self.docmean(p)
            if c is None or v[0] <= 0 or v[1] <= 0: return False
            if (c + self.off < 0) != (math.log(v[1] / v[0]) + self.off < 0): return False
        return True

    def final(self):
        """after the whole slice: "accept" iff the pooled and every part's (size-adjusted) log ratio are < 0 and the
        guard holds (self.why = "guard" when only the guard rejects)"""
        if self.stopped: return self.stopped
        if self.strat: parts_ok = all(self.part_logr(p) is not None and self.part_logr(p) + self.off < 0 for p in self.part)
        else: parts_ok = all(p[0] > 0 and p[1] > 0 and math.log(p[1] / p[0]) + self.off < 0 for p in self.part.values())
        if not (self.da() < 0 and parts_ok): return "reject"
        if self.guard_ok(): return "accept"
        self.why = "guard"; return "reject"

    def summary(self):
        parts = {}
        for p, v in self.part.items():
            e = {"logr": self.part_logr(p) if self.strat else (math.log(v[1] / v[0]) if v[0] > 0 and v[1] > 0 else None), "docmean": self.docmean(p),
                 "pooled_a": math.log(self.raw[p][1] / self.raw[p][0]) if self.raw[p][0] > 0 and self.raw[p][1] > 0 else None, "docs": v[2]}
            if self.strat:
                e["strata"] = {s: {"logr": math.log(x[1] / x[0]) if x[0] > 0 and x[1] > 0 else None, "docmean": self.docmean_ps(p, s), "docs": x[2]}
                               for (q, s), x in self.ps.items() if q == p}
            parts[p] = e
        out = {"k": self.k, "n": self.N, "stat": self.stat, "d": self.d(), "d_pooled": self.d_pooled(), "off": self.off, "se": self.se(), "z": self.z(),
               "capped": self.w is not None, "guard": self.guard, "parts": parts}
        if self.strat: out["strata"] = True                # (absent for old slices: their summary is unchanged)
        return out

    def docmean_ps(self, part, s):
        q = self.lps.get((part, s))
        return q[0] / q[1] if q and q[1] else None


def interleave(nx, nc):
    """the part of each position of a two-part order: position k (1-based) is x iff floor(k nx / N) steps up"""
    N = nx + nc; out = []
    for k in range(1, N + 1):
        out.append("x" if (k * nx) // N > ((k - 1) * nx) // N else "c")
    return out
