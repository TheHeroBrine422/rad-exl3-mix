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
Rules (reject early only; accept only on the whole slice: d + off < 0 and d_p + off < 0 on every part):
    {"kind": "curve", "p": P, "warmup": W, "b0": 2, "t0": 0.1}   reject once t >= W and z > b0 ((1 - t) / (1 - t0))^P,
                                                                 t = documents scored / documents in the slice
    {"kind": "stages", "at": [F1, ..], "z": [Z1, ..]}            reject at the document completing fraction F_i if z >= Z_i
    {"kind": "full"}                                             never early
"""
import math

INF = float("inf")


def boundary(rule, t):
    """the reject bound on z at slice fraction t (INF = no early reject there)"""
    k = rule.get("kind", "full")
    if k == "curve":
        if t < rule.get("warmup", 0.1) or t >= 1.0: return INF
        return rule.get("b0", 2.0) * ((1.0 - t) / (1.0 - rule.get("t0", 0.1))) ** rule["p"]
    return INF


class Paired:
    """running paired statistic of one candidate vs the incumbent over the documents scored so far"""

    def __init__(self, n_total, rule):
        self.N, self.rule = n_total, rule
        self.k = 0; self.SA = self.SB = self.SAA = self.SBB = self.SAB = 0.0
        self.part = {}                                   # part -> [sum A, sum B, docs, sum AA, sum BB, sum AB]
        self.stage = 0; self.stopped = None; self.off = float(rule.get("off", 0.0) or 0.0)
        self.stat = rule.get("stat", "pmean")

    def add(self, a, b, part="all"):
        self.k += 1; self.SA += a; self.SB += b; self.SAA += a * a; self.SBB += b * b; self.SAB += a * b
        p = self.part.setdefault(part, [0.0, 0.0, 0, 0.0, 0.0, 0.0]); p[0] += a; p[1] += b; p[2] += 1
        p[3] += a * a; p[4] += b * b; p[5] += a * b

    def d_pooled(self):
        return math.log(self.SB / self.SA) if self.SA > 0 and self.SB > 0 else 0.0

    def d(self):
        """the statistic's log ratio: the mean of the parts' log ratios ("pmean") or the pooled one"""
        if self.stat == "pooled": return self.d_pooled()
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

    def se(self):
        if self.stat == "pooled" or not self.part: return self._se(self.k, self.SA, self.SB, self.SAA, self.SBB, self.SAB)
        ss = [self._se(p[2], p[0], p[1], p[3], p[4], p[5]) for p in self.part.values()]
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

    def final(self):
        """after the whole slice: "accept" iff the pooled and every part's (size-adjusted) log ratio are < 0"""
        if self.stopped: return self.stopped
        parts_ok = all(p[0] > 0 and p[1] > 0 and math.log(p[1] / p[0]) + self.off < 0 for p in self.part.values())
        return "accept" if self.da() < 0 and parts_ok else "reject"

    def summary(self):
        return {"k": self.k, "n": self.N, "stat": self.stat, "d": self.d(), "d_pooled": self.d_pooled(), "off": self.off, "se": self.se(), "z": self.z(),
                "parts": {p: {"logr": math.log(v[1] / v[0]) if v[0] > 0 and v[1] > 0 else None, "docs": v[2]} for p, v in self.part.items()}}


def interleave(nx, nc):
    """the part of each position of a two-part order: position k (1-based) is x iff floor(k nx / N) steps up"""
    N = nx + nc; out = []
    for k in range(1, N + 1):
        out.append("x" if (k * nx) // N > ((k - 1) * nx) // N else "c")
    return out
