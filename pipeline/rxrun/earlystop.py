# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""seqrule.py -- the search's accept / reject rule on per-document KL sums, paired with the incumbent (w126, #166).

One module for the boundary simulation (boundsim.py), the driver (driver.py) and the hosts' early-stop watcher
(hostd patch: a byte-identical copy as pipeline/rxrun/earlystop.py; test_v3.py checks the copy). Pure Python.

A candidate is scored document by document (radiance --kld-progress) in a fixed order with x / c interleaved. After
document k the paired statistic over the scored documents is
    d  = log(sum B / sum A)                                  (B candidate, A incumbent: KL sums on the same positions)
    se = sd(u) / sqrt(k),  u_d = B_d / mean B - A_d / mean A (delta method, both parts pooled)
    z  = d / se.
Rules (reject early only; accept only on the whole slice: d < 0 and d < 0 on every part):
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
        self.part = {}                                   # part -> [sum A, sum B, docs]
        self.stage = 0; self.stopped = None

    def add(self, a, b, part="all"):
        self.k += 1; self.SA += a; self.SB += b; self.SAA += a * a; self.SBB += b * b; self.SAB += a * b
        p = self.part.setdefault(part, [0.0, 0.0, 0]); p[0] += a; p[1] += b; p[2] += 1

    def d(self):
        return math.log(self.SB / self.SA) if self.SA > 0 and self.SB > 0 else 0.0

    def se(self):
        if self.k < 3 or self.SA <= 0 or self.SB <= 0: return INF
        mA, mB = self.SA / self.k, self.SB / self.k
        su2 = self.SBB / (mB * mB) - 2 * self.SAB / (mA * mB) + self.SAA / (mA * mA)   # sum of u^2 (mean u is 0)
        return math.sqrt(max(su2, 0.0) / (self.k - 1)) / math.sqrt(self.k)

    def z(self):
        s = self.se()
        return self.d() / s if s not in (0.0, INF) else 0.0

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
        """after the whole slice: "accept" iff the pooled and every part's log ratio are < 0"""
        if self.stopped: return self.stopped
        parts_ok = all(p[1] < p[0] for p in self.part.values())
        return "accept" if self.d() < 0 and parts_ok else "reject"

    def summary(self):
        return {"k": self.k, "n": self.N, "d": self.d(), "se": self.se(), "z": self.z(),
                "parts": {p: {"logr": math.log(v[1] / v[0]) if v[0] > 0 and v[1] > 0 else None, "docs": v[2]} for p, v in self.part.items()}}


def interleave(nx, nc):
    """the part of each position of a two-part order: position k (1-based) is x iff floor(k nx / N) steps up"""
    N = nx + nc; out = []
    for k in range(1, N + 1):
        out.append("x" if (k * nx) // N > ((k - 1) * nx) // N else "c")
    return out
