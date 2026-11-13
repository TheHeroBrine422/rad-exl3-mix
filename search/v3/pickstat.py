# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pickstat.py -- RULES-picks.md's three statistics of a paired comparison and their paired bootstrap CIs (w145).

Per part, from per-document KL sums of A (the reference) and B (the candidate) on the same documents:
    (a) pooled    log(sum B / sum A)                          the board's estimator
    (b) capped    log(sum w B / sum w A), w = capped weights of A: a document holds at most `cap` (0.2) of the part's
                  A total (heavier documents scaled down to the cap, in A and B alike; seqrule.capped_weights)
    (c) docmean   mean_d log(B_d / A_d) over the documents with A_d, B_d > 0
CIs: paired bootstrap over resampling UNITS (a document, or a whole long trace: all its chunks together), the same
draws for every pair of a part, 10,000 draws, 95 % percentile interval. Inside a draw a unit drawn m times counts m
times; the cap is recomputed on the drawn sample (each drawn copy is one document for the cap).
numpy only; used by finalpick.py (final picks) and driver.py (the held-out monitor)."""
import numpy as np

CAP = 0.2
BOOT = 10000


def capped(a, n=None, cap=CAP):
    """weights w (same shape as a; rows = bootstrap draws or a single vector) so that every document copy holds at most
    cap of sum(n w a); n = copy counts (default 1). Exact water-filling as seqrule.capped_weights (the fixed set only
    grows; T = rest / (1 - copies fixed x cap)). A row where the cap cannot be met (copies fixed x cap >= 1, e.g. one
    document drawn 5 times at cap 0.2) stays uncapped."""
    a = np.asarray(a, float); one = a.ndim == 1
    A = a[None, :] if one else a
    N = np.ones_like(A) if n is None else np.broadcast_to(np.asarray(n, float), A.shape)
    fixed = np.zeros(A.shape, bool); live = (N * A).sum(1) > 0
    T = (N * A).sum(1)
    for _ in range(A.shape[1] + 1):
        k = (N * fixed).sum(1); den = 1 - k * cap
        live &= den > 1e-9
        T = np.where(live, (N * A * ~fixed).sum(1) / np.where(live, den, 1.0), T)
        new = live[:, None] & ~fixed & (N > 0) & (A > (cap + 1e-12) * T[:, None])
        if not new.any(): break
        fixed |= new
    W = np.where(fixed & live[:, None], cap * T[:, None] / np.where(A > 0, A, 1.0), 1.0)
    return W[0] if one else W


def point(A, B, cap=CAP):
    """(a), (b), (c) of B vs A on one part (1-d arrays of per-document KL sums)"""
    A = np.asarray(A, float); B = np.asarray(B, float); w = capped(A, cap=cap); ok = (A > 0) & (B > 0)
    return {"a": float(np.log(B.sum() / A.sum())), "b": float(np.log((w * B).sum() / (w * A).sum())),
            "c": float(np.log(B[ok] / A[ok]).mean()) if ok.any() else float("nan")}


def unit_index(units):
    """units: one label per document (None = every document its own unit) -> (unit id per document, number of units)"""
    if units is None: return None, None
    ids = {}
    u = np.array([ids.setdefault(str(x), len(ids)) for x in units])
    return u, len(ids)


def draws(n_docs, units=None, boot=BOOT, seed=0):
    """copy counts per document for `boot` bootstrap draws (boot x docs): units resampled with replacement"""
    rng = np.random.default_rng(seed)
    u, nu = unit_index(units)
    if u is None: return rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs), size=boot).astype(np.float32)
    cu = rng.multinomial(nu, np.full(nu, 1.0 / nu), size=boot).astype(np.float32)
    return cu[:, u]


def boot_pair(A, B, C, cap=CAP, chunk=1000):
    """bootstrap values of (a), (b), (c) for B vs A under the copy counts C (boot x docs) -> {"a": arr, "b": arr, "c": arr}"""
    A = np.asarray(A, float); B = np.asarray(B, float); ok = (A > 0) & (B > 0)
    lr = np.where(ok, np.log(np.where(ok, B, 1.0) / np.where(ok, A, 1.0)), 0.0)
    out = {"a": [], "b": [], "c": []}
    for i in range(0, len(C), chunk):
        n = C[i:i + chunk].astype(float)
        out["a"].append(np.log((n @ B) / (n @ A)))
        W = capped(np.broadcast_to(A, n.shape), n, cap)
        out["b"].append(np.log((n * W * B).sum(1) / (n * W * A).sum(1)))
        nk = n * ok
        out["c"].append((nk @ lr) / np.maximum(nk.sum(1), 1))
    return {k: np.concatenate(v) for k, v in out.items()}


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))] if len(x) else [float("nan")] * 2


def compare(A, B, units=None, cap=CAP, boot=BOOT, seed=0, off=0.0, C=None):
    """one part, B vs A: each statistic's point value + off (the size adjustment) and its 95 % CI"""
    p = point(A, B, cap); C = draws(len(A), units, boot, seed) if C is None else C
    bs = boot_pair(A, B, C, cap)
    return {k: {"d": p[k] + off, "ci": [x + off for x in ci(bs[k])]} for k in ("a", "b", "c")}


def max_share(S):
    """the largest document's share of the part's KL (S: 1-d per-document KL sums)"""
    S = np.asarray(S, float); t = S.sum()
    return float(S.max() / t) if t > 0 else 0.0
