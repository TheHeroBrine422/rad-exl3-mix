"""pair84.py -- paired KLD statistics over one or several goldens (#117, ../w84.md), the arithmetic of swap27's
tools/paired.py: per document the sum of KL over the scored rows (gen.u8), ratio of sums, document bootstrap (2000
resamples, seed 0). Several goldens (the parts of a two-part slice) are pooled position-weighted, resampled within
each part (stratified bootstrap). Rows files: radiance --kld-out .rows, [rows, 4] f32 (KL, ?, ?, top-1 match)."""
import json
import numpy as np


def doc_sums(golden, rows_file):
    m = json.load(open(f"{golden}/kld.json")); gen = np.fromfile(f"{golden}/gen.u8", np.uint8).astype(bool)
    R = np.fromfile(rows_file, np.float32).reshape(-1, 4).astype(np.float64)
    assert len(R) == m["rows"] == len(gen), (rows_file, len(R), m["rows"])
    s, n, t = [], [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); g = gen[sl]
        s.append(R[sl, 0][g].sum()); t.append(R[sl, 3][g].sum()); n.append(g.sum())
    return np.array(s), np.array(n, float), np.array(t), R[gen]


def pair(parts, B=2000):
    """parts: [(golden, control_rows, arm_rows)] -> dict (pooled over the parts)"""
    sa, sb, n, ta, tb, ra, rb, strata = [], [], [], [], [], [], [], []
    for j, (g, a, b) in enumerate(parts):
        a_ = doc_sums(g, a); b_ = doc_sums(g, b)
        sa.append(a_[0]); sb.append(b_[0]); n.append(a_[1]); ta.append(a_[2]); tb.append(b_[2]); ra.append(a_[3]); rb.append(b_[3])
        strata.append(len(a_[0]))
    rng = np.random.default_rng(0)
    bs_r, bs_d = [], []
    for _ in range(B):
        A_ = B_ = N_ = 0.0
        for j in range(len(parts)):
            i = rng.integers(0, strata[j], strata[j]); A_ += sa[j][i].sum(); B_ += sb[j][i].sum(); N_ += n[j][i].sum()
        bs_r.append(B_ / A_ - 1); bs_d.append((B_ - A_) / N_)
    SA, SB, N = sum(x.sum() for x in sa), sum(x.sum() for x in sb), sum(x.sum() for x in n)
    RA, RB = np.concatenate(ra), np.concatenate(rb)
    d = (SB - SA) / N
    return {"kld_a": SA / N, "kld_b": SB / N, "r": SB / SA, "diff": d, "se": float(np.std(bs_d)), "z": d / float(np.std(bs_d)),
            "rel_se": float(np.std(bs_r)), "ci95": [1 + float(np.percentile(bs_r, 2.5)), 1 + float(np.percentile(bs_r, 97.5))],
            "p99_a": float(np.percentile(RA[:, 0], 99)), "p99_b": float(np.percentile(RB[:, 0], 99)),
            "top1_a": float(RA[:, 3].mean()), "top1_b": float(RB[:, 3].mean()), "docs": int(sum(strata)), "positions": int(N)}


def fmt(p):
    return (f"{p['kld_a']:.6f} -> {p['kld_b']:.6f}: r {p['r']:.4f} [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}] z {p['z']:+.2f}; "
            f"p99 {p['p99_a']:.4f} -> {p['p99_b']:.4f}; top-1 {p['top1_a']:.5f} -> {p['top1_b']:.5f} "
            f"({p['docs']} docs, {p['positions']} pos)")
