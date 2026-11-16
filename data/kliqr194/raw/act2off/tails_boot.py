#!/usr/bin/env python3
# tails_boot.py ON_ROWS OFF_ROWS GOLDEN_DIR [QB] -- w178 (#194-02): the guards next to the mean (README rule: report the
# tails and top-1, decide on the mean). Over the golden's gen rows (gen.u8, as kl_mean.py): KL99 and KL99.9 of each arm
# (column 0 of the radiance rows file), top-1 agreement (column 3, 0/1 flag), the off / on ratio of the tails and the
# top-1 difference, with per-document bootstrap 95 % intervals (the same documents for both arms; QB resamples, default 500,
# seed 194). Columns checked: col 0 mean over gen rows = kl_mean.py's value; col 3 is a 0/1 flag. Prints one JSON line.
import json, sys
import numpy as np

on_f, off_f, gold = sys.argv[1], sys.argv[2], sys.argv[3]
QB = int(sys.argv[4]) if len(sys.argv) > 4 else 500
m = json.load(open(gold + "/kld.json")); n = m["rows"]
gen = np.fromfile(gold + "/gen.u8", np.uint8).astype(bool)
assert gen.size == n


def load(p):
    A = np.fromfile(p, np.float32).reshape(n, -1)
    assert set(np.unique(A[:, 3])) <= {0.0, 1.0}
    return A[:, 0].astype(np.float64), A[:, 3].astype(np.float64)


ka, ta = load(on_f); kb, tb = load(off_f)
docs = []
for d in m["docs"]:
    k = d["tokens"] - 1 - d["score_from"]
    s = slice(d["row0"], d["row0"] + k); g = gen[s]
    docs.append((ka[s][g], kb[s][g], ta[s][g], tb[s][g]))
keep = [i for i, d in enumerate(docs) if len(d[0])]
docs = [docs[i] for i in keep]
A_ = np.concatenate([d[0] for d in docs]); B_ = np.concatenate([d[1] for d in docs])
TA_ = np.concatenate([d[2] for d in docs]); TB_ = np.concatenate([d[3] for d in docs])
qs = (0.99, 0.999)


def stats(a, b, tab, tbb):
    qa, qb = np.quantile(a, qs), np.quantile(b, qs)
    return np.array([a.mean(), b.mean(), qa[0], qb[0], qa[1], qb[1], tab.mean(), tbb.mean()])


point = stats(A_, B_, TA_, TB_)
rng = np.random.default_rng(194)
R = []
for ix in rng.integers(0, len(docs), (QB, len(docs))):
    a = np.concatenate([docs[i][0] for i in ix]); b = np.concatenate([docs[i][1] for i in ix])
    ta2 = np.concatenate([docs[i][2] for i in ix]); tb2 = np.concatenate([docs[i][3] for i in ix])
    R.append(stats(a, b, ta2, tb2))
R = np.array(R)
lo, hi = np.percentile(R, [2.5, 97.5], axis=0)
out = {"on_rows": on_f, "off_rows": off_f, "docs": len(docs), "gen_positions": int(len(A_)), "QB": QB,
       "kl_on": point[0], "kl_off": point[1], "kl_off_over_on_pct": (point[1] / point[0] - 1) * 100,
       "kl99_on": point[2], "kl99_off": point[3], "kl99_ratio": point[3] / point[2],
       "kl999_on": point[4], "kl999_off": point[5], "kl999_ratio": point[5] / point[4],
       "top1_on": point[6], "top1_off": point[7], "top1_diff_pp": (point[7] - point[6]) * 100}
rr = R[:, 3] / R[:, 2]; r999 = R[:, 5] / R[:, 4]; rd = (R[:, 7] - R[:, 6]) * 100; rm = R[:, 1] / R[:, 0] - 1
out["kl99_ratio_ci95"] = [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]
out["kl999_ratio_ci95"] = [float(np.percentile(r999, 2.5)), float(np.percentile(r999, 97.5))]
out["top1_diff_pp_ci95"] = [float(np.percentile(rd, 2.5)), float(np.percentile(rd, 97.5))]
out["kl_off_over_on_pct_ci95"] = [float(np.percentile(rm, 2.5) * 100), float(np.percentile(rm, 97.5) * 100)]
print(json.dumps(out))
