# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""refs186ci.py GOLDEN ROWS_A ROWS_B [--reps N] [--name-a A --name-b B] -- paired per-document bootstrap of two
references on one golden's gen positions (#186, RULES-strata §7, w164). Documents are resampled together (the pair stays
on the same documents), 4,000 replicates by default, seed 186. Prints mean A, mean B, the ratio B/A and the difference
B-A with 95 % percentile intervals, and the number of documents. Rows files: float32, 4 columns per golden row, column 0 =
per-position KL (radiance --kld-out's .json.rows). A position counts when the golden's gen.u8 marks it.
Usage (ds8, CPU): nice -n 19 /data/q27/venv/bin/python -I refs186ci.py GOLDEN ROWS_A ROWS_B --name-a iq2r --name-b v2-2.50"""
import argparse, json, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden"); ap.add_argument("rows_a"); ap.add_argument("rows_b")
    ap.add_argument("--reps", type=int, default=4000); ap.add_argument("--seed", type=int, default=186)
    ap.add_argument("--name-a", default="A"); ap.add_argument("--name-b", default="B")
    a = ap.parse_args()
    m = json.load(open(os.path.join(a.golden, "kld.json")))
    gen = np.fromfile(os.path.join(a.golden, "gen.u8"), np.uint8).astype(bool)
    A = np.fromfile(a.rows_a, np.float32).reshape(m["rows"], -1)[:, 0].astype(np.float64)
    B = np.fromfile(a.rows_b, np.float32).reshape(m["rows"], -1)[:, 0].astype(np.float64)
    # per-document sums and counts over gen positions (document d covers rows row0 .. row0 + tokens - 1 - score_from)
    sa, sb, nn = [], [], []
    for d in m["docs"]:
        n = d["tokens"] - 1 - d["score_from"]; r = slice(d["row0"], d["row0"] + n); g = gen[r]
        sa.append(A[r][g].sum()); sb.append(B[r][g].sum()); nn.append(int(g.sum()))
    sa, sb, nn = map(np.asarray, (sa, sb, nn))
    keep = nn > 0; sa, sb, nn = sa[keep], sb[keep], nn[keep]
    D = len(nn)
    ma, mb = sa.sum() / nn.sum(), sb.sum() / nn.sum()
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, D, size=(a.reps, D))
    ra = sa[idx].sum(1) / nn[idx].sum(1); rb = sb[idx].sum(1) / nn[idx].sum(1)
    ratio = rb / ra; diff = rb - ra
    q = lambda x: np.percentile(x, [2.5, 97.5])
    print(json.dumps({"A": a.name_a, "B": a.name_b, "docs": D, "positions": int(nn.sum()),
                      "mean_A": ma, "mean_B": mb, "ratio_B_over_A": mb / ma, "ratio_ci95": q(ratio).tolist(),
                      "diff_B_minus_A": mb - ma, "diff_ci95": q(diff).tolist()}))


if __name__ == "__main__":
    main()
