#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""eval215.py RUN [RUN ...] -- w215's held-out sanity: gen-row KL (no grad, first 2,048 tokens, one segment) of the last
N qad-pool documents of eq215's seeded order (never reached by its 20 steps) under each RUN's final scales (state.pt;
RUN = "-" for the untrained student); prints KL / row per run and the relative difference to the first run. Same args
as eq215.sh's student (env: MODEL, LEVEL, CALIB, POOLS)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import qad181  # noqa: E402
import qmodel  # noqa: E402
import data181 as D  # noqa: E402

N, L = int(os.environ.get("N215", "40")), 2048


def main():
    E = os.environ
    qad181.mad_int_form(); qmodel.use_shift_conv()
    C = D.Corpus([], 8192, calib=E["CALIB"], pools=E["POOLS"], pool="qad")
    order = np.random.default_rng(0).permutation(len(C.items))
    docs = [(C.items[k][0], C.items[k][1]) for k in order[-N:]]
    stu = qmodel.Student(E["MODEL"], E["LEVEL"], ["cuda:0"])
    qad181.use_seg_attention(stu)
    base = None
    for run in sys.argv[1:]:
        if run == "-": stu.load_state({n: torch.zeros_like(p) for n, p in stu.P.items()})
        else: stu.load_state(torch.load(os.path.join(run, "state.pt"), weights_only=False, map_location="cpu")["snap"]["cur"])
        s = np.zeros(2)
        with torch.no_grad():
            for gi, i in docs:
                kl, m, _ = qad181.seg_loss(stu, C.g, (gi, i), 0, min(L, C.g[gi].ntok(i)), None)
                s += (float(kl), m)
        v = s[0] / s[1]; base = v if base is None else base
        print(f"{run}: held-out KL/row {v:.6f} over {int(s[1])} rows of {len(docs)} docs ({(v / base - 1) * 100:+.4f} % vs first)", flush=True)


if __name__ == "__main__":
    main()
