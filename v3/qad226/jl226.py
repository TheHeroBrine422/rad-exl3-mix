#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""jl226.py QROOT CTLNAME NAME [NAME ...] -- long-context judge readout (golden-jlong: positions >= 8,192 of 74 qad-pool docs): each read's KL / gen row and its
paired ratio vs the control read CTLNAME (e.g. JL-UN2.50, the unhealed release) with cmp214's per-document bootstrap CI (2,000 resamples, seed 0), docs better /
74, median per-document ratio. Rows: QROOT/scores/<NAME>.json.rows; golden QROOT/slices/golden-jlong (qscore.sh's JL golden). Prints one markdown row a read."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
import cmp214                                                    # noqa: E402
root, ctl = sys.argv[1:3]; G = f"{root}/slices/golden-jlong"
xc, n = cmp214.doc_sums(G, f"{root}/scores/{ctl}.json.rows")
print(f"| read | KL / gen row | ratio vs {ctl} [95 % CI] | docs better | median doc ratio |\n|---|---|---|---|---|")
print(f"| {ctl} | {xc.sum() / n.sum():.6f} | 1 | — | — |")
for nm in sys.argv[3:]:
    x, nb = cmp214.doc_sums(G, f"{root}/scores/{nm}.json.rows"); assert (nb == n).all(), nm
    r = cmp214.paired(xc, x, n); ok = n > 0
    print(f"| {nm} | {x.sum() / n.sum():.6f} | {r['ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['docs_better']} / {int(ok.sum())} | "
          f"{float(np.median(x[ok] / np.maximum(xc[ok], 1e-12))):.3f} |")
