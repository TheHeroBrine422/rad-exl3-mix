# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""refs186tails.py GOLDEN ROWS_DIR -- per reference on golden-t186n (#186, w164): the gen-position mean KLD, p99 (linear
interpolation between order statistics, as radiance's report) and top-1 agreement, over the positions the board counts
(gen.u8). Rows files: float32, 4 columns per golden row; column 0 = per-position KL, column 3 = top-1 agreement (both
checked against radiance's report on all rows: means equal). The report's own "all" figures count every row, prompt
positions included, so they are not the board's numbers. Output: TSV on stdout, sorted by name.
Usage (ds8, CPU): nice -n 19 /data/q27/venv/bin/python -I refs186tails.py GOLDEN ROWS_DIR > tails.tsv"""
import glob, os, sys

import numpy as np

G, R = sys.argv[1:3]
gen = np.fromfile(os.path.join(G, "gen.u8"), np.uint8).astype(bool)
print("name\tN_mean_kld\tN_p99\tN_top1\tgen_positions")
for p in sorted(glob.glob(os.path.join(R, "*-t186n.json.rows"))):
    name = os.path.basename(p)[:-len("-t186n.json.rows")]
    a = np.fromfile(p, np.float32).reshape(gen.size, -1)
    k = a[gen, 0].astype(np.float64); t = a[gen, 3].astype(np.float64)
    print(f"{name}\t{k.mean():.6f}\t{np.percentile(k, 99):.5f}\t{t.mean():.4f}\t{int(gen.sum())}")
