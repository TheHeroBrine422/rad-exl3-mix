#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""blowup222-detail.py GOLDEN D1.json SCORES_DIR [SCORES_DIR ...] -- what a break looks like row by row (w222 D1): for every (document, read) in blowup222.py's JSON: KL / row and top-1 agreement
(rows column 3) and the teacher's / student's NLL of the golden token before and after the onset (columns 1 / 2 of the rows file are not interpreted: unknown semantics),
the share of post-onset gen rows whose KL < 0.1 (recovery), the largest single-row KL and the number of rows with KL > 5."""
import glob, json, os, sys
import numpy as np
g, js = sys.argv[1:3]; dirs = sys.argv[3:]
m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool); D = json.load(open(js))["docs"]
orig = {int(d["orig"]): i for i, d in enumerate(m["docs"])}
def find(n):
    for d in dirs:
        f = f"{d}/{n}.json.rows"
        if os.path.exists(f): return f
print("| doc | read | onset (into gen) | pre KL / row | post KL / row | rows after with KL < 0.1 | top-1 agree pre → post | max row KL, rows > 5 |\n|---|---|---|---|---|---|---|---|")
for o, d in D.items():
    i = orig[int(o)]; x = m["docs"][i]; q = slice(x["row0"], x["row0"] + x["tokens"] - 1 - x["score_from"]); gg = gen[q]; idx = np.where(gg)[0]
    for n, r in d["reads"].items():
        f = find(n)
        if f is None or r["onset_pos"] is None: continue
        K = np.fromfile(f, np.float32).reshape(-1, 4)[q][idx]; p = np.searchsorted(x["score_from"] + idx, r["onset_pos"])
        pre, post = K[:p], K[p:]
        if len(pre) == 0: pre = post[:1]
        print(f"| {o} | {n} | {r['onset_in_generation']} | {pre[:, 0].mean():.3f} | {post[:, 0].mean():.3f} | {(post[:, 0] < 0.1).mean() * 100:.0f} % | {pre[:, 3].mean():.2f} → {post[:, 3].mean():.2f} | {K[:, 0].max():.1f}, {(K[:, 0] > 5).sum()} |")
