#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cmpgate222.py RUN_A RUN_B --ck-a ckpt-2e+05 --ck-b ckpt-2e+05 [--b-prefer-branch] -- bit comparison of two QAD runs' training records and one checkpoint each (w222 gates): every train.jsonl record of RUN_A
(main and branch rows, `what` ignored) is matched by plan `step` (and rows) with RUN_B's (with --b-prefer-branch its `branch` record when it has one; default: the record of the same kind main / branch) and compared on lr, train_kl, ema, per-corpus KL, n, rows, gnorm (exact equality; wall / pos_s / peak ignored); the checkpoint
scales (`scales/*.safetensors`: r, c) are compared tensor by tensor. Prints the verdict: IDENTICAL or the first differences. RUN_A's records not present in RUN_B (e.g. a longer run) are listed, not failed."""
import glob, json, os, sys
import torch
from safetensors import safe_open
a = sys.argv[1:]; ra, rb = a[0], a[1]; cka = a[a.index("--ck-a") + 1]; ckb = a[a.index("--ck-b") + 1]
KEYS = ("rows", "lr", "train_kl", "ema", "by", "n", "gnorm")
def recs(r):
    out = {}
    for l in open(f"{r}/train.jsonl"):
        x = json.loads(l)
        if "step" in x and "train_kl" in x and ("what" in x): out.setdefault(x["step"], []).append(x)
    return out
A, B = recs(ra), recs(rb); bad = []; n = 0; only = []
for st, xs in sorted(A.items()):
    if st not in B: only.append(st); continue
    for x in xs:
        # B's record of this step: the branch record when there is one (a branch is compared with a standalone run's main records), else the main record
        ys = [y for y in B[st] if y["rows"] == x["rows"]]
        pref = "branch" if "--b-prefer-branch" in a else x["what"].split()[0]
        y = next((y for y in ys if y["what"].startswith(pref)), ys[0] if ys else None)
        if y is None: bad.append((st, "no record with the same rows")); continue
        n += 1
        d = [k for k in KEYS if x.get(k) != y.get(k) and not (k == "ema" and "--b-prefer-branch" in a)]      # a branch record shows the main run's running average (a log field, not training state)
        if d: bad.append((st, x["what"], y["what"], {k: (x.get(k), y.get(k)) for k in d}))
sa, sb = f"{ra}/{cka}/scales", f"{rb}/{ckb}/scales"; nt = 0; sbad = []
for f in sorted(glob.glob(f"{sa}/*.safetensors")):
    g = f"{sb}/{os.path.basename(f)}"
    with safe_open(f, "pt") as z1, safe_open(g, "pt") as z2:
        for k in z1.keys():
            nt += 1
            if not torch.equal(z1.get_tensor(k), z2.get_tensor(k)): sbad.append((os.path.basename(f), k))
print(f"records compared {n} (steps only in A: {only}); scale tensors compared {nt}")
print("record differences:", bad[:6] if bad else "none"); print("scale differences:", sbad[:6] if sbad else "none")
print("VERDICT:", "IDENTICAL (bit for bit)" if not bad and not sbad and n and nt else "DIFFERENT")
