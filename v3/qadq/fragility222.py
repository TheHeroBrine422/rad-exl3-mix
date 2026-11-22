#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fragility222.py GOLDEN SCORES_DIR[,SCORES_DIR..] NAME[,NAME..] [--docs 3820,1893,1968,3947] -- the blow-up fragility table (w222 D1 follow-up). For each served read (NAME = the score name in a scores dir:
`ctl-2.50`, `UN2.00`, `FP25-s0-ckpt-5e+05`, `FV40-gdn-ckpt-5e+05`, `25A-ckpt-5e+05` ...) and each document: KL / gen row over the whole document, before and after the document's reference onset (the position
where the derailing checkpoints broke it: 3820 token 200, 1893 token 250, 1968 token 220, 3947 token 192; D1), the read's own onset (first gen position whose 32-gen-row rolling mean KL is >= 0.3 and >= 5x the
document's median rolling KL before it) and the KL / row after that onset. A read derails on a document when its KL / row after the reference onset is >= 0.3 (the derailments of D1 sit at 1-2.5; the mild ones at 0.1-0.25)."""
import json, os, sys
import numpy as np
g, dirs, names = sys.argv[1:4]; a = sys.argv[4:]
docs = [int(x) for x in (a[a.index("--docs") + 1] if "--docs" in a else "3820,1893,1968,3947").split(",")]
REF = {3820: 200, 1893: 250, 1968: 220, 3947: 192}
m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool); orig = {int(d["orig"]): i for i, d in enumerate(m["docs"])}
def find(n):
    for d in dirs.split(","):
        f = f"{d}/{n}.json.rows"
        if os.path.exists(f): return f
print("| read | " + " | ".join(f"doc {d}: whole / before / after ref {REF[d]} / own onset (KL after)" for d in docs) + " |\n|---|" + "---|" * len(docs))
for n in names.split(","):
    f = find(n)
    if f is None: print(f"| {n} | " + " | ".join("(not read yet)" for _ in docs) + " |"); continue
    K = np.fromfile(f, np.float32).reshape(-1, 4)[:, 0].astype(np.float64); cells = []
    for d in docs:
        i = orig[d]; x = m["docs"][i]; q = slice(x["row0"], x["row0"] + x["tokens"] - 1 - x["score_from"]); gg = gen[q]; idx = np.where(gg)[0]; pos = x["score_from"] + idx; kk = K[q][idx]
        b = pos < REF[d]; w, bf, af = kk.mean(), (kk[b].mean() if b.any() else float("nan")), (kk[~b].mean() if (~b).any() else float("nan"))
        own = "none"
        if len(kk) >= 64:
            ra = np.convolve(kk, np.ones(32) / 32, "valid"); base = np.median(ra[:max(1, len(ra) // 4)]) if len(ra) else 0
            j = np.where((ra >= 0.3) & (ra >= 5 * max(base, 1e-3)))[0]
            if len(j): p = int(pos[j[0]]); own = f"{p} ({kk[j[0]:].mean():.2f})"
        cells.append(f"{w:.3f} / {bf:.3f} / **{af:.3f}** / {own}" if af >= 0.3 else f"{w:.3f} / {bf:.3f} / {af:.3f} / {own}")
    print(f"| {n} | " + " | ".join(cells) + " |")
