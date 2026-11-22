#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""tokfreq222.py OUT.npz [--golden J217_GOLDEN] -- token-frequency bins for the queue readout (w222, user 11-22). Unigram counts of the TARGET tokens (the token each golden row predicts) over the QAD
training data: the gen rows of w214's three chunk goldens (xc8k-b2-c0 / c1, tr32k-b2-c0 with w214's gen masks) + the gen rows of the calib documents of pool train214. Bins = occurrence-weighted
terciles: tokens sorted by count, `rare` = the rarest tokens that together make up the first third of all training occurrences, `mid` the next third, `common` the last third (count 0 = rare).
OUT.npz: counts [n_vocab] uint32, binof [n_vocab] uint8 (0 rare / 1 mid / 2 common), edges (count thresholds), rowbin [rows of the golden] uint8 (bin of each row's target token; 255 for the
doc's last position), plus the share of the golden's gen rows in each bin (printed)."""
import json, os, sys
import numpy as np

W14 = os.environ.get("W14", "/data/q27/w214"); CAL = os.environ.get("CALIB", "/data/q27/challenge/golden/calib")
J = sys.argv[sys.argv.index("--golden") + 1] if "--golden" in sys.argv else "/data/q27/w217/slices/golden-j217"


def targets(d, gen, only=None):
    """target token ids of every gen row (and the row's golden index) of golden dir d; only = doc indices"""
    m = json.load(open(f"{d}/kld.json")); tok = np.memmap(f"{d}/tokens.i32", np.int32, "r"); off = np.concatenate([[0], np.cumsum([x["tokens"] for x in m["docs"]])])
    out = []
    for i, x in enumerate(m["docs"]):
        if only is not None and i not in only: continue
        n = x["tokens"] - 1 - x["score_from"]
        if n <= 0: continue
        r = x["row0"] + np.arange(n); g = gen[r].astype(bool)
        out.append(np.asarray(tok[off[i] + x["score_from"] + 1: off[i] + x["tokens"]], np.int64)[g])
    return np.concatenate(out), m


nv = json.load(open(f"{J}/kld.json"))["n_vocab"]; counts = np.zeros(nv, np.int64)
for c in ("xc8k-b2-c0", "xc8k-b2-c1", "tr32k-b2-c0"):
    t, _ = targets(f"{W14}/gold/{c}", np.fromfile(f"{W14}/genmask/{c}.u8", np.uint8)); counts += np.bincount(t, minlength=nv); print(c, len(t), "rows")
pool = json.load(open(f"{W14}/pools214.json"))["pool"]; tr = {i for i, p in enumerate(pool) if p == "train214"}
t, _ = targets(CAL, np.fromfile(f"{CAL}/gen.u8", np.uint8), tr); counts += np.bincount(t, minlength=nv); print("calib train214", len(t), "rows")
order = np.argsort(counts, kind="stable"); cum = np.cumsum(counts[order]) / counts.sum()
binof = np.zeros(nv, np.uint8); binof[order[cum > 1 / 3]] = 1; binof[order[cum > 2 / 3]] = 2
edges = [int(counts[order[np.searchsorted(cum, q)]]) for q in (1 / 3, 2 / 3)]
mj = json.load(open(f"{J}/kld.json")); tj = np.fromfile(f"{J}/tokens.i32", np.int32); offj = np.concatenate([[0], np.cumsum([x["tokens"] for x in mj["docs"]])]); genj = np.fromfile(f"{J}/gen.u8", np.uint8).astype(bool)
rowbin = np.full(mj["rows"], 255, np.uint8)
for i, x in enumerate(mj["docs"]):
    n = x["tokens"] - 1 - x["score_from"]
    if n > 0: rowbin[x["row0"]: x["row0"] + n] = binof[tj[offj[i] + x["score_from"] + 1: offj[i] + x["tokens"]]]
sh = [float((rowbin[genj] == b).mean()) for b in range(3)]
print("count edges (rare <= %d < mid <= %d < common)" % tuple(edges), "| distinct tokens per bin", [int(((binof == b) & (counts > 0)).sum()) for b in range(3)], "| zero-count tokens", int((counts == 0).sum()))
print("j217 gen rows by bin (rare / mid / common):", [round(s, 4) for s in sh])
np.savez(sys.argv[1], counts=counts.astype(np.uint32), binof=binof, edges=np.array(edges), rowbin=rowbin, share=np.array(sh))
