#!/usr/bin/env python3
"""unigram96.py calib GOLDEN POOLS-V3.json OUT.npy | xcorp TOKENS.i32 OUT.npy -- token counts (no logits) for prior U:
calib = the tokens of pools-v3's fit and qad documents (search and verify left out); xcorp = every token of the file."""
import json, sys, numpy as np
V = 248320
if sys.argv[1] == "calib":
    m = json.load(open(f"{sys.argv[2]}/kld.json")); P = json.load(open(sys.argv[3]))["pool"]
    t = np.fromfile(f"{sys.argv[2]}/tokens.i32", np.int32); off = 0; keep = []
    for i, d in enumerate(m["docs"]):
        if P[i] in ("fit", "qad"): keep.append(t[off:off + d["tokens"]])
        off += d["tokens"]
    assert off == len(t); t = np.concatenate(keep); out = sys.argv[4]
else:
    t = np.fromfile(sys.argv[2], np.int32); out = sys.argv[3]
c = np.bincount(t, minlength=V).astype(np.int64); assert len(c) == V
np.save(out, c); print(f"{len(t)} tokens, {int((c == 0).sum())} vocab rows never seen")
