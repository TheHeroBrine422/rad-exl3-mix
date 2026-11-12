#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""timing.py SCORES.jsonl RADMIX.jsonl OUT.json [key=value ...] -- the search-eval cost model for protocol.py (w123, #161).

SCORES.jsonl: tscore.sh lines (rc 0 only; each names its golden, whose kld.json gives the tokens prefilled).
RADMIX.jsonl: {"radmix", "s", "bytes"} lines (moves.sh). OUT: s_per_tok (median score phase / tokens), run_s (median
start + init + load + prep + exit: one radiance process), mix_s (median radmix), plus the slice defaults below
(override with key=value)."""
import json, sys

import numpy as np

sc, mx, out = sys.argv[1:4]
tok = {}
S = []
for l in open(sc):
    j = json.loads(l)
    if j.get("rc") != 0 or j.get("score") is None: continue
    g = j["golden"]
    if g not in tok: tok[g] = sum(d["tokens"] for d in json.load(open(f"{g}/kld.json"))["docs"])
    S.append((j["score"] / tok[g], sum(j[k] for k in ("start", "init", "load", "prep", "exit")), tok[g], j["score"]))
M = [json.loads(l)["s"] for l in open(mx)]
T = {"s_per_tok": float(np.median([s[0] for s in S])), "run_s": float(np.median([s[1] for s in S])), "mix_s": float(np.median(M)),
     "n_runs": len(S), "n_mix": len(M), "tok_per_s": float(np.median([s[2] / s[3] for s in S])),
     "runs_per_eval": 2, "decisions_per_slice": 20, "x_docs": 130, "calib_docs": 50, "calib_tok": 45000, "x_tok_cap": 1400}
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1); T[k] = type(T.get(k, 0.0))(v) if k in T else float(v)
json.dump(T, open(out, "w"), indent=1); print(json.dumps(T))
