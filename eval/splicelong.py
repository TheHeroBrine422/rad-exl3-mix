#!/usr/bin/env python3
"""splicelong.py FULLGOLDEN CAP CAP.f32 LONGGOLDEN LONG.f32 OUT.f32 -- one full-golden score file from a capped
run (every document cut to CAP tokens, as golden_pick.py --maxtok CAP lays it out) and a run of just the long documents whole (e.g. at --tp 2). Each full-golden
document is taken from LONG when LONGGOLDEN holds it (matched by source + tokens), else from the capped run, which
must then hold it uncut. Rows are the recorder's {kl, top1, nll_q, nll_p}; score OUT with aggscore.py FULLGOLDEN."""
import json, sys
import numpy as np
fg, cap, cp, lg, lp, op = sys.argv[1:7]
cap = int(cap)
fm, lm = (json.load(open(f"{g}/kld.json")) for g in (fg, lg))
cm, r = {"docs": []}, 0          # the capped golden's layout: each document cut to cap tokens, rows packed in order
for d in fm["docs"]:
    t = min(d["tokens"], cap); cm["docs"].append({"source": d["source"], "tokens": t, "row0": r}); r += t - 1 - d["score_from"]
cm["rows"] = r
C = np.fromfile(cp, np.float32).reshape(-1, 4); L = np.fromfile(lp, np.float32).reshape(-1, 4)
assert len(C) == cm["rows"] and len(L) == lm["rows"]
out = np.zeros((fm["rows"], 4), np.float32)
nlong = 0
for i, d in enumerate(fm["docs"]):
    n = d["tokens"] - 1 - d["score_from"]
    k = next((e for e in lm["docs"] if e["source"] == d["source"] and e["tokens"] == d["tokens"]), None)
    if k is not None:
        src, r0 = L, k["row0"]; nlong += 1
    else:
        c = cm["docs"][i]
        assert c["tokens"] == d["tokens"], (i, c, d)   # the capped run holds this one uncut
        src, r0 = C, c["row0"]
    out[d["row0"]:d["row0"] + n] = src[r0:r0 + n]
assert not (out == 0).all(1).any(), "rows left unscored"
out.tofile(op)
print(f"{op}: {fm['rows']} rows, {nlong} documents from the long run")
