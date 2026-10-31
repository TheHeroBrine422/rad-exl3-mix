#!/usr/bin/env python3
"""aggscore.py GOLDEN SCORE.f32 -- score.py's Accumulator report from the patched recorder's per-row
{kl, top1, nll_q, nll_p}, over the rows GOLDEN's gen.u8 marks; plus per-source means."""
import json, os, sys
import numpy as np
g, sp = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(g, "kld.json")))
s = np.fromfile(sp, np.float32).reshape(-1, 4).astype(np.float64)
assert len(s) == m["rows"], (len(s), m["rows"])
# a row the recorder never reached is all zeros (nll_p, the golden's own NLL, is never exactly 0 on a
# scored row): a crashed run must not average them in as perfect rows
unscored = int((s == 0).all(1).sum())
if unscored:
    sys.exit(f"{sp}: {unscored} of {len(s)} rows unscored -- the run did not finish")
gen = np.fromfile(os.path.join(g, "gen.u8"), np.uint8).astype(bool) if os.path.exists(os.path.join(g, "gen.u8")) else np.ones(len(s), bool)
src = np.empty(len(s), dtype=object)
pos = np.zeros(len(s), np.int64)      # the scored position's index in its document
dlen = np.zeros(len(s), np.int64)     # its document's token count
for d in m["docs"]:
    n = d["tokens"] - 1 - d["score_from"]; src[d["row0"]:d["row0"] + n] = d["source"]
    pos[d["row0"]:d["row0"] + n] = d["score_from"] + 1 + np.arange(n); dlen[d["row0"]:d["row0"] + n] = d["tokens"]
kl, t1, nq, np_ = s[gen, 0], s[gen, 1] > 0.5, s[gen, 2], s[gen, 3]
sg = src[gen]
long_ = np.array([str(x).startswith("agent-trace/") for x in sg])
rep = {"positions": int(kl.size),
       "kld": {"mean": float(kl.mean()), "median": float(np.median(kl)),
               "p99": float(np.quantile(kl, 0.99)), "p99_9": float(np.quantile(kl, 0.999))},
       "top1_agreement": float(t1.mean()),
       "ppl": {"candidate": float(np.exp(nq.mean())), "reference": float(np.exp(np_.mean())),
               "ratio": float(np.exp(nq.mean() - np_.mean()))},
       "kld_mean_long_agentic": float(kl[long_].mean()) if long_.any() else None,
       "nonfinite_rows": int((~np.isfinite(s[:, 0])).sum())}
# KLD by position in the document: error that builds up over context shows as a rising mean
edges = [0, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
pg, lg = pos[gen], dlen[gen]
def bins(sel):
    out = {}
    for a, b in zip(edges[:-1], edges[1:]):
        k = sel & (pg >= a) & (pg < b)
        if k.any(): out[f"{a}-{b}"] = {"n": int(k.sum()), "kld": float(kl[k].mean()), "top1": float(t1[k].mean())}
    return out
rep["by_position"] = bins(np.ones(len(kl), bool))
rep["by_position_long_docs"] = bins(lg > 16384)
rep["by_position_short_docs"] = bins(lg <= 16384)
rep["by_source"] = {x: {"n": int((sg == x).sum()), "kld": float(kl[sg == x].mean())} for x in sorted(set(sg))}
print(json.dumps(rep, indent=1))
