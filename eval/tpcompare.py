#!/usr/bin/env python3
"""tpcompare.py FULLGOLDEN SUBGOLDEN SUB.f32 FULL.f32 -- per-document agreement between a sub-golden run (e.g. TP2)
and the same documents' rows in a full-golden run (e.g. TP1). Rows are the recorder's {kl, top1, nll_q, nll_p}.
Only the rows the golden's gen.u8 marks are compared (as the official score). Documents are matched by (source, tokens) in order; a document cut differently in the full run is skipped."""
import json, sys
import numpy as np
fg, sg, sp, fp = sys.argv[1:5]
fm, sm = json.load(open(f"{fg}/kld.json")), json.load(open(f"{sg}/kld.json"))
S = np.fromfile(sp, np.float32).reshape(-1, 4).astype(np.float64)
F = np.fromfile(fp, np.float32).reshape(-1, 4).astype(np.float64)
assert len(S) == sm["rows"], (len(S), sm["rows"])
SG = np.fromfile(f"{sg}/gen.u8", np.uint8).astype(bool)   # the rows the official score counts
used = set()
print(f"{'source':24} {'tokens':>7} {'kl sub':>9} {'kl full':>9} {'rel':>8} {'|dkl| p99':>10} {'top1 agree':>10}")
for d in sm["docs"]:
    n = d["tokens"] - 1 - d["score_from"]
    j = next((k for k, e in enumerate(fm["docs"]) if k not in used and e["source"] == d["source"]
              and e["tokens"] == d["tokens"] and e["score_from"] == d["score_from"]), None)
    if j is None:
        print(f"{d['source']:24} {d['tokens']:>7}  (no identical document in the full run)"); continue
    used.add(j)
    mk = SG[d["row0"]:d["row0"] + n]
    s = S[d["row0"]:d["row0"] + n][mk]; f = F[fm["docs"][j]["row0"]:fm["docs"][j]["row0"] + n][mk]
    if (f == 0).all(1).any():
        print(f"{d['source']:24} {d['tokens']:>7}  (full run did not score it)"); continue
    dk = np.abs(s[:, 0] - f[:, 0])
    print(f"{d['source']:24} {d['tokens']:>7} {s[:,0].mean():9.6f} {f[:,0].mean():9.6f} "
          f"{(s[:,0].mean()/f[:,0].mean()-1)*100:+7.2f}% {np.percentile(dk,99):10.2e} {(s[:,1]==f[:,1]).mean():10.4f}")
