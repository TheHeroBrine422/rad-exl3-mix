#!/usr/bin/env python3
"""designcheck.py RADLS.jsonl MIX|CONTAINER ... -- can the long-trace arms carry longmine.py's regression at all? Per
arm the family bits (longmine.fambits, with the repo's copies standing in for iceman's /root/models paths), then the
spread of the two contrasts (attention - trunk, GDN - trunk bits) and the design's condition number."""
import json, os, sys
import numpy as np
sys.argv = [sys.argv[0], "/dev/null", sys.argv[1], "/dev/null"] + sys.argv[2:]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "longmine.py")).read()
ns = {"__name__": "lm"}; exec(src.split("# ---- rows keyed by")[0], ns)        # the bits part only
rows = []
for s in sys.argv[4:]:
    try: b = ns["fambits"](s)
    except SystemExit as e: print(f"{os.path.basename(s)}: skipped ({e})"); continue
    rows.append(b); print(f"{os.path.basename(s)[:40]:40s} trunk {b['trunk']:5.2f}  attn-trunk {b['attn'] - b['trunk']:+5.2f}  gdn-trunk {b['gdn'] - b['trunk']:+5.2f}  head {b.get('head', 0):5.2f}")
X = np.array([[1, b["trunk"], b["attn"] - b["trunk"], b["gdn"] - b["trunk"]] for b in rows])
print(f"{len(rows)} arms; contrast SD: attn {X[:, 2].std():.3f} bits, gdn {X[:, 3].std():.3f} bits; corr(attn, gdn contrasts) "
      f"{np.corrcoef(X[:, 2], X[:, 3])[0, 1]:+.2f}; corr(trunk, attn) {np.corrcoef(X[:, 1], X[:, 2])[0, 1]:+.2f}; "
      f"condition (centred) {np.linalg.cond(X - np.r_[0, X[:, 1:].mean(0)]):.1f}")
