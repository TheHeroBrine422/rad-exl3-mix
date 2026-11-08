#!/usr/bin/env python3
"""readout.py O -- README.md's rule for #89-06c from the score files (O = /data/yb/w79; d1105's under /data/yb/d1105).
Per seed: healed / stack KL ratio on g08-search1 and on g08-verify1 + verify2 pooled (gen rows of both), then the
verdict for K4 (K5: seed 0's 1.012 at the held-out-best lr already fails, unless step 0 flagged the fold path)."""
import os, re, sys
import numpy as np
O = sys.argv[1]; D5 = "/data/yb/d1105"
def f32(t):
    for d in (O, D5):
        if os.path.exists(f"{d}/{t}.f32"): return f"{d}/{t}.f32"
def kl(t):
    f = f32(t)
    if f is None: return None
    a = np.fromfile(f, np.float32).reshape(-1, 4); g = a[:, 2] > 0
    return float(a[g, 0].sum()), int(g.sum())
res = open(f"{O}/result.txt").read()
m = re.findall(r"K4 lr picked by held-out: (\S+)", res); pick = m[-1] if m else None
print(f"#89-06c readout (K4 lr {pick})")
ok_all, inconcl = True, False
for s in ("", "_s1"):
    b, h = f"k4edyswjasym_r0{s}", f"k4edyswjasym_r0_lr{pick}{s}"
    rs = (kl(h)[0] / kl(b)[0]) if kl(h) and kl(b) else None
    v = [(kl(b + x), kl(h + x)) for x in (".v1", ".v2")]
    rv = sum(y[0] for _, y in v) / sum(x[0] for x, _ in v) if all(x and y for x, y in v) else None
    print(f"seed {s or '_s0'}: search {rs if rs is None else round(rs, 4)}, verify1+2 {rv if rv is None else round(rv, 4)}")
    if rs is None or rv is None: ok_all = None; continue
    if not (rs <= 0.95 and rv <= 0.97): ok_all = False
    if rs <= 0.97: inconcl = True
if ok_all is None: print("verdict: incomplete")
elif ok_all: print("verdict: HEAL K4 (both seeds <= 0.95 on search and <= 0.97 on verify)")
elif inconcl: print("verdict: small (0.95-0.97 band on some seed): K4 healing worth <= ~5%, lead decides with its cost")
else: print("verdict: do not heal K4")
