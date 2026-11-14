#!/usr/bin/env python3
"""mkmap.py KIND A B OUT.npy -- exl3hashm output maps (int8 [256], q = MAP[S], S = the hash's byte sum) for #177 (w139).
KIND: hg       Gaussian quantiles of S's mid-CDF over the 2^16 states (hscreen's hg), scaled to the plain hash's sd
      hc:C     cubic shaping v + C v^3 of the standardised S (hscreen's hc), scaled to the plain hash's sd
Both keep the codebook's sd equal to the plain hash's (q = S - 126: ~36.95 int8 steps, value sd ~1.000), clip +-127.
S values the 2^16 states never reach map to 0."""
import sys
import numpy as np
from statistics import NormalDist

kind, A, B, out = sys.argv[1], int(sys.argv[2], 16), int(sys.argv[3], 16), sys.argv[4]
s = np.arange(65536, dtype=np.uint64)
h = ((s * A + B) % (1 << 32)).astype(np.uint32) & np.uint32(0x3F3F3F3F)
S = sum(((h >> (8 * k)) & 0xFF).astype(np.int64) for k in range(4))
q0 = (S - 126).astype(np.float64); sd0 = q0.std()
vals, cnt = np.unique(S, return_counts=True)
if kind == "hg":
    c = np.cumsum(cnt) / cnt.sum() - cnt / cnt.sum() / 2
    nd = NormalDist(); v = np.array([nd.inv_cdf(min(max(p, 1e-9), 1 - 1e-9)) for p in c])
elif kind.startswith("hc:"):
    C = float(kind[3:]); z = (vals - S.mean()) / S.std(); v = z + C * z ** 3
else:
    raise SystemExit(f"kind {kind}")
w = cnt / cnt.sum(); v = (v - (w * v).sum()) / np.sqrt((w * (v - (w * v).sum()) ** 2).sum())
m = np.zeros(256, np.int8); m[vals] = np.clip(np.rint(v * sd0), -127, 127).astype(np.int8)
q = m.astype(np.float64)[S]
np.save(out, m)
print(f"{kind}: A {A:#x} B {B:#x}: S {S.min()}..{S.max()}, q sd {q.std():.3f} (plain {sd0:.3f}), mean {q.mean():.3f}, "
      f"range {q.min():.0f}..{q.max():.0f}, distinct {len(np.unique(q))} -> {out}")
