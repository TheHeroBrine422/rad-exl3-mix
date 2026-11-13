"""kldcmp.py RES -- #170: the ROCm 10.1 build's KLD rows against the 7.14 build's on the same pack and golden (val10.sh).
Rows: per position kl, nll_q, nll_p, top1 (float32 x 4, core/kld.cpp). Per pair: the means, |d mean KLD|, the max and
mean |d| per position, the share of positions with d = 0, top-1 flips, byte-identity. (w130)"""
import json, os, sys
import numpy as np

R = sys.argv[1]
PAIRS = [("V2-r7f", "V2-r7"), ("V2-r7", "V2-r10"), ("K4-r7", "K4-r10"), ("K4-r10", "K4-r10p"), ("K4H-r7", "K4H-r10"),
         ("K4H-r10", "K4H-r10x"), ("K4A-r7-f0", "K4A-r10-f0"), ("K4A-r10-f0", "K4A-r10-f1")]


def load(t):
    p = f"{R}/kld-{t}.json"
    if not os.path.exists(p + ".rows"):
        return None
    j = json.load(open(p))["all"]
    return j, np.fromfile(p + ".rows", np.float32).reshape(-1, 4)


print(f"{'A':12} {'B':12} {'pos':>7} {'KLD A':>10} {'KLD B':>10} {'|dmean|':>9} {'max|d|':>9} {'mean|d|':>9} {'d=0':>6} "
      f"{'top1 A':>7} {'top1 B':>7} {'flips':>5} bytes")
for a, b in PAIRS:
    A, B = load(a), load(b)
    if A is None or B is None:
        continue
    (ja, ra), (jb, rb) = A, B
    if ra.shape != rb.shape:
        print(a, b, "shape differs", ra.shape, rb.shape); continue
    d = np.abs(ra[:, 0].astype(np.float64) - rb[:, 0])
    ident = open(f"{R}/kld-{a}.json.rows", "rb").read() == open(f"{R}/kld-{b}.json.rows", "rb").read()
    print(f"{a:12} {b:12} {len(ra):7d} {ja['kld']['mean']:10.6f} {jb['kld']['mean']:10.6f} "
          f"{abs(ja['kld']['mean'] - jb['kld']['mean']):9.2e} {d.max():9.2e} {d.mean():9.2e} {np.mean(d == 0):6.3f} "
          f"{ja['top1_agreement']:7.4f} {jb['top1_agreement']:7.4f} {int((ra[:, 3] != rb[:, 3]).sum()):5d} "
          f"{'SAME' if ident else 'differ'}")
