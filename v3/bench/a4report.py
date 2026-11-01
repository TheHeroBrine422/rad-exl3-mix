#!/usr/bin/env python3
"""a4report.py A4TIME.jsonl [SPEEDCAL.jsonl] -- int8 path (with decode) vs f16 T2, per shape / K / M; the decode share
at M 2048 from the profiled runs; next to speedcal's bare matmul ratio (what a decode-free int8 GEMM would give)."""
import json, sys, collections
R = {}; P = {}
for l in open(sys.argv[1]):
    d = json.loads(l)
    if "ms" in d: R[(d["path"], d["shape"], d["bits"], d["M"])] = d["ms"]; P[(d["path"], d["shape"], d["bits"])] = d.get("kern_ms") or P.get((d["path"], d["shape"], d["bits"]))
SC = {}
if len(sys.argv) > 2:
    for l in open(sys.argv[2]):
        d = json.loads(l)
        if "ms" in d and d["split"] == "tp1": SC[(d["kern"], d["shape"], d["M"])] = d["ms"]
shapes = list(dict.fromkeys(k[1] for k in R))
print("f16 / nfq8 time ratio (>1: int8 faster), K2..6, at M 512 / 2048 / 4096")
for s in shapes:
    for M in (512, 2048, 4096):
        row = [f"{R[('f16', s, b, M)] / R[('nfq8', s, b, M)]:.2f}" if ('nfq8', s, b, M) in R and ('f16', s, b, M) in R else "  - " for b in range(2, 7)]
        bare = f"  bare matmul x{SC[('f16', s, M)] / SC[('iu8', s, M)]:.2f}" if ('iu8', s, M) in SC else ""
        print(f"  {s:8s} M{M:<5d} " + "  ".join(row) + bare)
print("decode share of the path's GPU time at M 2048 (wdec kernels / all), K2..6")
for path in ("f16", "nfq8"):
    for s in shapes:
        row = []
        for b in range(2, 7):
            k = P.get((path, s, b))
            if not k: row.append("  - "); continue
            dec = sum(v for n, v in k.items() if "wdec" in n); row.append(f"{dec / sum(k.values()):.2f}")
        print(f"  {path:4s} {s:8s} " + "  ".join(row))
