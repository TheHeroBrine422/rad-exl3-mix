#!/usr/bin/env python3
"""smallops.py CSV... -- per stamped decode run (the most frequent pass kind): ms a pass of GEMVs (exl3i_gemm_nt with
> 1 launch, exl3i_mad_gemv, iqr_gemv with > 1 launch), head (the one-launch GEMV / logits / vocab kernels), bf16 a / b
(r4d_gemm_bf16_nt), attention core (attn decode + combine), and every other kernel (the small ops). (w152, #179)"""
import csv, statistics as st, sys
from collections import defaultdict
for path in sys.argv[1:]:
    P = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")): P[int(r["pass"])].append(r)
    cnt = defaultdict(int)
    for v in P.values(): cnt[int(v[0]["records"])] += 1
    rec = max(cnt, key=cnt.get); full = max(len(v) for v in P.values() if int(v[0]["records"]) == rec)
    agg = defaultdict(list)
    for v in P.values():
        if int(v[0]["records"]) != rec or len(v) != full or any(not r["end_ns"] for r in v): continue
        k = defaultdict(float); n = defaultdict(int)
        for r in v: k[r["kernel"]] += (int(r["end_ns"]) - int(r["start_ns"])) / 1e6; n[r["kernel"]] += 1
        c = defaultdict(float)
        for name, t in k.items():
            if "logits_gemm" in name or "vocab_" in name or (("exl3i_gemm_nt" in name or "iqr_gemv" in name) and n[name] == 1): c["head"] += t
            elif "exl3i_gemm_nt" in name or "exl3i_mad_gemv" in name or "iqr_gemv" in name: c["gemv"] += t
            elif "r4d_gemm_bf16_nt" in name: c["ab_bf16"] += t
            elif "attn_decode" in name or "splitkv_combine" in name: c["attn"] += t
            else: c["small"] += t
        for f in ("gemv", "head", "ab_bf16", "attn", "small"): agg[f].append(c[f])
    print(f"{path.rsplit('/', 1)[-1][:44]:44s} " + "  ".join(f"{f} {st.median(agg[f]):6.3f}" for f in ("gemv", "head", "ab_bf16", "attn", "small")) + f"  (passes {len(agg['gemv'])})")
