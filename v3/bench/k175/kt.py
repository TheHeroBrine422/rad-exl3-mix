#!/usr/bin/env python3
"""kt.py RES ROW... -- #175 (w137) device dispatch timestamps of unprofiled 8k prefills (rp.sh, RADIANCE_AQL_STAMP): kernels split into bursts
(idle > 100 ms between requests), the last 3 bursts with > 500 kernels are the measured prefills. Per prefill (mean):
span (first start .. last end), busy (union of kernel intervals), sum of kernel durations, overlap (sum - busy: kernels
running concurrently), idle (span - busy: launch gaps / host waits), queues used; and the kernel time by category
(kernel-name patterns), the top kernels by name."""
import csv, glob, re, sys
from collections import defaultdict

CATS = [("GDN", r"gdn|chunk_scan|kkt|conv_prep|delta|ssm_scan"), ("attention", r"attn|paged|flash|rope|kv_store"),
        ("actq / quant", r"actq|quant|had_"), ("GEMM", r"gemm|exl3i_a8|exl3i_|wdec|8a8|iqr|mmq|wmma"),
        ("norm", r"rmsnorm|norm"), ("elementwise", r"add|mul|silu|sigmoid|elementwise"), ("other", r".")]


def cat(n):
    for c, p in CATS:
        if re.search(p, n, re.I):
            return c


def bursts(ks, gap=100e6):
    out, cur, end = [], [], 0
    for k in ks:
        if cur and k[0] - end > gap:
            out.append(cur); cur, end = [], 0
        cur.append(k); end = max(end, k[1])
    if cur:
        out.append(cur)
    return out


def union(ks):
    tot, s0, e0 = 0, None, None
    for s, e, *_ in sorted(ks):
        if e0 is None or s > e0:
            if e0 is not None:
                tot += e0 - s0
            s0, e0 = s, e
        else:
            e0 = max(e0, e)
    return tot + (e0 - s0 if e0 is not None else 0)


res, rows = sys.argv[1], sys.argv[2:]
S = {}
for r in rows:
    ks = []
    for d in csv.DictReader(open(f"{res}/stamps-{r}.csv")):   # RADIANCE_AQL_STAMP: pass,seq,queue,kernel,...,start_ns,end_ns
        if int(d["end_ns"]) > int(d["start_ns"]) > 0:
            ks.append((int(d["start_ns"]), int(d["end_ns"]), d["kernel"], d["queue"]))
    ks.sort()
    bs = [b for b in bursts(ks) if len(b) > 500][-3:]
    st = defaultdict(float); byname = defaultdict(float)
    for b in bs:
        span = max(k[1] for k in b) - b[0][0]; busy = union(b); tot = sum(k[1] - k[0] for k in b)
        st["span"] += span / 3e6; st["busy"] += busy / 3e6; st["sum"] += tot / 3e6; st["n"] += len(b) / 3
        st["queues"] = max(st["queues"], len({k[3] for k in b}))
        for k in b:
            st["cat " + cat(k[2])] += (k[1] - k[0]) / 3e6
            byname[re.sub(r"\(.*", "", k[2])[:90]] += (k[1] - k[0]) / 3e6
    st["overlap"] = st["sum"] - st["busy"]; st["idle"] = st["span"] - st["busy"]
    S[r] = (st, byname, len(bs))
print(f"{'ms per 8k prefill (unprofiled, AQL stamps)':44s}" + "".join(f"{r:>12s}" for r in rows))
for k in ("n", "queues", "span", "busy", "sum", "overlap", "idle"):
    print(f"{k:44s}" + "".join(f"{S[r][0][k]:12.1f}" for r in rows))
for c, _ in CATS:
    print(f"{'  ' + c:44s}" + "".join(f"{S[r][0]['cat ' + c]:12.1f}" for r in rows))
for r in rows:
    print(f"\n{r}: {S[r][2]} prefills; top kernels (ms per prefill)")
    for n, t in sorted(S[r][1].items(), key=lambda x: -x[1])[:14]:
        print(f"  {t:8.1f}  {n}")
