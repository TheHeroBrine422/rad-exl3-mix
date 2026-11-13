#!/usr/bin/env python3
"""tprof.py LOG [LOG ...] -- tprof.sh's --profile-ops dumps -> the a/b projections' share of a prefill step and of a decode
step (w142, #178), next to the trunk's activation quantizer per call (what an int8 a/b's own had_quant_act_i8g would
add, one per GDN layer). The first dump (step 4) holds PREFILL_STEPS prefill chunks (env, default 2: a 9,001-token prompt
at 8,192 batched tokens) and 4 - PREFILL_STEPS decode steps; decode = (last - first) / steps between; prefill = the first
dump minus its decode steps."""
import os, re, sys
PS = int(os.environ.get("PREFILL_STEPS", "2"))

ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(host|dev)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")


def dumps(path):
    out, cur = [], None
    for l in open(path, errors="replace"):
        if "per-op timing, rank 0" in l: cur = {}; out.append(cur); continue
        if cur is None: continue
        m = ROW.match(l)
        if m: cur[(m.group(7), m.group(8), int(m.group(9)), m.group(10))] = (int(m.group(1)), float(m.group(2)))
    return out


def cat(k):
    op, geo, _, w = k
    if re.search(r"\.ssm_[ab]\.weight$", w) and op.startswith("gemm"): return "ab"
    if "actq" in op or "quant_act" in op: return "actq"
    return "other"


for path in sys.argv[1:]:
    D = dumps(path)
    if len(D) < 2: print(f"{path}: {len(D)} dumps"); continue
    first, last = D[0], D[-1]
    def tot(d, c=None): return sum(v[1] for k, v in d.items() if c is None or cat(k) == c)
    def calls(d, c): return sum(v[0] for k, v in d.items() if cat(k) == c)
    steps = (len(D) - 1) * 4
    dec = {c: (tot(last, c) - tot(first, c)) / steps for c in ("ab", "actq", None)}
    pre = {c: tot(first, c) - (4 - PS) * dec[c] for c in ("ab", "actq", None)}
    n_ab = sum(1 for k in first if cat(k) == "ab")
    n_q = sum(1 for k in first if cat(k) == "actq")
    dq_calls = (calls(last, "actq") - calls(first, "actq")) / steps
    print(f"{path}: {len(D)} dumps, {steps} decode steps between first and last; {n_ab} a/b gemm ops, {n_q} activation-quantizer ops")
    for name, d in ((f"prefill ({PS} chunks, 9k tokens)", pre), ("decode (per step)", dec)):
        print(f"  {name:30s} all ops {d[None]:9.3f} ms   a/b gemm {d['ab']:8.3f} ms ({100 * d['ab'] / d[None]:.2f} %)"
              f"   activation quantizers {d['actq']:8.3f} ms ({100 * d['actq'] / d[None]:.2f} %, "
              f"{1000 * d['actq'] / max(1, n_q):.1f} us per op)")
    print("  a/b gemm rows (first dump):", sorted({(k[0], k[1]) for k in first if cat(k) == "ab"}))
    print("  quantizer rows (first dump):", sorted({(k[0], k[1]) for k in first if cat(k) == "actq"})[:4])
