#!/usr/bin/env python3
"""perfsum147.py PERFDIR [ROW...] -- (w147, #178-02) run147.sh's timing per row and round (w144's perfsum144.py fields):
static VRAM, 8k prefill wall (pq.py requests 1-3), TTFT at 8k and server ms per decode step at ctx 0 / 8k (bench.py runs
1-3, medians); from the --profile-ops table (2 x 8k prefill): all ops' total ms, the a / b projections (gemm_nt N 48 +
exl3i_dense N 128: calls, total ms, us a call), the GDN norms' AQ sites (exl3i_actq K 5120 with 2 / 3 planes; the
attention norm's 3-plane site is told apart by its consumers: 16 of them) and the GDN ops. One JSON line per (row, round),
then a median-over-rounds table."""
import collections, glob, json, os, re, statistics as st, sys
P = sys.argv[1]
rows = sys.argv[2:] or sorted({re.sub(r"^e2e-|-r\d+\.jsonl$", "", os.path.basename(f)) for f in glob.glob(f"{P}/e2e-*-r*.jsonl")})
LINE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+dev\s+\S+\s+\S+\s+(\S+)\s+(.*?)\s+#\d+\s+(\S+)\s*$")
def prof(path):
    agg = collections.defaultdict(lambda: [0, 0.0]); total = 0.0
    for l in open(path):
        m = LINE.match(l)
        if not m: continue
        calls, ms, op, par, w = int(m.group(1)), float(m.group(2)), m.group(4), m.group(5), m.group(6)
        total += ms
        N = re.search(r"\bN=(\d+)", par); N = int(N.group(1)) if N else None
        if op == "gemm_nt" and N == 48: k = "ab gemm_nt N48 (bf16)"
        elif op == "exl3i_dense" and "ssm_ab" in w: k = "ab exl3i_dense N128 (T4)"
        elif op == "exl3i_actq":
            pl = re.search(r"planes=(\d+)", par).group(1); K = re.search(r"\bK=(\d+)", par).group(1)
            k = f"exl3i_actq K{K} planes={pl}"
        elif op.startswith("gdn_"): k = op
        elif op == "rmsnorm": k = "rmsnorm"
        else: continue
        agg[k][0] += calls; agg[k][1] += ms
    return total, {k: {"calls": c, "ms": round(t, 3), "us_call": round(1000 * t / c, 2)} for k, (c, t) in sorted(agg.items())}
out = []
for row in rows:
    for f in sorted(glob.glob(f"{P}/e2e-{row}-r*.jsonl")):
        r = re.search(r"-r(\d+)\.jsonl$", f).group(1); t = f"{row}-r{r}"
        J = [json.loads(l) for l in open(f)]
        rec = {"row": row, "round": int(r)}
        for ctx in ("8k", "0"):
            ok = [j for j in J if j["ctx"] == ctx and not j.get("warmup")]
            rec[f"step_ms_{ctx}"] = st.median(j["srv_ms_per_decode_step"] for j in ok)
            if ctx == "8k": rec["ttft_ms_8k"] = st.median(j["client_ttft_ms"] for j in ok)
        pq = [float(m.group(1)) for m in re.finditer(r"req [123]: prompt \d+ tokens, ([\d.]+) s", open(f"{P}/bench-e2e-{t}.txt").read())]
        rec["prefill_s_8k"] = st.median(pq) if pq else None
        m = re.search(r"static weights\s+([\d.]+) GiB", open(f"{P}/serve-e2e-{t}.log").read()); rec["static_gib"] = float(m.group(1)) if m else None
        pf = f"{P}/serve-prof-{t}.log"
        if os.path.exists(pf): rec["prof_total_ms"], rec["prof"] = prof(pf)
        out.append(rec); print(json.dumps(rec))
print()
keys = ["static_gib", "prefill_s_8k", "ttft_ms_8k", "step_ms_0", "step_ms_8k", "prof_total_ms"]
print(f'{"row":>6} {"n":>2} ' + " ".join(f"{k:>13}" for k in keys))
for row in rows:
    R = [o for o in out if o["row"] == row]
    if R: print(f"{row:>6} {len(R):>2} " + " ".join(f"{st.median([o[k] for o in R if o.get(k) is not None]):>13.5g}" for k in keys))
print()
for row in rows:
    R = [o for o in out if o["row"] == row and "prof" in o]
    for k in sorted({k for o in R for k in o["prof"]}):
        v = [o["prof"][k] for o in R if k in o["prof"]]
        print(f"{row:>6} {k:<28} calls {v[0]['calls']:>6}  ms {st.median(x['ms'] for x in v):>9.3f}  us/call {st.median(x['us_call'] for x in v):>8.2f}")
