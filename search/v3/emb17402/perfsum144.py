#!/usr/bin/env python3
"""perfsum144.py PERFDIR [ROW...] -- (w144, #174-02) perf144.sh's results per row and round: static VRAM (the load's
"static weights" line), the embedding's placement line, 8k prefill wall (pq.py requests 1-3; 0 is the warm-up), TTFT at
8k and server ms per decode step at ctx 0 / 8k (bench.py runs 1-3, medians), the embedding op's calls and us a call from
the --profile-ops table. One JSON line per (row, round) and a median-over-rounds table."""
import glob, json, os, re, statistics as st, sys
P = sys.argv[1]
rows = sys.argv[2:] or sorted({re.sub(r"^e2e-|-r\d+\.jsonl$", "", os.path.basename(f)) for f in glob.glob(f"{P}/e2e-*-r*.jsonl")})
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
        log = open(f"{P}/serve-e2e-{t}.log").read()
        m = re.search(r"static weights\s+([\d.]+) GiB", log); rec["static_gib"] = float(m.group(1)) if m else None
        m = re.search(r"I placement: '?token_embd[^\n]*", log); rec["emb_placement"] = m.group(0)[:140] if m else None
        pf = f"{P}/serve-prof-{t}.log"
        if os.path.exists(pf):
            for l in open(pf):
                if re.search(r"\bembed_(tq|lookup|exl3i)\b", l) and "#" in l:
                    x = l.split(); rec["emb_op"] = next(w for w in x if w.startswith("embed_")); rec["emb_calls"] = int(x[0])
                    rec["emb_total_ms"] = float(x[1]); rec["emb_us_call"] = float(x[2]); break
        out.append(rec); print(json.dumps(rec))
print()
keys = ["static_gib", "prefill_s_8k", "ttft_ms_8k", "step_ms_0", "step_ms_8k", "emb_us_call", "emb_calls"]
print(f'{"row":>10} ' + " ".join(f"{k:>13}" for k in keys))
for row in rows:
    R = [o for o in out if o["row"] == row]
    if not R: continue
    print(f"{row:>10} " + " ".join(f"{st.median([o[k] for o in R if o.get(k) is not None]) if any(o.get(k) is not None for o in R) else float('nan'):>13.4g}" for k in keys))
