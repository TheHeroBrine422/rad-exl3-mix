#!/usr/bin/env python3
"""speedtable197.py SPEED_DIR OUT.md ROW... -- #197 (w185): decode and prefill per serve row from bench.py's JSONL (w177's
bench, one line per request). Per context (0 / 4k / 16k), the median over the non-warm-up runs of the server's ms per
decode step (srv_ms_per_decode_step) and of the client's TTFT; prefill tok/s = prompt tokens / TTFT. Plain python3."""
import json, os, statistics, sys
SD, OUT = sys.argv[1:3]; rows = sys.argv[3:]
L = ["| row | decode ms/token ctx 0 | ctx 4k | ctx 16k | TTFT 4k (prefill tok/s) | TTFT 16k (prefill tok/s) | runs |", "|---|---|---|---|---|---|---|"]
for r in rows:
    f = os.path.join(SD, f"{r}.jsonl")
    if not os.path.exists(f):
        L.append(f"| {r} | — | — | — | — | — | not run |"); continue
    recs = [json.loads(l) for l in open(f) if l.strip()]
    by = {}
    for x in recs:
        if x.get("warmup"): continue
        by.setdefault(str(x["ctx"]), []).append(x)
    def med(ctx, key):
        v = [x[key] for x in by.get(ctx, []) if x.get(key) is not None]
        return statistics.median(v) if v else None
    dec = {c: med(c, "srv_ms_per_decode_step") for c in ("0", "4k", "16k")}
    def pre(c):
        t = med(c, "client_ttft_ms"); p = med(c, "prompt_tokens")
        return None if t is None else f"{t / 1000:.2f} s ({p / (t / 1000):.0f} tok/s)"
    fmt = lambda v: "—" if v is None else f"{v:.2f}"
    n = {c: len(by.get(c, [])) for c in ("0", "4k", "16k")}
    L.append(f"| {r} | {fmt(dec['0'])} | {fmt(dec['4k'])} | {fmt(dec['16k'])} | {pre('4k') or '—'} | {pre('16k') or '—'} | {n['0']}/{n['4k']}/{n['16k']} |")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
