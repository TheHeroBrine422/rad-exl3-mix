#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""smk236.py RUN_DIR [FROM TO] -- w236: a smoke window's numbers from qad220's run.log (rank 0's per-step records, "what": "main", steps FROM < step <= TO)
and smoke236's mem.log: s / step (trainer wall; elapsed between log lines), rank imbalance (rank_wall max / mean, mean and worst step), rows a second,
rank 0's cards (reserved peak, free min, ooms, alloc retries, carried-state peak), host RSS a rank (max), park-pool peak / misses, evictions, prefix rebuild
tokens, MemAvailable min (GB) and VRAM used max per GPU (GiB, rocm order 0-3) over the run."""
import json, re, sys
import numpy as np

d = sys.argv[1]; lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 10 ** 9)
ts, R = [], []
for ln in open(f"{d}/run.log"):
    m = re.match(r"(\d\d):(\d\d):(\d\d) (\{.*\"what\": \"main\".*)", ln)
    if m:
        r = json.loads(m[4])
        if lo < r["step"] <= hi: ts.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])); R.append(r)
out = {"run": d, "steps": [R[0]["step"], R[-1]["step"]] if R else None, "n": len(R)}
if R:
    el = [(b - a) % 86400 for a, b in zip(ts, ts[1:])]
    w = np.array([r["wall"] for r in R]); rw = np.array([r["rank_wall"] for r in R if r.get("rank_wall")])
    rows = [r["rows"] for r in R]
    out.update({"wall_mean": round(float(w.mean()), 2), "wall_median": round(float(np.median(w)), 2), "wall_max": round(float(w.max()), 2),
                "wall_mean_after2": round(float(w[2:].mean()), 2) if len(w) > 2 else None, "elapsed_mean_after2": round(float(np.mean(el[2:])), 2) if len(el) > 2 else None,
                "elapsed_mean": round(float(np.mean(el)), 2) if el else None,
                "rows_per_s": round((rows[-1] - rows[0]) / max(1, sum(el)), 1) if el else None,
                "rank_wall_mean": [round(float(x), 1) for x in rw.mean(0)] if len(rw) else None,
                "rank_imbalance_mean": round(float((rw.max(1) / rw.mean(1)).mean()), 3) if len(rw) else None,
                "rank_imbalance_max": round(float((rw.max(1) / rw.mean(1)).max()), 3) if len(rw) else None})
    mem = {}
    for r in R:
        for c, v in (r.get("mem") or {}).items():
            z = mem.setdefault(c, {"res_peak": 0, "free_min": 99, "ooms": 0, "retries": 0})
            z["res_peak"] = max(z["res_peak"], v["res_peak"]); z["free_min"] = min(z["free_min"], v["free"]); z["ooms"] = max(z["ooms"], v["ooms"]); z["retries"] = max(z["retries"], v["retries"])
        for c, v in (r.get("state_peak_gib") or {}).items(): mem.setdefault(c, {})["state_peak"] = max(mem[c].get("state_peak", 0), v)
    out["rank0_cards"] = mem
    out["rank_rss_gib_max"] = [round(float(x), 2) for x in np.max([r["rank_rss_gib"] for r in R if r.get("rank_rss_gib")], 0)] if any(r.get("rank_rss_gib") for r in R) else None
    hs = [h for r in R for h in (r.get("host_stages") or [])]
    if hs: out["pool_peak_gib_max"] = max(h.get("pool_peak", 0) for h in hs); out["pool_misses_max"] = max(h.get("pool_misses", 0) for h in hs); out["parked_gib_max"] = max(h.get("parked", 0) for h in hs)
    out["evicted_sum"] = int(sum(r.get("evicted", 0) for r in R)); out["prefix_tok_sum"] = int(sum(r.get("prefix_tok", 0) for r in R))
    out["train_kl_first_last"] = [round(R[0]["train_kl"], 5), round(R[-1]["train_kl"], 5)]
try:
    M = [l.split() for l in open(f"{d}/mem.log") if l.strip()]
    out["memavail_gb_min"] = min(int(x[1]) for x in M)
    if len(M[0]) > 2: out["vram_gib_max_gpu0_3"] = [max(float(x[k]) for x in M if len(x) > k) for k in range(2, len(M[0]))]
except FileNotFoundError: pass
print(json.dumps(out, indent=1))
