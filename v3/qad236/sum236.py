#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sum236.py PLAN.json [STEPS.tsv] -- w236: the numbers a finals plan is judged on (plan231 summary + per-rank deep concurrency from the queues).
Prints: steps / budget step, the wall-time proxy sum over steps of the slower rank's cost (plan231 cost units) up to the budget step, rank imbalance,
carried (max / p95 GiB a rank), forced documents, sources' exhaustion steps, trace starts (planned vs actual, rank), deep traces at once per rank (max),
whether every trace ends by the budget step, the source mix (rows) per reported phase, trace rows by depth band and phase, and the step windows with
the most depth (for the smoke's deepest window)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D                                      # noqa: E402

p = json.load(open(sys.argv[1])); S = p["summary"]; R = int(p["ranks"]); V = int(p["V"]); last = S["last"]
items = p["items"]; br = {int(k): v for k, v in p.get("branch", {}).items()}
steps = D.plan([(0, x[1], x[2], x[3]) for x in items], None, V, int(p["Q"]), queues=p["queues"], starts={k: v[1] for k, v in br.items() if v[1]})
deep_items = {k for k, x in enumerate(items) if [x[0], x[1], x[2]] in [t[:3] for t in S["trace_starts"]]}
dr = np.zeros((len(steps), R), int); depth = np.zeros(len(steps))
for k, st in enumerate(steps):
    for s, segs in enumerate(st):
        for (it, a, b) in segs:
            if it in deep_items: dr[k, s % R] += 1; depth[k] = max(depth[k], b)
out = {"steps": S["steps"], "last": last, "rows_to_last": S["rows_to_last"], "share_rows": S["share_rows"],
       "step_cost_per_rank": S["step_cost_per_rank"], "carried_gib_per_rank": S["carried_gib_per_rank"], "forced_over_bound": S["forced_over_bound"],
       "source_exhausted_at_step": S["source_exhausted_at_step"], "planned_starts": S["planned_starts"],
       "trace_starts_doc_tokens_stream_step": [[t[0], t[1], t[2], t[3], t[4]] for t in S["trace_starts"]],
       "trace_ends_by_last": S["trace_ends_by_last"], "max_deep_per_rank": int(dr[:last].max()), "max_deep_total": int(dr[:last].sum(1).max()),
       "rows_by_phase": S["rows_by_phase"], "phases_rows": p["spec"].get("phases_rows"),
       "trace_rows_by_depth_band_64k_and_phase": S["trace_rows_by_depth_band_64k_and_phase"], "unused_entries": S["unused_entries"]}
ph = S["rows_by_phase"]; npz = len(next(iter(ph.values())))
out["calib_share_by_phase"] = [round((ph.get("calib", [0] * npz)[j] + ph.get("calibdeep", [0] * npz)[j]) / max(1, sum(v[j] for v in ph.values())), 3) for j in range(npz)]
if len(sys.argv) > 2:
    t = np.loadtxt(sys.argv[2], skiprows=1, delimiter="\t")
    c = t[:last, 1:1 + R]; out["wall_proxy_sum_max_rank_cost_to_last"] = round(float(c.max(1).sum()))
    car = t[:last, 1 + R:1 + 2 * R]; out["carried_tok_rank_max"] = int(car.max())
    w = 20; dsum = np.convolve(depth[:last], np.ones(w), "valid"); k0 = int(dsum.argmax())
    out["deepest_20_step_window"] = {"from_step": k0, "mean_max_trace_depth": int(dsum[k0] / w), "cost_max_rank_mean": round(float(c[k0:k0 + w].max(1).mean()), 1)}
print(json.dumps(out, indent=1, default=str))
