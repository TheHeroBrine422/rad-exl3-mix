#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkspec236.py QROOT INVDIR OUT_SPEC [--margin M] [--tail-rows T] [--budget B] -- w236: the V3.1 finals' plan231 SPEC (one plan for every size).

Sources (inv236.py's id lists in INVDIR; dict order = deep-trace start order: the 2 xgap traces first, the 6 calib traces after):
  xgdeep    xg-fin's 2 longest agent sessions (deep)              calibdeep  calib's 6 released whole traces (deep)
  calib     trainable calib (qad / fit / search + released verify; INVDIR/calib-exclude.json out), whole docs (cap 131072 > the longest)
  xgap      the rest of xg-fin (cap 131072)                       tset1  teacher set 1, branch-trained (INVDIR/tset1-exclude.json out)
  tset2     teacher set 2 (whole cuts)
Bounds (w236 variants, report): carried <= 500k tokens a rank (= 2 x 250k traces, what w233 sized the 12 GB park pool for), at most 8 documents > 32k
a rank at once (4 deferred the teacher sessions to a pile-up at the end), trace_hold, the 8 trace starts evenly over steps 0 .. --last-start (880).
Budget: every source is used whole (single epoch), so the budget is the gen rows the sources hold / (1 + margin), rounded down to 0.1M (or --budget).
Shares: a calib-heavy end = share window over the last --tail-rows assigned rows (calib 0.8 / xgap 0.1 / tset2 0.1, tset1 0); before it each source's
share = its rows left after the end window's expected take, so every source runs out near the end together (the deep traces sit outside the shares).
phases_rows (report): the decay start (0.8 x budget) and the end window's start."""
import json, os, sys

q, inv, outp = sys.argv[1:4]; a = sys.argv[4:]
opt = lambda k, d: float(a[a.index(k) + 1]) if k in a else d
margin, tail = opt("--margin", 0.01), opt("--tail-rows", 2.5e6)
last_start = opt("--last-start", 880)                     # the 8 deep traces start evenly over steps 0 .. last_start (the plan's budget step is ~1180)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad231"))
import plan231 as PL                                     # noqa: E402

ext = f"{q}/ext"; calib = "/data/q27/challenge/golden/calib"; ld = lambda f: json.load(open(f"{inv}/{f}"))
xg_all = list(range(len(json.load(open(f"{ext}/gold/xg-fin/kld.json"))["docs"])))
t1_all = list(range(len(json.load(open(f"{ext}/gold/tset1/kld.json"))["docs"])))
cex, t1ex, xdeep = set(ld("calib-exclude.json")), set(ld("tset1-exclude.json")), ld("xg-deep.json")
ids = {"calib": [i for i in ld("calib-train.json") if i not in cex], "xgap": [i for i in xg_all if i not in set(xdeep)],
       "tset1": [i for i in t1_all if i not in t1ex]}
for k, v in ids.items(): json.dump(v, open(f"{inv}/ids-{k}.json", "w"))
G = lambda nm: {"golden": nm, "dir": f"{ext}/gold/{nm}", "genmask": f"{ext}/genmask/{nm}.u8"}
src = {"xgdeep": {**G("xg-fin"), "deep": True, "ids": f"{inv}/xg-deep.json"},
       "calibdeep": {"golden": "calib", "dir": calib, "deep": True, "ids": f"{inv}/calib-deep.json"},
       "calib": {"golden": "calib", "dir": calib, "ids": f"{inv}/ids-calib.json", "cap": 131072},
       "xgap": {**G("xg-fin"), "ids": f"{inv}/ids-xgap.json", "cap": 131072},
       "tset1": {**G("tset1"), "ids": f"{inv}/ids-tset1.json", "branch": True, "session_key": "sidx"},
       "tset2": {**G("tset2")}}
starts = [int(round(last_start * j / 7)) for j in range(8)]
for k in ("xgdeep", "calibdeep"): src[k]["starts"] = starts   # plan231 reads one list for all traces (in source order): the same list on both
spec = {"V": 24, "Q": 1024, "ranks": 2, "seed": 0, "margin": margin, "max_carried_tokens": 500000, "long_tokens": 32768, "max_long": 8, "trace_hold": True,
        "state_budget_gb": 9.5, "deep_first": 0,
        "exclude_calib": [f"{inv}/verify_v31.json", f"{inv}/calib-exclude.json"],
        "fixed_eval": {"pools": "/data/q27/w214/pools214.json", "pool": "train214", "seed": 2181, "rows": 20000, "len": 2048},
        "sources": src}
excl = set(json.load(open(f"{inv}/verify_v31.json"))) | cex
calg = PL.D.Gold(calib, "calib"); excl |= set(PL.fixed_eval_docs(spec["fixed_eval"], calg))
rows = {nm: sum(nr for _, _, nr in PL.Src(nm, sp, excl).items) for nm, sp in src.items()}
have = sum(rows.values()); deep = rows["xgdeep"] + rows["calibdeep"]; mix = have - deep
B = opt("--budget", int(have / (1 + margin) / 1e5) * 1e5)
tw = {"calib": 0.8, "xgap": 0.1, "tset2": 0.1}
mt = tail * mix / have                                   # the end window's expected mixed rows (deep rows aside)
left = {nm: max(0.0, rows[nm] - tw.get(nm, 0.0) * mt) for nm in ("calib", "xgap", "tset1", "tset2")}
for nm, v in left.items(): src[nm]["share"] = round(v / sum(left.values()), 4)
spec["budget_rows"] = B
spec["share_windows"] = [[B * (1 + margin) - tail, tw]]
spec["phases_rows"] = sorted([0.8 * B, B * (1 + margin) - tail])
json.dump(spec, open(outp + ".part", "w"), indent=1); os.replace(outp + ".part", outp)
print(json.dumps({"rows": rows, "have": have, "deep": deep, "budget_rows": B, "shares_before_end": {k: src[k]["share"] for k in left},
                  "end_window_from": spec["share_windows"][0][0], "end_window_shares": tw}, indent=1))
