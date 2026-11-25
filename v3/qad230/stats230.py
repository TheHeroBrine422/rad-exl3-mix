#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""stats230.py RENDER GEN -- w230 report numbers of one generation run (counts only): cuts, sessions / docs, generated tokens (mean / median / p90 / max),
stops, depth bands (t1) / settings / groups (t2), the prefill / decode split from radiance's per-request timings (prompt_n = tokens prefilled,
cache_n = prefix-cache hits; prompt_ms / predicted_ms overlap across concurrent requests) and the wall span of the generation files."""
import collections, json, os, sys

import numpy as np

render, gen = sys.argv[1:3]
cuts = {json.loads(l)["cut"]: json.loads(l) for l in open(f"{render}/cuts.jsonl")}
st = collections.Counter(); by = collections.defaultdict(lambda: collections.Counter()); glen = []; mt = []
for f in os.listdir(f"{gen}/gen"):
    if not f.endswith(".json"): continue
    d = json.load(open(f"{gen}/gen/{f}")); c = cuts[d["cut"]]; t = d["timings"]; n = len(d["gen"])
    st["cuts"] += 1; st["gen"] += n; st[f"stop_{d['stop_type']}"] += 1; st["prompt_tokens"] += d["prompt_len"]
    st["prompt_n"] += t.get("prompt_n", 0); st["cache_n"] += t.get("cache_n", 0); st["prompt_ms"] += t.get("prompt_ms", 0); st["pred_ms"] += t.get("predicted_ms", 0)
    st["draft_n"] += t.get("draft_n", 0); st["draft_acc"] += t.get("draft_n_accepted", 0); glen.append(n); mt.append(os.path.getmtime(f"{gen}/gen/{f}"))
    for k, v in (("setting", c["setting"]), ("kind", c["kind"]), ("band", c.get("band")), ("origin", c["origin"]), ("follower", c.get("follower"))):
        by[k][str(v)] += 1
    by["gen_by_setting"][c["setting"]] += n
glen = np.array(sorted(glen)); out = dict(st)
out["sessions_or_docs"] = len({c["sidx"] for c in cuts.values() if c["cut"] in {int(f[:-5]) for f in os.listdir(f"{gen}/gen") if f.endswith('.json')}})
out["gen_mean"] = float(glen.mean()); out["gen_median"] = int(np.median(glen)); out["gen_p90"] = int(np.percentile(glen, 90)); out["gen_max"] = int(glen.max())
out["prefill_share_of_request_time"] = st["prompt_ms"] / max(st["prompt_ms"] + st["pred_ms"], 1)
out["prefilled_per_generated"] = st["prompt_n"] / max(st["gen"], 1); out["cache_hit_share"] = st["cache_n"] / max(st["prompt_tokens"], 1)
out["draft_accept"] = st["draft_acc"] / max(st["draft_n"], 1); out["span_h"] = (max(mt) - min(mt)) / 3600
out.update({k: dict(sorted(v.items())) for k, v in by.items()})
print(json.dumps(out, indent=1))
