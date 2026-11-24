#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""stats228.py RENDER GEN -- w228 report numbers of one generation run: cuts, sessions / documents, generated tokens, stops, the depth
histogram of the cuts (prefix tokens, 16k bins), kinds, and the prefill / decode split from radiance's per-request timings (prompt_n =
tokens prefilled, cache_n = prefix-cache hits, prompt_ms / predicted_ms = each request's time in prefill / decode, overlapping across the
concurrent requests). Counts only: prints no text."""
import collections, json, os, sys

render, gen = sys.argv[1:3]
cuts = {json.loads(l)["cut"]: json.loads(l) for l in open(f"{render}/cuts.jsonl")}
st = collections.Counter(); depth = collections.Counter(); kinds = collections.Counter(); sess = set(); glen = []
for f in os.listdir(f"{gen}/gen"):
    if not f.endswith(".json"): continue
    d = json.load(open(f"{gen}/gen/{f}")); c = cuts[d["cut"]]; t = d["timings"]
    st["cuts"] += 1; st["gen"] += len(d["gen"]); st[f"stop_{d['stop_type']}"] += 1; st["follow"] += bool(c.get("follow"))
    st["prompt_n"] += t.get("prompt_n", 0); st["cache_n"] += t.get("cache_n", 0); st["prompt_tokens"] += d["prompt_len"]
    st["prompt_ms"] += t.get("prompt_ms", 0); st["pred_ms"] += t.get("predicted_ms", 0); st["wall_s"] += d["wall"]
    st["draft_n"] += t.get("draft_n", 0); st["draft_acc"] += t.get("draft_n_accepted", 0)
    depth[min(c["p"] // 16384, 7) * 16] += 1; kinds[c["kind"]] += 1; sess.add(c["sidx"]); glen.append(len(d["gen"]))
glen.sort()
out = dict(st); out["sessions_or_docs"] = len(sess)
out["gen_mean"] = st["gen"] / max(st["cuts"], 1); out["gen_median"] = glen[len(glen) // 2] if glen else 0
out["prefill_share_of_request_time"] = st["prompt_ms"] / max(st["prompt_ms"] + st["pred_ms"], 1)
out["prefilled_per_generated"] = st["prompt_n"] / max(st["gen"], 1); out["cache_hit_share"] = st["cache_n"] / max(st["prompt_tokens"], 1)
out["draft_accept"] = st["draft_acc"] / max(st["draft_n"], 1)
out["depth_hist_k"] = {f"{k}-{k + 16}k": v for k, v in sorted(depth.items())}; out["kinds"] = dict(kinds.most_common())
print(json.dumps(out, indent=1))
