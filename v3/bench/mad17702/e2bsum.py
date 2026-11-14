#!/usr/bin/env python3
"""e2bsum.py RESDIR -- e2b.sh's rounds: per row, 8k prefill (server ttft -> prompt / ttft tok/s) and decode at ctx 0
(server ms per decode step), medians over the measured runs of every round, and each round's decode median. (w150;
w137's e2sum.py with TAG-rROUND names)"""
import glob, json, os, statistics as st, sys
from collections import defaultdict

R = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*-r*.jsonl"))):
    tag, rnd = os.path.basename(f)[:-6].rsplit("-r", 1)
    for l in open(f):
        j = json.loads(l)
        if j["warmup"]:
            continue
        s = j.get("srv_log") or {}
        d = R[tag]
        if j["ctx"] == "8k":
            d["ttft"].append(s["ttft_s"] * 1e3); d["ptok"].append(s["prompt"] / s["ttft_s"])
        else:
            d["dec"].append(j["srv_ms_per_decode_step"]); d["d" + rnd].append(j["srv_ms_per_decode_step"])
print(f"{'row':34s} {'8k ttft ms':>11s} {'tok/s':>7s} {'decode ms/step':>15s}  {'per round':>24s}  n")
for tag, d in sorted(R.items()):
    m = lambda k: st.median(d[k]) if d[k] else float("nan")
    rr = " / ".join(f"{st.median(d[k]):.2f}" for k in sorted(d) if k.startswith("d") and k[1:].isdigit())
    print(f"{tag:34s} {m('ttft'):11.1f} {m('ptok'):7.0f} {m('dec'):15.2f}  {rr:>24s}  {len(d['dec'])}")
