#!/usr/bin/env python3
"""e2sum.py RESDIR -- e2.sh's rounds: per pack and row, 8k prefill (server ttft -> prompt / ttft tok/s), decode ms / step
at ctx 8k and ctx 0 (server ms per decode step), medians of the measured runs over every round, plus each round's ctx-0
decode median. (w152, #179; from w137's e2sum.py)"""
import glob, json, os, statistics as st, sys
from collections import defaultdict

R = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*-r*.jsonl"))):
    name, rnd = os.path.basename(f)[:-6].rsplit("-r", 1)
    pack, row = name.rsplit("-", 1)
    for l in open(f):
        j = json.loads(l)
        if j["warmup"]:
            continue
        s = j.get("srv_log") or {}
        d = R[(pack, row)]
        if j["ctx"] == "8k":
            d["ttft"].append(s["ttft_s"] * 1e3); d["ptok"].append(s["prompt"] / s["ttft_s"])
            d["dec8"].append(j["srv_ms_per_decode_step"])
        else:
            d["dec"].append(j["srv_ms_per_decode_step"]); d["d" + rnd].append(j["srv_ms_per_decode_step"])
print(f"{'pack':34s} {'row':8s} {'8k ttft ms':>11s} {'tok/s':>7s} {'dec@8k':>7s} {'dec@0':>7s}  {'per round @0':>24s}  n")
for (pack, row), d in sorted(R.items()):
    m = lambda k: st.median(d[k]) if d[k] else float("nan")
    rr = " / ".join(f"{st.median(d[k]):.2f}" for k in sorted(d) if k.startswith("d") and k[1:].isdigit())
    print(f"{pack:34s} {row:8s} {m('ttft'):11.1f} {m('ptok'):7.0f} {m('dec8'):7.2f} {m('dec'):7.2f}  {rr:>24s}  {len(d['dec'])}")
