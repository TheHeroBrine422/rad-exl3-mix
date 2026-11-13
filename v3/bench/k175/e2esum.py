#!/usr/bin/env python3
"""e2esum.py RES -- #175 (w137) pf.sh e2e rounds: per row and round, 8k prefill (server log ttft -> prompt / ttft tok/s,
the web UI's number; client TTFT) and decode at ctx 0 (server ms per decode step), medians of the measured runs."""
import glob, json, os, statistics as st, sys
from collections import defaultdict
R = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(sys.argv[1], "e2e-*-r*.jsonl"))):
    row, rnd = os.path.basename(f)[4:-6].rsplit("-r", 1)
    for l in open(f):
        j = json.loads(l)
        if j["warmup"]:
            continue
        s = j.get("srv_log") or {}
        if j["ctx"] == "8k":
            R[row]["ttft"].append(s.get("ttft_s", float("nan")) * 1e3); R[row]["ptok"].append(s["prompt"] / s["ttft_s"])
            R[row]["cttft"].append(j["client_ttft_ms"]); R[row]["np"].append(j["prompt_tokens"])
            R[row]["r" + rnd].append(s["prompt"] / s["ttft_s"])
        else:
            R[row]["dec"].append(j["srv_ms_per_decode_step"])
print(f"{'row':10s} {'prompt':>7s} {'8k ttft ms':>11s} {'tok/s':>7s} {'(r1 / r2)':>17s} {'client ttft':>12s} {'decode ms/step':>15s}  n")
for row, d in R.items():
    m = lambda k: st.median(d[k])
    rr = " / ".join(f"{st.median(d[k]):.0f}" for k in sorted(d) if k.startswith("r") and k[1:].isdigit())
    print(f"{row:10s} {m('np'):7.0f} {m('ttft'):11.1f} {m('ptok'):7.0f} {rr:>17s} {m('cttft'):12.1f} {m('dec'):15.2f}  {len(d['ptok'])}")
