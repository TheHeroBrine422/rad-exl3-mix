#!/usr/bin/env python3
"""calibshare119.py KLD.json GEN.u8 OUT.json -- calib gen-token share per stratum (RULES.md weights: source, with
agent-trace/* -> agent-trace, pg19-long -> pg19, arxiv-long -> arxiv), from the whole-calib golden."""
import json, sys, collections, numpy as np

def stratum(s):
    if s.startswith("agent-trace/"): return "agent-trace"
    return {"pg19-long": "pg19", "arxiv-long": "arxiv"}.get(s, s)

if __name__ == "__main__":
    m = json.load(open(sys.argv[1])); gen = np.fromfile(sys.argv[2], np.uint8)
    c = collections.Counter()
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; c[stratum(d["source"])] += int(gen[d["row0"]:d["row0"] + k].sum())
    T = sum(c.values()); json.dump({k: v / T for k, v in c.most_common()}, open(sys.argv[3], "w"), indent=0)
    print(len(c), "strata", T, "gen tokens")
