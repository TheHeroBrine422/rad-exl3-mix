#!/usr/bin/env python3
"""tracegold.py CALIBMETA POOLS EXCLUDE_GOLDEN -- d1108 amendment 4: document args for mkgold.py (idx:cap:from) of a
golden of calib's own agentic text in the SEARCH pool: the 2 long 27B-generated traces (cobra, ripgrep), scored in
2,048-token windows at offsets 0, 2k, ..., 30k (each with its full prefix, so contexts up to 32k), plus every
search-pool document of the agentic sources (swe-*, nemotron-swe, hermes-fc, glaive-fc-v2, when2call) not already in
EXCLUDE_GOLDEN, capped at 4,096."""
import json, sys
m = json.load(open(f"{sys.argv[1]}/kld.json")); p = json.load(open(sys.argv[2]))["pool"]
used = {d["orig"] for d in json.load(open(f"{sys.argv[3]}/kld.json"))["docs"]}
AG = ("swe-", "nemotron-swe", "hermes-fc", "glaive-fc", "when2call")
out = []
for i, d in enumerate(m["docs"]):
    if p[i] != "search": continue
    if d["source"].startswith("agent-trace"):
        out += [f"{i}:{a + 2048}:{a}" for a in range(0, 32768, 2048) if a + 2048 <= d["tokens"]]
    elif d["source"].startswith(AG) and i not in used:
        out.append(f"{i}:4096")
print(" ".join(out))
