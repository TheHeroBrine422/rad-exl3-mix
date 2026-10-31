#!/usr/bin/env python3
"""skipdone.py RESULTS.jsonl... < jobs -- pass through the sweep job lines whose results are not all in RESULTS yet"""
import json, sys
seen = set()
for f in sys.argv[1:]:
    try:
        for l in open(f):
            r = json.loads(l); seen.add((r["layer"], r["w"], r["method"], r["bits"]))
    except FileNotFoundError:
        pass
for line in sys.stdin:
    a = line.split()
    if a[0] == "trellis.py": need = [(int(a[1]), a[2], "trellis_" + a[3], int(a[4]))]
    else: need = [(int(a[1]), a[2], a[3], int(b)) for b in a[4].split(",")]
    if not all(n in seen for n in need): print(line, end="")
