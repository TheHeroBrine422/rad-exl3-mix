#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""report232.py FINAL_GOLDEN CHAIN_LOG -- the numbers of the finished xg-fin golden: gen rows / tokens / docs per cell (share of gen rows), length histogram, totals, and the
chain's wall (first `chain232 start` to `chain232 done` line of CHAIN_LOG, box-local CT stamps `MM-DD H:MM AM/PM CST`; restarts add the gap)."""
import collections, datetime, json, re, sys
import numpy as np
g, lg = sys.argv[1:3]
m = json.load(open(f"{g}/kld.json")); meta = [json.loads(l) for l in open(f"{g}/corpus.meta.jsonl")]; gen = np.fromfile(f"{g}/gen.u8", np.uint8)
cell = collections.defaultdict(lambda: [0, 0, 0]); hist = collections.OrderedDict()
edges = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 131073]
for d, x in zip(m["docs"], meta):
    r = int(gen[d["row0"]:d["row0"] + d["tokens"] - 1].sum()); c = cell[x["cell"]]; c[0] += 1; c[1] += d["tokens"]; c[2] += r
    k = [f"{lo}-{hi - 1}" for lo, hi in zip(edges, edges[1:]) if lo <= d["tokens"] < hi][0]; h = hist.setdefault(k, [0, 0, 0]); h[0] += 1; h[1] += d["tokens"]; h[2] += r
G = sum(c[2] for c in cell.values()); T = sum(d["tokens"] for d in m["docs"])
print(f"docs {len(m['docs'])} tokens {T} rows {m['rows']} gen rows {G}")
for k, c in sorted(cell.items(), key=lambda kv: -kv[1][2]): print(f"  {k:18s} docs {c[0]:5d} tokens {c[1]:9d} gen rows {c[2]:8d} ({100 * c[2] / G:5.2f} %)")
print("length histogram (tokens a doc): docs / tokens / gen rows")
for k, h in hist.items(): print(f"  {k:14s} {h[0]:5d} {h[1]:9d} {h[2]:8d}")
ts = [(datetime.datetime.strptime("2026 " + mm.group(1), "%Y %m-%d %I:%M %p"), l) for l in open(lg) if (mm := re.match(r"(\d\d-\d\d \d+:\d\d [AP]M) CST", l))]
s = [t for t, l in ts if "chain232 start" in l]; e = [t for t, l in ts if "chain232 done" in l]
if s and e: print(f"wall {(e[-1] - s[0]).total_seconds() / 3600:.2f} h (first start {s[0]:%m-%d %H:%M} -> done {e[-1]:%m-%d %H:%M}; {len(s)} start(s))")
