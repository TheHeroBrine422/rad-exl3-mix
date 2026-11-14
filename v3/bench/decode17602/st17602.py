#!/usr/bin/env python3
"""st17602.py STAMPS.csv -- #176-02 (w159): RADIANCE_AQL_STAMP's CSV (pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,
play,records) reduced to what the price fit needs, as JSON on stdout. Decode passes = the tape length (records) with the
most passes; only complete passes (every dispatch timed, the kind's full length) count. Per pass: wall (first start to
last end) and busy (union of dispatch intervals); per tape record (seq): kernel, queue, median duration and median start
offset from the pass start (us). Standard library only (python3 -I). (w159)"""
import csv, json, statistics as st, sys
from collections import defaultdict

P = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1], newline="")):
    P[int(r["pass"])].append(r)
kinds = defaultdict(list)
for p, rows in P.items():
    kinds[int(rows[0]["records"])].append(rows)
rec = max(kinds, key=lambda q: len(kinds[q]))
full = max(len(r) for r in kinds[rec])
passes = [r for r in kinds[rec] if len(r) == full and all(x["start_ns"] and x["end_ns"] for x in r)]


def union(iv):
    tot, ca, cb = 0, None, None
    for x, y in sorted(iv):
        if cb is None or x > cb:
            if cb is not None: tot += cb - ca
            ca, cb = x, y
        elif y > cb: cb = y
    return tot + (cb - ca if cb is not None else 0)


walls, busys = [], []
dur, off, kern, que = defaultdict(list), defaultdict(list), {}, {}
for rows in passes:
    iv = [(int(r["start_ns"]), int(r["end_ns"])) for r in rows]
    t0 = min(x for x, _ in iv)
    walls.append(max(y for _, y in iv) - t0); busys.append(union(iv))
    for r, (x, y) in zip(rows, iv):
        s = int(r["seq"]); dur[s].append(y - x); off[s].append(x - t0); kern[s] = r["kernel"]; que[s] = r["queue"]
out = {"records": rec, "passes": len(passes), "passes_all_kinds": {str(k): len(v) for k, v in kinds.items()},
       "wall_us": st.median(walls) / 1e3 if walls else None, "busy_us": st.median(busys) / 1e3 if busys else None,
       "seqs": [{"seq": s, "kernel": kern[s], "queue": que[s], "us": st.median(dur[s]) / 1e3,
                 "start_us": st.median(off[s]) / 1e3} for s in sorted(dur)]}
json.dump(out, sys.stdout)
