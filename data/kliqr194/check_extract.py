#!/usr/bin/env python3
# check_extract.py INCUMBENTS.jsonl SIZEKL183.txt -- w177 (#194): the full-precision extraction must reproduce sizekl183.py's
# printed rows (bpw to 3 decimals, KL to 5, the tag). Prints MATCH / DIFF per size and one summary line.
import json, sys
inc = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pr = [l.split() for l in open(sys.argv[2]) if l.strip()]
ok = len(inc) == len(pr)
for r, p in zip(inc, pr):
    m = f"{r['bpw']:.3f}" == p[0] and f"{r['kld_t183']:.5f}" == p[1] and r["tag"] == p[-1]
    ok = ok and m
    print(f"{r['bpw']:.3f} {r['kld_t183']:.5f} {r['tag']}  sizekl183: {p[0]} {p[1]} {p[-1]}  {'MATCH' if m else 'DIFF'}")
print("CHECK ALL MATCH" if ok else "CHECK MISMATCH", len(inc), len(pr))
