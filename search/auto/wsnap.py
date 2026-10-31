#!/usr/bin/env python3
"""wsnap.py STEP -- ver1 queue lines (stdout) for the best-so-far fit-slice pick per target as of eval step STEP (results order), slice-normalised; the held-out snapshot of the search at that point. Report on stderr."""
import json, re, os, sys
STEP = int(sys.argv[1]); TG = {"2.00","2.50","3.00","3.50","4.00","4.50","5.00","6.00","7.00"}
off = {"a": 1.0, "f2": 0.848, "f3": 0.958}
best = {}; done = set()
for step, l in enumerate(open("/root/evq/results.jsonl")):
    r = json.loads(l); done.add(r["tag"])
    if step >= STEP or "kld" not in r: continue
    m = re.fullmatch(r"(a|f2|f3)-(knap-.+-sc4\.00-e4-([\d.]+))", r["tag"])
    if not m or m.group(3) not in TG: continue
    k = r["kld"] / off[m.group(1)]
    if k < best.get(m.group(3), (9,))[0]: best[m.group(3)] = (k, m.group(2))
for t in sorted(best, key=float):
    k, n = best[t]; w = f"w-{n}"; p = f"/root/knap/mixes/{n}.mix"
    st = "measured" if w in done else ("MISSING MIX" if not os.path.exists(p) else "queue")
    print(f"# {t} {n} fit-norm {k:.5f} {st}", file=sys.stderr)
    if st == "queue": print(f"{w} /root/models/final/q38-27b-sc4.00-e4.rad {p} golden-ver1 corpus/ver1.jsonl")
