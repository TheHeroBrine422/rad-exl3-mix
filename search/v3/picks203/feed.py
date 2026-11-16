# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""feed.py CANDS.json EVALS RESULTS MAXOUT [--only-old] [--once] -- phase-1 feeder (RULES-picks am. 2026-11-16 §2).
Appends one coordinator eval per candidate (tag p1-<tag>, its exact mix, slices s195c2-c5 whole, no early stop) while
at most MAXOUT phase-1 evals are outstanding (submitted, not all four slices in RESULTS). MAXOUT may be changed live
in the file MAXOUT next to CANDS (read each loop). --only-old: skip candidates that are the live incumbent / pool."""
import json, os, sys, time

cp, evp, rp, mx = sys.argv[1:5]; mx = int(mx); old = "--only-old" in sys.argv; once = "--once" in sys.argv
SL = [f"s195c{i}" for i in (2, 3, 4, 5)]
G = "/data/q27/w180/slices/golden-{}"
mxf = os.path.join(os.path.dirname(os.path.abspath(cp)), "MAXOUT")
while True:
    C = json.load(open(cp))
    todo = [c for s in sorted(C, key=float) for c in C[s] if not (old and set(c["why"]) & {"incumbent", "pool"})]
    sub = set()
    for l in open(evp):
        t = json.loads(l)["tag"]
        if t.startswith("p1-"): sub.add(t)
    done = {}
    for l in open(rp):
        if '"p1-' not in l: continue
        r = json.loads(l); done.setdefault(r["tag"], set()).add(r.get("slice"))
    fin = {t for t, s in done.items() if set(SL) <= s}
    out = len(sub - fin)
    if os.path.exists(mxf): mx = int(open(mxf).read().strip() or mx)
    new = [c for c in todo if "p1-" + c["tag"] not in sub]
    with open(evp, "a") as f:
        for c in new[:max(0, mx - out)]:
            f.write(json.dumps({"tag": "p1-" + c["tag"], "mix": c["mix"], "slices": [{"name": s, "parts": {"m": G.format(s)}} for s in SL]}) + "\n")
            print(time.strftime("%H:%M:%S"), "submit p1-" + c["tag"], flush=True); out += 1
    if once or (not new and not out): print("feed: nothing left" if not new else "once", flush=True); break
    time.sleep(60)
