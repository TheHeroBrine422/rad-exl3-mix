# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""feed.py CANDS.json EVALS RESULTS MAXOUT [--only-old] [--once] -- phase-1 feeder (RULES-picks am. 2026-11-16 §2).
Work unit = (candidate, slice) over s195c2-c5, whole slices, no early stop, the candidate's exact search mix.
Since 11-16 ~5:20 PM CST one coordinator eval per (candidate, slice): tag p1s<N>-<tag> (N = 2..5), so each slice's result
lands as soon as it is scored (partial reads visible, 6-minute units balance better). Earlier candidates were submitted
as one 4-slice eval tagged p1-<tag>; both forms count. At most MAXOUT evals outstanding (file MAXOUT next to CANDS
overrides, read each loop). --only-old: skip the live incumbent / pool."""
import json, os, sys, time

cp, evp, rp, mx = sys.argv[1:5]; mx = int(mx); old = "--only-old" in sys.argv; once = "--once" in sys.argv
SL = [f"s195c{i}" for i in (2, 3, 4, 5)]
G = "/data/q27/w180/slices/golden-{}"
mxf = os.path.join(os.path.dirname(os.path.abspath(cp)), "MAXOUT")


def base(tag):
    if tag.startswith("p1-"): return tag[3:]
    if tag.startswith("p1s") and tag[3:5].rstrip("-").isdigit(): return tag.split("-", 1)[1]
    return None


while True:
    C = json.load(open(cp))
    todo = [c for s in sorted(C, key=float) for c in C[s] if not (old and set(c["why"]) & {"incumbent", "pool"})]
    sub = {}                                    # eval tag -> (base tag, [slices])
    for l in open(evp):
        e = json.loads(l); b = base(e["tag"])
        if b: sub[e["tag"]] = (b, [s["name"] for s in e["slices"]])
    got = set()
    for l in open(rp):
        if '"p1' not in l: continue
        r = json.loads(l)
        if r.get("fail"): continue
        b = base(r["tag"])
        if b: got.add((r["tag"], r.get("slice")))
    covered = {(b, s) for t, (b, ss) in sub.items() for s in ss}
    out = sum(1 for t, (b, ss) in sub.items() if any((t, s) not in got for s in ss))
    if os.path.exists(mxf): mx = int(open(mxf).read().strip() or mx)
    new = [(c, s) for c in todo for s in SL if (c["tag"], s) not in covered]
    with open(evp, "a") as f:
        for c, s in new[:max(0, mx - out)]:
            t = f"p1s{s[-1] if s[-2] == 'c' else s[-2:]}-{c['tag']}"
            f.write(json.dumps({"tag": t, "mix": c["mix"], "slices": [{"name": s, "parts": {"m": G.format(s)}}]}) + "\n")
            print(time.strftime("%H:%M:%S"), "submit", t, flush=True); out += 1
    if once or (not new and not out): print("feed: nothing left" if not new else "once", flush=True); break
    time.sleep(60)
