# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""feed2.py CANDS.json EVALS RESULTS MAXOUT [--only-old] [--once] -- phase-2 feeder (RULES-picks am. 2026-11-16 5:15 PM §2): slices golden-final750 (calib verify c1 / c3 + final-x) and s195c6-c10, one eval per (candidate, slice), tags p2<N>-<tag>.
Work unit = (candidate, slice) over s195c2-c5, whole slices, no early stop, the candidate's exact search mix.
(Copied from feed.py; phase-1 docs below are its history.)
lands as soon as it is scored (partial reads visible, 6-minute units balance better). Earlier candidates were submitted
as one 4-slice eval tagged p1-<tag>; both forms count. At most MAXOUT evals outstanding (file MAXOUT next to CANDS
overrides, read each loop). --only-old: skip the live incumbent / pool."""
import json, os, sys, time

cp, evp, rp, mx = sys.argv[1:5]; mx = int(mx); old = "--only-old" in sys.argv; once = "--once" in sys.argv
SL = [f"s195c{i}" for i in (6, 7, 8, 9, 10)] + ["final750"]   # final750 (ver4) last: overfit audit first (AGENTS.md)
GP = {"final750": "/data/q27/final/golden-final750", **{f"s195c{i}": f"/data/q27/w180/slices/golden-s195c{i}" for i in (6, 7, 8, 9, 10)}}
SHORT = {"final750": "f", **{f"s195c{i}": str(i) for i in (6, 7, 8, 9, 10)}}
mxf = os.path.join(os.path.dirname(os.path.abspath(cp)), "MAXOUT")


def base(tag):
    if tag.startswith("p2") and "-" in tag and tag[2:tag.index("-")] in SHORT.values(): return tag.split("-", 1)[1]
    return None


while True:
    C = json.load(open(cp))
    todo = [c for s in sorted((k for k in C if not k.startswith("_")), key=float) for c in C[s] if not (old and set(c["why"]) & {"incumbent", "pool"})]
    sub = {}                                    # eval tag -> (base tag, [slices])
    for l in open(evp):
        e = json.loads(l); b = base(e["tag"])
        if b: sub[e["tag"]] = (b, [s["name"] for s in e["slices"]])
    got = set()
    for l in open(rp):
        if '"p2' not in l: continue
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
            t = f"p2{SHORT[s]}-{c['tag']}"
            f.write(json.dumps({"tag": t, "mix": c["mix"], "slices": [{"name": s, "parts": {"m": GP[s]}}]}) + "\n")
            print(time.strftime("%H:%M:%S"), "submit", t, flush=True); out += 1
    if once or (not new and not out): print("feed: nothing left" if not new else "once", flush=True); break
    time.sleep(60)
