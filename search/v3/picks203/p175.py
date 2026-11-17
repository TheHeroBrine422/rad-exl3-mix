# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""p175.py CANDS.json DRV EVALS [--late-only] -- RULES-picks am. 11-17 1:50 AM (+ addendum): adds to the phase-1 candidate set
(scored on s195c2-c5 by feed.py) (a) the old search's last tentatives at sizes stopped for the round (1.5 block-04933, 1.75
block-04937 / -04948, 5.5 cross-04934) and, unless --late-only, (b) every round model at 4.5 / 6 / 6.5 / 7 since the s195s23
rotation: accepts / joins, the pool at the end, near misses (stage-1 z <= -1 on s195s23 and stage-2 z <= +0.5 on s195s45),
inbox models. Dedup by assignment hash; each keeps its exact eval mix."""
import hashlib, json, os, sys
cp, drv, evp = sys.argv[1:4]; late_only = "--late-only" in sys.argv
st = [json.loads(l) for l in open(f"{drv}/state.jsonl")]
mix = {}
for l in open(evp):
    e = json.loads(l); mix[e["tag"]] = e["mix"]
C = json.load(open(cp)); H = lambda a: hashlib.sha1(json.dumps(a, sort_keys=True).encode()).hexdigest()[:16]
have = {c["hash"]: c for v in C.values() for c in v}
add = [(1.5, "v3s183-1.5-block-04933", "late-confirmed"), (1.75, "v3s183-1.75-block-04937", "late-tentative"),
       (1.75, "v3s183-1.75-block-04948", "late-tentative"), (5.5, "v3s183-5.5-cross-04934", "late-tentative")]
if not late_only:
    i = max(k for k, e in enumerate(st) if e.get("ev") == "rotate" and e.get("to") == "s195s23")
    S = (4.5, 6.0, 6.5, 7.0); sub = {e["tag"]: e for e in st if e.get("ev") == "submit"}; s1 = {}
    for e in st[i:]:
        if e.get("size") not in S: continue
        if e.get("ev") == "tentative": s1[e["tag"]] = e
        if e.get("ev") == "decision" and e.get("stage") == 2:
            if e["decision"] in ("accept", "join"): add.append((e["size"], e["tag"], "round-confirmed"))
            elif e.get("z", 9) <= 0.5 and (e.get("z1") if e.get("z1") is not None else s1.get(e["tag"], {}).get("z", 0)) <= -1:
                add.append((e["size"], e["tag"], "round-near-miss"))
        if e.get("ev") == "pool": add += [(e["size"], t, "round-pool") for t in (e.get("members") or [])]
        if e.get("ev") == "incumbent": add.append((e["size"], e["tag"], "round-incumbent"))
    for t, e in sub.items():
        if e.get("size") in S and e.get("kind") == "manual" and t in mix and st.index(e) >= i: add.append((e["size"], t, "round-inbox"))
n = 0
for size, t, why in add:
    p = f"{drv}/assign/{t}.json"
    if not os.path.exists(p) or t not in mix: print("missing", t); continue
    h = H(json.load(open(p)))
    if h in have:
        if why not in have[h]["why"]: have[h]["why"].append(why)
        continue
    c = {"tag": t, "hash": h, "why": [why], "dups": [], "mix": mix[t]}
    C.setdefault(str(float(size)), []).append(c); have[h] = c; n += 1
json.dump(C, open(cp + ".part", "w")); os.replace(cp + ".part", cp)
print("added", n)
