# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""add_extra.py CANDS.json DRIVER.json EXTRA_BC.json SOLVES.json... -- RULES-picks am. 11-16 6:15 PM: adds (a) extra_solve.py
outputs (tag x203-<size>-<var>, assignment saved next to CANDS under assign/) and (b)/(c) existing search tags (their exact
eval mix from evals.jsonl) to the phase-1 candidate set, dedup by assignment hash (sha1 as cands.py)."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import menu as MN
cp, dj, bc = sys.argv[1:4]; sols = sys.argv[4:]
D = json.load(open(dj)); R = os.path.dirname(os.path.abspath(dj)); C = json.load(open(cp))
M = MN.Menu.load(D["menu"] if os.path.isabs(D["menu"]) else os.path.join(R, D["menu"]))
ev = {}
for l in open(D["evals"]):
    e = json.loads(l); ev[e["tag"]] = e["mix"]
H = lambda a: hashlib.sha1(json.dumps(a, sort_keys=True).encode()).hexdigest()[:16]
have = {c["hash"]: c for v in C.values() for c in v}
AD = os.path.join(os.path.dirname(os.path.abspath(cp)), "assign"); os.makedirs(AD, exist_ok=True)
n = 0
def put(size, tag, a, mix, why):
    global n
    h = H(a)
    if h in have: have[h]["why"].append(why); have[h]["dups"].append(tag); return
    c = {"tag": tag, "hash": h, "why": [why], "dups": [], "mix": mix}
    C.setdefault(str(float(size)), []).append(c); have[h] = c; n += 1
    json.dump(a, open(f"{AD}/{tag}.json", "w"))
for s, t, why, _ in json.load(open(bc)):
    a = json.load(open(f"{R}/drv/assign/{t}.json")); put(s, t, a, ev[t], why)
for p in sols:
    for x in json.load(open(p)):
        a = x["assign"]; mix = M.eval_mix(a, D["base"]); mix["act"] = D.get("act", "a8g128")
        put(x["size"], f"x203-{x['size']:g}-{x['var']}", a, mix, "extra-solve")
json.dump(C, open(cp + ".part", "w")); os.replace(cp + ".part", cp)
print("added", n, {s: len(v) for s, v in C.items() if s in ("3.25", "3.75", "6.5")})
