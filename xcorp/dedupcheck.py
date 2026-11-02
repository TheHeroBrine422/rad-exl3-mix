#!/usr/bin/env python3
"""dedupcheck.py INDEX.json CORPUS.jsonl -- build.py's in-challenge rule applied to a corpus: share of documents it
rejects (on calib itself: should be ~all documents that have >= 2 non-boilerplate long lines)."""
import hashlib, json, sys
idx = json.load(open(sys.argv[1])); bad = set(idx["lines"]); boil = set(idx["boiler"])
h = lambda x: hashlib.blake2b(x.encode(), digest_size=8).hexdigest()
n = rej = elig = 0
for l in open(sys.argv[2]):
    p = json.loads(l)["prompt"]; n += 1
    ll = [x.strip() for x in p.split("\n") if len(x.strip()) >= 50 and h(x.strip()) not in boil]
    hit = sum(h(x) in bad for x in ll); elig += len(ll) >= 2
    rej += hit >= 2 and hit >= 0.2 * len(ll)
print(f"{n} docs, {elig} with >= 2 long lines, rejected {rej}")
