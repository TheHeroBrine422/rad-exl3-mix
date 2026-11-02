#!/usr/bin/env python3
"""index.py OUT.json CORPUS.jsonl... -- the line index build.py checks candidates against: blake2b-64 hashes of every
line >= 50 characters (stripped) of the challenge's published documents ("lines"), and of those lines that occur in
>= 3 documents ("boiler": template text, licence headers -- not evidence of a shared document)."""
import collections, hashlib, json, sys
cnt = collections.Counter()
for f in sys.argv[2:]:
    for l in open(f):
        p = json.loads(l)["prompt"]
        cnt.update({hashlib.blake2b(x.strip().encode(), digest_size=8).hexdigest() for x in p.split("\n") if len(x.strip()) >= 50})
json.dump({"lines": list(cnt), "boiler": [h for h, c in cnt.items() if c >= 3]}, open(sys.argv[1], "w"))
print(f"{len(cnt)} distinct long lines, {sum(c >= 3 for c in cnt.values())} boilerplate")
