#!/usr/bin/env python3
"""verify.py [N=300] -- sanity checks of the finished corpus: manifest ids unique and every (file, line) points at the line with that id;
`tokens` re-computed with the Qwen tokenizer on N random documents per origin; no xgap document an exact duplicate of a calib / public /
xcorp document (index) or of another xgap document; no `pisess` file anywhere; calib is the 4,789-document split only."""
import collections, glob, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rows = [json.loads(l) for l in open(f"{ROOT}/manifest.jsonl")]
print("manifest rows", len(rows), collections.Counter(r["origin"] for r in rows))
ids = collections.Counter(r["id"] for r in rows)
print("duplicate ids:", sum(1 for v in ids.values() if v > 1))
bad = [f for f in glob.glob(f"{ROOT}/**", recursive=True) if "pisess" in os.path.basename(f)]
print("pisess files under corpus root:", bad)
cache = {}
def line(r):
    if r["file"] not in cache: cache[r["file"]] = open(f"{ROOT}/{r['file']}").read().split("\n")
    return json.loads(cache[r["file"]][r["line"]])
rng = random.Random(1)
mismatch = tokdiff = 0; checked = 0
for o in ("calib", "xcorp-v1", "xgap-v1"):
    sub = [r for r in rows if r["origin"] == o]
    for r in rng.sample(sub, min(N, len(sub))):
        d = line(r); checked += 1
        if d["id"] != r["id"]: mismatch += 1
        if ntok(d["prompt"]) != r["tokens"] or d["tokens"] != r["tokens"]: tokdiff += 1
    cache.clear()
print(f"id/line mismatches {mismatch}, token-count mismatches {tokdiff} of {checked} sampled docs")
idx = load_index(); exact = collections.Counter(); seen = {}; dup_in = 0; n = 0
for f in sorted(glob.glob(f"{ROOT}/corpus/xgap-v1/*/*.jsonl")):
    for l in open(f):
        p = json.loads(l)["prompt"]; n += 1
        d = idx.check(p)
        if d and d[0] == "exact": exact[d[1]] += 1
        h = norm_hash(p)
        if h in seen: dup_in += 1
        seen[h] = 1
print(f"xgap docs {n}: exact duplicates of calib/public/xcorp {dict(exact)}, of another xgap doc {dup_in}")
print("calib docs:", sum(1 for r in rows if r["origin"] == "calib"), "(public split is never included)")
