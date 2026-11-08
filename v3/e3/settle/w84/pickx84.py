#!/usr/bin/env python3
"""pickx84.py GOLDEN CORPUS META N SEED OUTDIR OUTCORPUS [--exclude-orig KLD.json ...] -- the xcorp part of
golden-search27b (#117, ../w84.md amendment 1): N pieces of an existing 27B bf16 xcorp golden (night1105's xc8k-b1:
3,389 first-8k pieces of xcorp v1, TP2 --kld-record, never scored before), stratified by category (each category's
count proportional to its piece count in the golden, at least 1, largest remainder; uniform within a category, fixed
SEED), excluding pieces whose xcorp document is in any --exclude-orig golden's docs[].orig (e.g. hin27/mX, the #101
H_in blend's documents). Writes OUTDIR as golden_pick.py does (kld.json with docs[].orig = the piece's index in
GOLDEN plus "xdoc" / "category", tokens.i32, top.bin, rows.f32, gen.u8 = all scored, as the source golden has no
gen.u8), OUTCORPUS (the corpus lines) and OUTDIR.docs (one line: the picked piece indices)."""
import json, os, random, sys, collections
import numpy as np

a = sys.argv[1:]
excl_files = []
while "--exclude-orig" in a:
    i = a.index("--exclude-orig"); excl_files.append(a[i + 1]); del a[i:i + 2]
g, corp, meta, N, seed, od, oc = a[0], a[1], a[2], int(a[3]), int(a[4]), a[5], a[6]
M = [json.loads(l) for l in open(meta)]
m = json.load(open(f"{g}/kld.json")); assert len(M) == len(m["docs"]), (len(M), len(m["docs"]))
for x, d in zip(M, m["docs"]): assert x["tokens"] == d["tokens"] and x["source"] == d["source"], (x, d)
EX = {d["orig"] for f in excl_files for d in json.load(open(f))["docs"]}
cand = collections.defaultdict(list)
for i, x in enumerate(M):
    if x["doc"] in EX: continue
    cand[x["category"]].append(i)
cnt = collections.Counter(x["category"] for x in M)
share = {c: N * cnt[c] / len(M) for c in cnt}
n = {c: max(1, int(share[c])) for c in cnt}
while sum(n.values()) < N:                                   # largest remainder
    c = max((c for c in cnt if len(cand[c]) > n[c]), key=lambda c: share[c] - n[c]); n[c] += 1
while sum(n.values()) > N:
    c = max((c for c in cnt if n[c] > 1), key=lambda c: n[c] - share[c]); n[c] -= 1
rng = random.Random(seed); pick = []
for c in sorted(cnt):
    idx = cand[c][:]; rng.shuffle(idx); pick += idx[:n[c]]
pick.sort()
print(f"{len(pick)} pieces ({len(EX)} xcorp docs excluded): " + ", ".join(f"{c} {n[c]}" for c in sorted(n)), file=sys.stderr)
top = m["top"]
toks = np.memmap(f"{g}/tokens.i32", np.int32, "r")
tb = np.memmap(f"{g}/top.bin", np.uint8, "r").reshape(m["rows"], top * 8)
rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
lines = open(corp).read().rstrip("\n").split("\n"); assert len(lines) == len(M)
os.makedirs(od, exist_ok=True)
docs, T, TB, RW, r = [], [], [], [], 0
for i in pick:
    d = m["docs"][i]; nt = d["tokens"]; sf = d["score_from"]; k = nt - 1 - sf; assert k > 0
    T.append(np.asarray(toks[toff[i]:toff[i] + nt])); TB.append(np.asarray(tb[d["row0"]:d["row0"] + k])); RW.append(np.asarray(rw[d["row0"]:d["row0"] + k]))
    docs.append({"source": d["source"], "tokens": nt, "score_from": sf, "row0": r, "orig": i, "xdoc": M[i]["doc"], "category": M[i]["category"]}); r += k
np.concatenate(T).tofile(f"{od}/tokens.i32"); np.concatenate(TB).tofile(f"{od}/top.bin"); np.concatenate(RW).tofile(f"{od}/rows.f32")
np.ones(r, np.uint8).tofile(f"{od}/gen.u8")
mm = dict(m); mm["rows"] = r; mm["docs"] = docs
json.dump(mm, open(f"{od}/kld.json", "w"))
open(oc, "w").write("\n".join(lines[i] for i in pick) + "\n")
open(f"{od}.docs", "w").write(" ".join(map(str, pick)) + "\n")
print(f"{od}: {len(pick)} docs, {r} rows", file=sys.stderr)
