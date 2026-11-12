#!/usr/bin/env python3
"""pickx.py GOLDEN CORPUS META N SEED OUTDIR OUTCORPUS [--exclude-orig KLD.json ...] [--cap-cat CATS:SCORED ...]
-- the xcorp part of a two-part 27B search slice (w120, 11-12): v3/e3/settle/w84/pickx84.py (N pieces of an existing 27B
bf16 xcorp golden, stratified by category, largest remainder, fixed SEED; pieces whose xcorp document is in any
--exclude-orig golden's docs[].xdoc / docs[].orig are skipped) plus a per-category cap on scored positions:
--cap-cat agentic,tools,long,reasoning:2048 cuts every piece of those categories to its first 2048 scored positions
(tokens = score_from + 1 + 2048; the corpus line gets "maxtok", radiance's per-document token cap, as golden_pick.py
writes it). Prefix cuts keep the golden's rows exact (causal model). Why: on 27c / 27d / 27f one agentic / tools piece
held 23-54% of the xcorp part's KL at 8k (eval/slices/README.md). Output as pickx84 (kld.json docs[] with orig =
the piece's index in GOLDEN, xdoc, category; tokens.i32, top.bin, rows.f32, gen.u8 all ones; OUTDIR.docs)."""
import collections, json, os, random, sys

import numpy as np

a = sys.argv[1:]
excl_files, capcat = [], {}
while "--exclude-orig" in a:
    i = a.index("--exclude-orig"); excl_files.append(a[i + 1]); del a[i:i + 2]
while "--cap-cat" in a:
    i = a.index("--cap-cat"); cats, c = a[i + 1].rsplit(":", 1); del a[i:i + 2]
    for x in cats.split(","): capcat[x] = int(c)
g, corp, meta, N, seed, od, oc = a[0], a[1], a[2], int(a[3]), int(a[4]), a[5], a[6]
M = [json.loads(l) for l in open(meta)]
m = json.load(open(f"{g}/kld.json")); assert len(M) == len(m["docs"]), (len(M), len(m["docs"]))
for x, d in zip(M, m["docs"]): assert x["tokens"] == d["tokens"] and x["source"] == d["source"], (x, d)
EX = set()
for f in excl_files:                     # earlier x slices carry xdoc (orig = the piece index of THEIR golden); mX: orig
    for d in json.load(open(f))["docs"]: EX.add(d["xdoc"] if "xdoc" in d else d["orig"])
cand = collections.defaultdict(list)
for i, x in enumerate(M):
    if x["doc"] in EX: continue
    cand[x["category"]].append(i)
cnt = collections.Counter(x["category"] for x in M)
share = {c: N * cnt[c] / len(M) for c in cnt}
n = {c: min(max(1, int(share[c])), len(cand[c])) for c in cnt}
while sum(n.values()) < N:                                   # largest remainder
    c = max((c for c in cnt if len(cand[c]) > n[c]), key=lambda c: share[c] - n[c]); n[c] += 1
while sum(n.values()) > N:
    c = max((c for c in cnt if n[c] > 1), key=lambda c: n[c] - share[c]); n[c] -= 1
rng = random.Random(seed); pick = []
for c in sorted(cnt):
    idx = cand[c][:]; rng.shuffle(idx); pick += idx[:n[c]]
pick.sort()
print(f"{len(pick)} pieces ({len(EX)} xcorp docs excluded; caps {capcat or 'none'}): " + ", ".join(f"{c} {n[c]}" for c in sorted(n)), file=sys.stderr)
top = m["top"]
toks = np.memmap(f"{g}/tokens.i32", np.int32, "r")
tb = np.memmap(f"{g}/top.bin", np.uint8, "r").reshape(m["rows"], top * 8)
rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
lines = open(corp).read().rstrip("\n").split("\n"); assert len(lines) == len(M)
os.makedirs(od, exist_ok=True)
docs, T, TB, RW, out_lines, r, ncut = [], [], [], [], [], 0, 0
for i in pick:
    d = m["docs"][i]; nt = d["tokens"]; sf = d["score_from"]; k = nt - 1 - sf; assert k > 0
    c = capcat.get(M[i]["category"])
    line = lines[i]
    if c and k > c:
        k = c; nt = sf + 1 + c; ncut += 1
        j = json.loads(line); j["maxtok"] = nt; line = json.dumps(j, ensure_ascii=False)
    T.append(np.asarray(toks[toff[i]:toff[i] + nt])); TB.append(np.asarray(tb[d["row0"]:d["row0"] + k])); RW.append(np.asarray(rw[d["row0"]:d["row0"] + k]))
    docs.append({"source": d["source"], "tokens": nt, "score_from": sf, "row0": r, "orig": i, "xdoc": M[i]["doc"], "category": M[i]["category"]})
    out_lines.append(line); r += k
np.concatenate(T).tofile(f"{od}/tokens.i32"); np.concatenate(TB).tofile(f"{od}/top.bin"); np.concatenate(RW).tofile(f"{od}/rows.f32")
np.ones(r, np.uint8).tofile(f"{od}/gen.u8")
mm = dict(m); mm["rows"] = r; mm["docs"] = docs
json.dump(mm, open(f"{od}/kld.json", "w"))
open(oc, "w").write("\n".join(out_lines) + "\n")
open(f"{od}.docs", "w").write(" ".join(map(str, pick)) + "\n")
print(f"{od}: {len(pick)} docs, {r} rows ({ncut} cut to their category cap)", file=sys.stderr)
