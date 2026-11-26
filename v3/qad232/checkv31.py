#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""checkv31.py CALIB_GOLDEN CALIB_CORPUS SLICE SLICE_CORPUS DOCIDX... -- w232 part B checks of a slice cut by eval/golden_pick.py (whole docs, no cap) from the published
calib golden: docs / orig indices (= the calib indices), per-doc tokens / score_from / rows, total rows = the sum of the calib rows of those docs, gen rows (gen.u8 sum) = the calib gen sum over
the same ranges, tokens / rows.f32 / top.bin / gen.u8 byte-equal to the calib ranges (ALL docs; the files are read in 64 MB pieces), the slice corpus lines = calib.jsonl's lines,
and the corpus hash recorded in kld.json (the calib's: slices cut from calib keep `corpus_fnv1a64` of calib.jsonl, as golden-jc228 / jlong do). Prints one summary line, exit 1 on failure."""
import json, sys
import numpy as np
g, cc, s, sc = sys.argv[1:5]; idx = [int(x) for x in sys.argv[5:]]
m = json.load(open(f"{g}/kld.json")); k = json.load(open(f"{s}/kld.json")); top = m["top"]
toks = np.memmap(f"{g}/tokens.i32", np.int32, "r"); toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
tb = np.memmap(f"{g}/top.bin", np.uint8, "r").reshape(m["rows"], top * 8); rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
gen = np.fromfile(f"{g}/gen.u8", np.uint8)
st = np.memmap(f"{s}/tokens.i32", np.int32, "r"); sb = np.memmap(f"{s}/top.bin", np.uint8, "r").reshape(k["rows"], top * 8)
sr = np.memmap(f"{s}/rows.f32", np.float32, "r").reshape(k["rows"], 3); sg = np.fromfile(f"{s}/gen.u8", np.uint8)
cl = open(cc).read().rstrip("\n").split("\n"); sl = open(sc).read().rstrip("\n").split("\n")
bad = []; r = 0; t = 0; gsum = 0; spot = 0
if [d["orig"] for d in k["docs"]] != idx: bad.append("slice doc orig indices != the requested calib indices")
if len(k["docs"]) != len(idx) or len(sl) != len(idx): bad.append(f"doc counts {len(k['docs'])} / corpus {len(sl)} vs {len(idx)} requested")
for j, i in enumerate(idx):
    d = m["docs"][i]; e = k["docs"][j]; n = d["tokens"] - 1 - d["score_from"]; r0 = d["row0"]
    if (e["tokens"], e["score_from"], e["row0"], e["source"]) != (d["tokens"], d["score_from"], r, d["source"]): bad.append(f"doc {i}: header {e} vs {d} (row0 {r})"); continue
    if not np.array_equal(st[t:t + d["tokens"]], toks[toff[i]:toff[i] + d["tokens"]]): bad.append(f"doc {i}: tokens differ")
    if not np.array_equal(sg[r:r + n], gen[r0:r0 + n]): bad.append(f"doc {i}: gen differs")
    if not np.array_equal(sr[r:r + n], rw[r0:r0 + n]): bad.append(f"doc {i}: rows.f32 differ")
    for a in range(0, n, 4096):
        b = min(n, a + 4096)
        if not np.array_equal(sb[r + a:r + b], tb[r0 + a:r0 + b]): bad.append(f"doc {i}: top.bin differs in rows {a}..{b}"); break
    if sl[j] != cl[i]: bad.append(f"doc {i}: corpus line differs")
    gsum += int(gen[r0:r0 + n].sum()); r += n; t += d["tokens"]
if k["rows"] != r or len(sg) != r or int(sg.sum()) != gsum: bad.append(f"rows {k['rows']} / gen {int(sg.sum())} vs calib ranges rows {r} gen {gsum}")
if k["corpus_fnv1a64"] != m["corpus_fnv1a64"]: bad.append("corpus hash differs from calib's")
print(f"checkv31 {s.rstrip('/').split('/')[-1]}: docs {len(idx)} rows {r} gen rows {gsum} tokens {t} max doc tokens {max(m['docs'][i]['tokens'] for i in idx)} "
      f"min score_from {min(m['docs'][i]['score_from'] for i in idx)} hash {k['corpus_fnv1a64']}: {'OK' if not bad else 'FAIL ' + '; '.join(bad[:6])}")
sys.exit(1 if bad else 0)
