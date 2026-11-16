#!/usr/bin/env python3
"""assemble197.py OUT.bf16 VOCAB D VROW.bin [--cls ID_FILE:BF16_FILE ...] -- #197 (w185): scatter a row-split head
(KL-IQR's output.kB classes, each with its vocabulary ids) into one [VOCAB, D] bf16 matrix.
Checks: the ids of all classes partition 0..VOCAB-1 exactly; the class-local row of every id agrees with vrow
(vrow[id] = class index in the top 4 bits, row in the low 28 bits, if the file's layout is that; otherwise reported).
Prints one JSON line. Run with `python3 -I`."""
import json, os, sys
import numpy as np

a = sys.argv[1:]
out, V, D, vrow_f = a[0], int(a[1]), int(a[2]), a[3]
pairs = []
vals = None
if "--cls-values" in a:
    j = a.index("--cls-values"); vals = [int(x) for x in a[j + 1].split(",")]; del a[j:j + 2]
i = a.index("--cls") + 1
while i < len(a):
    idf, bff = a[i].split(":"); pairs.append((idf, bff)); i += 1
if vals is None: vals = list(range(len(pairs)))   # vrow's top 4 bits = the class's width (3..7), given by --cls-values
vrow = np.fromfile(vrow_f, dtype="<i4") if vrow_f != "-" else None
res = {"out": out, "classes": []}
seen = np.zeros(V, dtype=np.int64)
cls_rows = []
for idf, bff in pairs:
    ids = np.fromfile(idf, dtype="<i4")
    n = ids.size
    assert os.path.getsize(bff) == n * D * 2, (bff, os.path.getsize(bff), n, D)
    assert ids.min() >= 0 and ids.max() < V
    np.add.at(seen, ids, 1)
    cls_rows.append((ids, bff))
    res["classes"].append({"ids": idf, "rows": int(n)})
res["covered_once"] = bool((seen == 1).all())
res["n_multi"] = int((seen > 1).sum()); res["n_missing"] = int((seen == 0).sum())
mm = np.memmap(out + ".part", dtype="<u2", mode="w+", shape=(V, D))
for ids, bff in cls_rows:
    src = np.memmap(bff, dtype="<u2", mode="r", shape=(ids.size, D))
    step = 8192
    for r0 in range(0, ids.size, step):
        mm[ids[r0:r0 + step]] = src[r0:r0 + step]
    del src
mm.flush(); del mm
os.replace(out + ".part", out)
if vrow is not None and vrow.size == V:
    cls_val = (vrow.astype(np.uint32) >> 28).astype(np.int64)
    row_idx = (vrow.astype(np.uint32) & 0x0FFFFFFF).astype(np.int64)
    ok = 0; bad = 0
    for k, (ids, bff) in enumerate(cls_rows):
        sel = np.where(cls_val == vals[k])[0]
        if sel.size == 0 or sel.size != ids.size: bad += ids.size; continue
        good = (ids[row_idx[sel]] == sel)
        ok += int(good.sum()); bad += int((~good).sum())
    res["vrow_check"] = {"ok": ok, "bad": bad, "values": vals, "class_counts": {str(v): int((cls_val == v).sum()) for v in sorted(set(cls_val.tolist()))}}
print(json.dumps(res))
