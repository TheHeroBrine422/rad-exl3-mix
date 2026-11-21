#!/usr/bin/env python3
"""mksmoke.py GOLD_DIR OUT_DIR N_DOCS CAP -- first N_DOCS documents of a golden, each cut to its first CAP tokens (smoke test)."""
import json, os, sys
import numpy as np
g, o, nd, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
k = json.load(open(os.path.join(g, "kld.json")))
top = np.memmap(os.path.join(g, "top.bin"), np.uint8, "r").reshape(-1, 4096)
rw = np.memmap(os.path.join(g, "rows.f32"), np.uint8, "r").reshape(-1, 12)
ge = np.fromfile(os.path.join(g, "gen.u8"), np.uint8)
tk = np.fromfile(os.path.join(g, "tokens.i32"), "<i4")
os.makedirs(o, exist_ok=True)
T, TOP, RW, GE, docs, off, r0 = [], [], [], [], [], 0, 0
for d in k["docs"][:nd]:
    n = min(d["tokens"], cap)
    T.append(tk[off:off + n]); TOP.append(top[d["row0"]:d["row0"] + n - 1]); RW.append(rw[d["row0"]:d["row0"] + n - 1])
    GE.append(ge[d["row0"]:d["row0"] + n - 1]); docs.append(dict(source=d["source"], tokens=n, score_from=0, row0=r0))
    r0 += n - 1; off += d["tokens"]
np.concatenate(T).astype("<i4").tofile(os.path.join(o, "tokens.i32"))
np.concatenate(TOP).tofile(os.path.join(o, "top.bin")); np.concatenate(RW).tofile(os.path.join(o, "rows.f32"))
np.concatenate(GE).tofile(os.path.join(o, "gen.u8"))
k.update(rows=r0, docs=docs)
json.dump(k, open(os.path.join(o, "kld.json"), "w"))
print(r0, "rows")
