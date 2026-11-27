#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""inv236.py QROOT CALIB_GOLDEN POOLS_V3 V31_SPLIT OUTDIR -- w236: the V3.1 finals' source inventory + held-out checks.

Writes to OUTDIR (keep on our machines: it lists per-document ids):
  calib-train.json     trainable calib doc ids (pools qad / fit / search + the 432 released verify docs), the 6 deep traces excluded
  calib-deep.json      the six released whole calib traces (docs > 200k tokens outside verify_v31)
  xg-deep.json         the 2 longest xg-fin agent-sessions docs (deep traces 7, 8)
  inventory.json       per source / pool: docs, tokens, gen rows (calib gen mask), length bands, overlap checks
Overlap checks (printed): calib ids vs verify_v31 / golden-v31ver / golden-v31tr; token-window hashes (first 256 tokens and a 256-token window at
the doc's middle, docs >= 1k tokens; "pre:" hits are mostly shared system prompts, "mid:" hits are shared text) of every training doc vs every held-out golden (verify_v31 docs, v31ver, v31tr, v31x, jgap, jt1, jt2, public); xg_id / src_doc
overlaps of xg-fin / tset1 / tset2 with jgap / v31x / jt1 / jt2."""
import hashlib, json, os, sys
import numpy as np

q, calib_dir, pools_f, split_f, out = sys.argv[1:6]
sys.path.insert(0, os.path.join(q, "code", "v3", "qad181"))
import data181 as D                                      # noqa: E402

os.makedirs(out, exist_ok=True)
pool = json.load(open(pools_f))["pool"]; split = json.load(open(split_f))
v31, rel = set(split["verify_v31"]), set(split["released"])
assert not v31 & rel and all(pool[i] == "verify" for i in v31 | rel) and len(v31 | rel) == sum(p == "verify" for p in pool)
C = D.Gold(calib_dir, "calib")
inv = {}


def band(n): return next(b for b in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1 << 30) if n <= b)


def stats(g, ids, cap=1 << 30):
    docs = len(ids); tok = sum(min(cap, g.ntok(i)) for i in ids); rows = sum(g.nrows(i, cap) for i in ids)
    bands = {}
    for i in ids:
        b = band(g.ntok(i)); x = bands.setdefault(b, [0, 0, 0]); x[0] += 1; x[1] += g.ntok(i); x[2] += g.nrows(i, 1 << 30)
    return {"docs": docs, "tokens": tok, "rows": rows, "bands_docs_tok_rows": {str(k): v for k, v in sorted(bands.items())}}


groups = {"qad": [], "fit": [], "search": [], "verify_released": [], "verify_v31": []}
for i, p in enumerate(pool):
    groups["verify_v31" if i in v31 else "verify_released" if i in rel else p].append(i)
long_ = [i for i in range(len(pool)) if C.ntok(i) > 200000]
print("calib docs > 200k tokens:", [(i, pool[i], "v31" if i in v31 else "rel" if i in rel else "", C.ntok(i), C.nrows(i, 1 << 30)) for i in long_])
deep = [i for i in long_ if i not in v31]
assert len(deep) == 6, deep
train = [i for g in ("qad", "fit", "search", "verify_released") for i in groups[g] if i not in deep]
inv["calib"] = {g: stats(C, ids) for g, ids in groups.items()}
inv["calib"]["train_no_deep"] = stats(C, train); inv["calib"]["deep6"] = stats(C, deep)
inv["calib"]["train_no_deep_cap8192"] = stats(C, train, 8192)
inv["calib"]["deep6_docs"] = [[i, pool[i], C.ntok(i), C.nrows(i, 1 << 30)] for i in deep]
assert not set(train) & v31 and not set(deep) & v31
json.dump(train, open(f"{out}/calib-train.json", "w")); json.dump(deep, open(f"{out}/calib-deep.json", "w"))

ext = f"{q}/ext"
src = {}
for nm in ("xg-fin", "tset1", "tset2"):
    g = D.Gold(f"{ext}/gold/{nm}", nm, genfile=f"{ext}/genmask/{nm}.u8")
    meta = [json.loads(l) for l in open(f"{ext}/gold/{nm}/corpus.meta.jsonl")]
    assert len(meta) == len(g.docs); src[nm] = (g, meta)
    inv[nm] = stats(g, list(range(len(g.docs)))); inv[nm]["cap131072"] = stats(g, list(range(len(g.docs))), 131072)
g, meta = src["xg-fin"]
ag = sorted((i for i, m in enumerate(meta) if m.get("cell") == "agent-sessions"), key=lambda i: -g.ntok(i))
xdeep = ag[:2]
inv["xg-fin"]["agent_top8"] = [[i, meta[i]["source"], g.ntok(i), g.nrows(i, 1 << 30)] for i in ag[:8]]
json.dump(xdeep, open(f"{out}/xg-deep.json", "w"))

# held-out sets
def slice_gold(nm): return D.Gold(f"{q}/slices/{nm}", nm)
held = {nm: slice_gold(nm) for nm in ("golden-v31ver", "golden-v31tr", "golden-v31x", "golden-jgap", "golden-jt1", "golden-jt2")}
held["public"] = D.Gold(os.path.join(os.path.dirname(calib_dir.rstrip("/")), "public"), "public")
chk = {}
for nm in ("golden-v31ver", "golden-v31tr"):
    o = {d["orig"] for d in held[nm].docs}; chk[f"{nm}_orig_in_verify_v31"] = len(o & v31); chk[f"{nm}_orig_in_train"] = len(o & (set(train) | set(deep)))
    chk[f"{nm}_docs"] = len(o)


def keys(g, i):
    n = g.ntok(i); a = g.ids(i, 0, min(256, n)).tobytes(); m = n // 2; b = g.ids(i, max(0, m - 128), min(n, m + 128)).tobytes()
    return [("pre", hashlib.blake2b(a, digest_size=12).digest())] + ([("mid", hashlib.blake2b(b, digest_size=12).digest())] if n >= 1024 else [])


hk = {}
for nm, g in held.items():
    for i in range(len(g.docs)):
        for k in keys(g, i): hk.setdefault(k, set()).add(nm)
for i in v31:
    for k in keys(C, i): hk.setdefault(k, set()).add("verify_v31")
trainsets = {"calib-train": (C, train), "calib-deep": (C, deep)}
for nm, (g, meta) in src.items(): trainsets[nm] = (g, range(len(g.docs)))
for tn, (g, ids) in trainsets.items():
    hits = {}
    for i in ids:
        for k in keys(g, i):
            for h in hk.get(k, ()): hits.setdefault(f"{k[0]}:{h}", set()).add(i)
    chk[f"hash_{tn}"] = {h: len(v) for h, v in hits.items()}
    chk[f"hash_{tn}_examples"] = {h: sorted(v)[:5] for h, v in hits.items()}
# id-level overlaps of the external sources with the judges
xid = lambda m: m.get("xg_id") or m.get("src_doc")
jg = {xid(json.loads(l)) for l in open(f"{q}/slices/golden-jgap/corpus.meta.jsonl")}
v31x = {xid(json.loads(l)) for l in open(f"{q}/slices/golden-v31x.jsonl.meta.jsonl")}
jt1 = {json.loads(l)["src_doc"] for l in open(f"{q}/slices/golden-jt1.jsonl.meta.jsonl")}
jt2 = {json.loads(l)["src_doc"] for l in open(f"{q}/slices/golden-jt2.jsonl.meta.jsonl")}
for nm, (g, meta) in src.items():
    ids_ = {xid(m) for m in meta}
    chk[f"id_{nm}"] = {"jgap": len(ids_ & jg), "v31x": len(ids_ & v31x), "jt1": len(ids_ & jt1), "jt2": len(ids_ & jt2)}
    chk[f"holdout_flag_{nm}"] = sum(1 for m in meta if m.get("holdout"))
t1 = {m["src_doc"] for m in src["tset1"][1]}; xf = {m["xg_id"] for m in src["xg-fin"][1]}
chk["tset1_sessions_also_in_xg-fin"] = len(t1 & xf)
chk["xg_deep_also_tset1_session"] = [src["xg-fin"][1][i]["xg_id"] in t1 for i in xdeep]
inv["checks"] = chk
json.dump(inv, open(f"{out}/inventory.json.part", "w"), indent=1); os.replace(f"{out}/inventory.json.part", f"{out}/inventory.json")
print(json.dumps(inv, indent=1))
