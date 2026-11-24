#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""plan226-build.py EXT_PLAN.json CALIB_DIR DEEP_IDS.json OUT.json [--front N | --front-min TOKENS] [--gold NAME=DIR:GENMASK ...] -- the deep-context QAD plan (w226, ledger use qad226-deep,
role train). Takes w222's extended plan (plan222-build.py) unchanged and adds documents deeper than the 8k cap:
  DEEP_IDS.json = list of calib doc ids (fit / search pool, > 8k tokens, outside verify / golden-jlong / train214 / j217), or of [golden, doc] pairs (new long goldens
  given with --gold NAME=DIR:GENMASK).
The N (default 24 = one per stream; --front-min T: those > T tokens) longest become `front_items` (qad220: the first places of the order, one per stream, so they
are trained from step 0: a stream trains 1,024 tokens a step, ~200 steps in a 4M run); the rest go to the head of the extension's `items`. The front items take
the order's first places in a spread (van der Corput over the N places; N = 24: 0, 12, 6, 18, ..), the longest first, so deep documents sit on streams far apart.
--streams "S:NAME/DOC,NAME/DOC;S:..." instead: those documents head stream S's queue in that order (`front_items` entries [name, doc, rows, S]; deep
documents chained on a few streams, host RAM capping the deep states held at once); the other DEEP_IDS documents go to the extension head.
w226: a 24-doc front (ext-plan-226b) made the first ~15 steps 89 % calib from 24 docs and the 0.25M read 1.168 vs unhealed on j217; 226c fronts only the 7 > 20k. Rows are counted at the plan's cap (8192), as the trainer checks."""
import json, sys
import numpy as np
src, cdir, idp, outp = sys.argv[1:5]; a = sys.argv[5:]
NF = int(a[a.index("--front") + 1]) if "--front" in a else 24
FMIN = int(a[a.index("--front-min") + 1]) if "--front-min" in a else None
STREAMS = a[a.index("--streams") + 1] if "--streams" in a else None
golds = {"calib": (cdir, f"{cdir}/gen.u8")}
for k, x in enumerate(a):
    if x == "--gold": nm, rest = a[k + 1].split("=", 1); golds[nm] = tuple(rest.split(":", 1))
px = json.load(open(src)); CAP = px["cap"]
meta = {nm: (json.load(open(f"{d}/kld.json"))["docs"], np.fromfile(g, np.uint8)) for nm, (d, g) in golds.items()}
def rows(nm, i, cap):
    docs, gen = meta[nm]; d = docs[i]; j = np.arange(max(0, d["score_from"]), min(cap, d["tokens"] - 1)); r = d["row0"] + j - d["score_from"]
    return int(gen[r].sum()) if len(r) else 0
ids = [x if isinstance(x, list) else ["calib", x] for x in json.load(open(idp))]
have = {(nm, i) for nm, i, _ in px["items"] + px.get("front_items", [])}
assert not any((nm, i) in have for nm, i in ids), "a deep document is already in the plan"
ids.sort(key=lambda x: -meta[x[0]][0][x[1]]["tokens"])
if FMIN is not None: NF = sum(1 for nm, i in ids if meta[nm][0][i]["tokens"] > FMIN)
if STREAMS:
    tagged = []
    for part in STREAMS.split(";"):
        st, docs = part.split(":", 1)
        for x in docs.split(","): nm, i = x.split("/"); tagged.append([nm, int(i), int(st)])
    tk = {(nm, i) for nm, i, _ in tagged}; assert all([nm, i] in ids for nm, i in tk), "--streams lists a document not in DEEP_IDS"
    front = [[nm, i, rows(nm, i, CAP), st] for nm, i, st in tagged]; head = [[nm, i, rows(nm, i, CAP)] for nm, i in ids if (nm, i) not in tk]
    NF = 0
# the k-th longest goes to stream SPREAD[k]: deep streams far apart in the step's stream order, so the pipeline (2 segments ahead) never holds two deep
# states at once (each is up to ~8 GiB a card at 256k)
def spread(n):
    if n == 24: return [0, 12, 6, 18, 3, 15, 9, 21, 1, 13, 7, 19, 4, 16, 10, 22, 2, 14, 8, 20, 5, 17, 11, 23]     # ext-plan-226b's
    out, k = [], 0
    while len(out) < n:
        x, f, v = k, 0.5, 0.0
        while x: v += f * (x & 1); x >>= 1; f /= 2
        p = int(v * n)
        if p not in out: out.append(p)
        k += 1
    return out
SPREAD = spread(NF)
slot = [None] * NF
for k, (nm, i) in enumerate(ids[:NF]): slot[SPREAD[k]] = [nm, i, rows(nm, i, CAP)]
assert len(ids) >= NF, f"{len(ids)} deep documents < --front {NF}: lower --front"
if not STREAMS: front = [x for x in slot if x is not None]; head = [[nm, i, rows(nm, i, CAP)] for nm, i in ids[NF:]]
out = dict(px); out["front_items"] = px.get("front_items", []) + front; out["items"] = head + px["items"]
out["new_goldens"] = px["new_goldens"] + [nm for nm in golds if nm != "calib" and nm not in px["new_goldens"]]   # the trainer loads these from --ext-root/gold
out["ledger_use"] = "qad226-deep"; out["w226"] = {"from": src, "front": len(out["front_items"]), "head": len(head),
                                                 "deep_tokens": {f"{nm}:{i}": meta[nm][0][i]["tokens"] for nm, i in ids},
                                                 "deep_rows_uncapped": sum(rows(nm, i, 1 << 30) for nm, i in ids)}
for k in ("rows_total", "steps", "dense_steps", "departs_from_w214_at_step"): out.pop(k, None)   # w222's counts no longer hold
json.dump(out, open(outp, "w"))
print(f"{len(front)} front items (longest {front[0] if front else None}), {len(head)} at the head of the extension; deep rows uncapped {out['w226']['deep_rows_uncapped']}")
for x in front + head: print(f"  {x[0]}:{x[1]} tokens {meta[x[0]][0][x[1]]['tokens']} rows@{CAP} {x[2]} {('stream %d' % x[3]) if len(x) > 3 else 'front' if x in front else 'head'}")
