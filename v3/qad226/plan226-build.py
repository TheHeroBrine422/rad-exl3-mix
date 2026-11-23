#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""plan226-build.py EXT_PLAN.json CALIB_DIR DEEP_IDS.json OUT.json [--front N] [--gold NAME=DIR:GENMASK ...] -- the deep-context QAD plan (w226, ledger use qad226-deep,
role train). Takes w222's extended plan (plan222-build.py) unchanged and adds documents deeper than the 8k cap:
  DEEP_IDS.json = list of calib doc ids (fit / search pool, > 8k tokens, outside verify / golden-jlong / train214 / j217), or of [golden, doc] pairs (new long goldens
  given with --gold NAME=DIR:GENMASK).
The N (default 24 = one per stream) longest become `front_items` (qad220: the first places of the order, one per stream, so they are trained from step 0: a stream
trains 1,024 tokens a step, ~200 steps in a 4M run); the rest go to the head of the extension's `items`. Rows are counted at the plan's cap (8192), as the trainer checks."""
import json, sys
import numpy as np
src, cdir, idp, outp = sys.argv[1:5]; a = sys.argv[5:]
NF = int(a[a.index("--front") + 1]) if "--front" in a else 24
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
# the k-th longest goes to stream SPREAD[k]: deep streams far apart in the step's stream order, so the pipeline (2 segments ahead) never holds two deep
# states at once (each is up to ~8 GiB a card at 256k)
SPREAD = [0, 12, 6, 18, 3, 15, 9, 21, 1, 13, 7, 19, 4, 16, 10, 22, 2, 14, 8, 20, 5, 17, 11, 23]
slot = [None] * NF
for k, (nm, i) in enumerate(ids[:NF]): slot[SPREAD[k] if NF == 24 else k] = [nm, i, rows(nm, i, CAP)]
assert len(ids) >= NF, f"{len(ids)} deep documents < --front {NF}: lower --front"
front = [x for x in slot if x is not None]; head = [[nm, i, rows(nm, i, CAP)] for nm, i in ids[NF:]]
out = dict(px); out["front_items"] = px.get("front_items", []) + front; out["items"] = head + px["items"]
out["ledger_use"] = "qad226-deep"; out["w226"] = {"from": src, "front": len(out["front_items"]), "head": len(head),
                                                 "deep_tokens": {f"{nm}:{i}": meta[nm][0][i]["tokens"] for nm, i in ids},
                                                 "deep_rows_uncapped": sum(rows(nm, i, 1 << 30) for nm, i in ids)}
for k in ("rows_total", "steps", "dense_steps", "departs_from_w214_at_step"): out.pop(k, None)   # w222's counts no longer hold
json.dump(out, open(outp, "w"))
print(f"{len(front)} front items (longest {front[0] if front else None}), {len(head)} at the head of the extension; deep rows uncapped {out['w226']['deep_rows_uncapped']}")
for nm, i, r in front + head: print(f"  {nm}:{i} tokens {meta[nm][0][i]['tokens']} rows@{CAP} {r} {'front' if [nm, i, r] in front else 'head'}")
