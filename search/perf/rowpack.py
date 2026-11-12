#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rowpack.py ROOT OUT.npz -- compact radiance --kld-out score rows into one file for the slice-size study (w123, #161).

ROOT holds score reports (*.json with "ref_dir", + .json.rows) and the goldens they name (kld.json + gen.u8), under the
same relative paths as on the scoring host (ref_dir's leading /data/ stripped). Only scored positions (the golden's gen
mask) are kept, per document from row0 over tokens - 1 - score_from rows (eval/docstat convention).

OUT.npz: per golden G (key prefix "G/"): doc_n (scored positions per document), doc_rows, doc_tok (tokens prefilled),
gpos (each scored position's row inside its document: the prefill cost of a per-document cap), doc_src, doc_cat, and per score
report T scored on G: "G/kl/T" (float32 KL at every scored position, documents in order) and "G/t1/T" (uint8 top-1
agreement). index.json beside OUT lists goldens -> reports with their mean KL / positions (checked against the report)."""
import glob, json, os, sys

import numpy as np

root, out = sys.argv[1], sys.argv[2]
gold, arrs, index = {}, {}, {}
for f in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
    if os.path.basename(f) == "kld.json" or not os.path.exists(f + ".rows"): continue
    j = json.load(open(f)); g = j["ref_dir"].replace("/data/", "", 1); gname = os.path.basename(g)
    if not os.path.exists(f"{root}/{g}/kld.json"):          # a copy of the same golden elsewhere (checked equal by md5)
        g = os.path.relpath(os.path.dirname(sorted(glob.glob(f"{root}/**/{gname}/kld.json", recursive=True))[0]), root)
    if g not in gold:
        m = json.load(open(f"{root}/{g}/kld.json")); gen = np.fromfile(f"{root}/{g}/gen.u8", np.uint8).astype(bool)
        sl = [slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]) for d in m["docs"]]
        gold[g] = (gname, m, gen, sl)
        arrs[f"{gname}/doc_n"] = np.array([int(gen[s].sum()) for s in sl], np.int64)
        arrs[f"{gname}/doc_rows"] = np.array([s.stop - s.start for s in sl], np.int64)      # rows prefilled per doc
        arrs[f"{gname}/doc_tok"] = np.array([d["tokens"] for d in m["docs"]], np.int64)        # tokens prefilled
        arrs[f"{gname}/gpos"] = np.concatenate([np.nonzero(gen[s])[0] for s in sl]).astype(np.int32)  # row of each scored pos
        arrs[f"{gname}/doc_src"] = np.array([str(d.get("source", "?")) for d in m["docs"]])
        arrs[f"{gname}/doc_cat"] = np.array([str(d.get("category", "calib")) for d in m["docs"]])
        index[gname] = {"path": g, "docs": len(sl), "positions": int(arrs[f"{gname}/doc_n"].sum()), "reports": {}}
    gname, m, gen, sl = gold[g]
    R = np.fromfile(f + ".rows", np.float32).reshape(-1, 4)
    if len(R) != len(gen): raise SystemExit(f"{f}: {len(R)} rows vs {len(gen)}")
    kl = np.concatenate([R[s, 0][gen[s]] for s in sl]); t1 = np.concatenate([R[s, 3][gen[s]] for s in sl])
    tag = os.path.relpath(f, root)[:-5].replace("/scores/", ":")
    full = R[:, 0].astype(np.float64).mean()               # the report counts every row; the score counts gen rows
    if abs(full - j["all"]["kld"]["mean"]) > 1e-5 * max(1, j["all"]["kld"]["mean"]) or len(R) != j["all"]["positions"]:
        raise SystemExit(f"{f}: mean {full} / {len(R)} vs report {j['all']['kld']['mean']} / {j['all']['positions']}")
    arrs[f"{gname}/kl/{tag}"] = kl.astype(np.float32); arrs[f"{gname}/t1/{tag}"] = (t1 > 0.5).astype(np.uint8)
    index[gname]["reports"][tag] = {"model": j["model"], "kld_gen": float(kl.astype(np.float64).mean()), "positions": len(kl), "kld_report": j["all"]["kld"]["mean"]}
np.savez_compressed(out, **arrs)
json.dump(index, open(out[:-4] + ".index.json", "w"), indent=1)
for g, v in index.items(): print(g, v["docs"], v["positions"], len(v["reports"]))
