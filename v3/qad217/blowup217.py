#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""blowup217.py SCORES_DIR GOLDEN CORPUS.jsonl MODEL_DIR [--json OUT] -- the blow-up documents of #217 (RULES217: an arm's document KL / gen row >= 0.3 and >= 5x the unhealed
control's, at 0.25 M or 0.5 M). For every document that blows up under any scored checkpoint (all of SCORES_DIR/*-ckpt-*.json.rows, sizes by the arm name's prefix 25 / 30 / 40):
id (calib corpus index = ledger doc id), category / source, tokens, gen rows; per checkpoint that blows up it: KL / row unhealed vs the checkpoint; where in the document the KL
jumps (first position whose 32-row rolling mean KL over gen rows is >= 0.3 and >= 5x the control's rolling mean there, the KL / row before and after it) and the token there with
a few tokens of context (decoded). Table of every checkpoint's KL / row on each such document to stdout / OUT."""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
sd, g, corp, mdl = sys.argv[1:5]; out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool); toks = np.fromfile(f"{g}/tokens.i32", np.int32)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])]); docs = [json.loads(l) for l in open(corp).read().rstrip("\n").split("\n")]
SZ = {"25": "2.50", "30": "3.00", "40": "4.00"}
def rows(f): return np.fromfile(f, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
def sl(i): d = m["docs"][i]; return slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"])
ctl = {s: rows(f"{sd}/ctl-{s}.json.rows") for s in SZ.values() if os.path.exists(f"{sd}/ctl-{s}.json.rows")}
cks = {}
for f in sorted(glob.glob(f"{sd}/*-ckpt-*.json.rows")):
    n = os.path.basename(f)[:-len(".json.rows")]; p = n[:2]
    if p in SZ and SZ[p] in ctl and not n.startswith("repro"): cks[n] = (SZ[p], rows(f))
def per_doc(k, i): s = sl(i); gg = gen[s]; return float(k[s][gg].sum() / max(1, gg.sum()))
bl = {}
for n, (sz, k) in cks.items():
    for i in range(len(m["docs"])):
        a, c = per_doc(k, i), per_doc(ctl[sz], i)
        if a >= 0.3 and a >= 5 * c: bl.setdefault(i, []).append(n)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(mdl)
res = {}; md = []
for i in sorted(bl):
    d = m["docs"][i]; s = sl(i); gg = gen[s]; t0 = int(toff[i]); sf = d["score_from"]; nt = d["tokens"]
    md.append(f"\n### doc {d['orig']} ({docs[i].get('category')}, {d['source']}): {nt} tokens, {int(gg.sum())} gen rows")
    md += ["| checkpoint | KL / row (unhealed → checkpoint) | blows up | jump at position (rolling 32 gen-row mean ≥ 0.3 and ≥ 5× control) | KL / row before → after | token there (context) |", "|---|---|---|---|---|---|"]
    r = {}
    for n, (sz, k) in sorted(cks.items(), key=lambda x: (x[1][0], x[0])):
        a, c = per_doc(k, i), per_doc(ctl[sz], i)
        kk, cc = k[s], ctl[sz][s]; idx = np.where(gg)[0]
        if len(idx) >= 32:
            ra = np.convolve(kk[idx], np.ones(32) / 32, "valid"); rc = np.convolve(cc[idx], np.ones(32) / 32, "valid")
            j = np.where((ra >= 0.3) & (ra >= 5 * rc))[0]
        else: j = []
        if a >= 0.3 and a >= 5 * c and len(j):
            pos = int(sf + idx[j[0]]); before = float(kk[idx[:j[0]]].mean()) if j[0] > 0 else float("nan"); after = float(kk[idx[j[0]:]].mean())
            tk = [int(x) for x in toks[t0 + max(0, pos - 6): t0 + pos + 3]]
            ctxt = tok.decode(toks[t0 + max(0, pos - 6): t0 + pos]).replace("\n", "\\n") + " ⟦" + tok.decode(toks[t0 + pos: t0 + pos + 1]).replace("\n", "\\n") + "⟧"
            jump = f"{pos} of {nt}"; ba = f"{before:.3f} → {after:.3f}"
        else: pos = None; jump = ba = ctxt = "-"
        r[n] = {"ctl": c, "arm": a, "blowup": bool(a >= 0.3 and a >= 5 * c), "jump_pos": pos}
        if a >= 0.3 or n in bl[i] or True: md.append(f"| {n} | {c:.4f} → {a:.4f} | {'**yes**' if r[n]['blowup'] else 'no'} | {jump} | {ba} | {ctxt} |")
    res[int(d["orig"])] = {"category": docs[i].get("category"), "source": d["source"], "tokens": nt, "gen_rows": int(gg.sum()), "checkpoints": r}
print("\n".join(md))
if out: json.dump(res, open(out, "w"), indent=1)
