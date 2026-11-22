#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""blowup222.py GOLDEN CORPUS.jsonl MODEL_DIR CTLDIR OUT_PREFIX RESULTS.tsv[,RESULTS.tsv] SCORES_DIR [SCORES_DIR ...] -- D1 (w222): every document whose KL / gen row rises > 10x vs the unhealed release
(>= 8 gen rows, checkpoint >= 0.01 / row) in ANY served read found in the scores dirs (all `*.json.rows` except ctl-* / repro*; size from the results tsv, else the arm name's prefix 25 / 30 / 40 or
suffix; the same file under several names counts once). Per document: calib id (= ledger doc id), category / source, tokens, gen rows, unhealed KL / row, the reads that break it (arm, ckpt) and the
onset in each (first gen position whose 32-gen-row rolling mean KL is >= 10x the unhealed rolling mean there and >= 0.1; the KL before / after; the token with context, its script class, and how far
into the generation). Writes OUT_PREFIX.md (tables) and OUT_PREFIX.json."""
import collections, glob, json, os, re, sys, unicodedata
import numpy as np
g, corp, mdl, ctld, outp, tsvs = sys.argv[1:7]; dirs = sys.argv[7:]
m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool); toks = np.fromfile(f"{g}/tokens.i32", np.int32)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])]); docs = [json.loads(l) for l in open(corp).read().rstrip("\n").split("\n")]
SZ = {"25": "2.50", "30": "3.00", "40": "4.00"}
tsvsize = {}
for t in tsvs.split(","):
    if os.path.exists(t):
        for l in open(t):
            c = l.rstrip("\n").split("\t")
            if len(c) >= 19 and c[0] != "when": tsvsize[c[18]] = c[5]
def size_of(n):
    if n in tsvsize: return tsvsize[n]
    arm = n.split("-ckpt-")[0]
    if arm[:2] in SZ: return SZ[arm[:2]]
    mm = re.search(r"(25|30|40)$", arm) or re.search(r"-(25|30|40)(-|$)", arm)
    return SZ[mm.group(1)] if mm else None
def rows(f): return np.fromfile(f, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
def sl(i): d = m["docs"][i]; return slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"])
ND = len(m["docs"]); ctl = {s: rows(f"{ctld}/ctl-{s}.json.rows") for s in SZ.values()}
def per_doc(k):
    s = np.zeros(ND); n = np.zeros(ND)
    for i in range(ND): q = sl(i); gg = gen[q]; s[i] = k[q][gg].sum(); n[i] = gg.sum()
    return s, n
cdoc = {z: per_doc(k) for z, k in ctl.items()}
seen, reads, unknown = {}, {}, []
for d in dirs:
    for f in sorted(glob.glob(f"{d}/*.json.rows")):
        n = os.path.basename(f)[:-len(".json.rows")]
        if n.startswith("ctl-") or n.startswith("repro") or "ckpt" not in n: continue
        key = (os.path.getsize(f), os.stat(os.path.realpath(f)).st_ino if os.path.islink(f) else None)
        sz = size_of(n)
        if sz is None: unknown.append(n); continue
        h = hash(open(os.path.realpath(f), "rb").read()); 
        if h in seen: continue
        seen[h] = n; reads[n] = (sz, rows(f))
BLOW = {}
tot = collections.Counter(); per_read = {}
for n, (sz, k) in reads.items():
    s, nn = per_doc(k); cs, _ = cdoc[sz]; a = s / np.maximum(nn, 1); u = cs / np.maximum(nn, 1)
    b = np.where((nn >= 8) & (a >= 10 * u) & (a >= 0.01))[0]
    per_read[n] = [int(m["docs"][i]["orig"]) for i in b]
    for i in b: BLOW.setdefault(int(i), []).append(n)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(mdl)
def script(t):
    t = t.strip()
    if not t: return "whitespace / newline"
    c = t[0]
    if c.isdigit(): return "digit"
    if c.isalpha():
        nm = unicodedata.name(c, "?").split()[0]; return {"LATIN": "Latin", "CJK": "CJK", "HIRAGANA": "Japanese kana", "KATAKANA": "Japanese kana", "ARABIC": "Arabic", "CYRILLIC": "Cyrillic", "HANGUL": "Hangul", "DEVANAGARI": "Devanagari", "THAI": "Thai", "GREEK": "Greek", "HEBREW": "Hebrew"}.get(nm, nm.title())
    return "punctuation / symbol"
md = [f"# D1: documents whose KL / gen row rises > 10x vs unhealed (w222)\n", f"{len(reads)} distinct reads scanned ({len(unknown)} skipped, size unknown: {', '.join(unknown[:8]) or 'none'}); documents with >= 1 break: {len(BLOW)} of {sum(1 for d in range(ND) if cdoc['2.50'][1][d] >= 8)} with >= 8 gen rows.\n"]
cnt = collections.Counter(n for v in BLOW.values() for n in v)
md.append("Reads with any breaking document (arm-ckpt: ids): " + "; ".join(f"{n}: {','.join(map(str, v))}" for n, v in sorted(per_read.items()) if v) + "\n")
res = {}
for i in sorted(BLOW, key=lambda i: -len(BLOW[i])):
    d = m["docs"][i]; q = sl(i); gg = gen[q]; idx = np.where(gg)[0]; t0 = int(toff[i]); sf = d["score_from"]; nt = d["tokens"]
    g0 = int(sf + idx[0]) if len(idx) else 0
    md.append(f"\n## doc {d['orig']}: {docs[i].get('category')} / {d['source']} ({docs[i].get('kind')}), {nt} tokens, {int(gg.sum())} gen rows (generation starts at token {g0}); unhealed KL / row 2.50 {cdoc['2.50'][0][i] / max(1, cdoc['2.50'][1][i]):.4f}, 3.00 {cdoc['3.00'][0][i] / max(1, cdoc['3.00'][1][i]):.4f}, 4.00 {cdoc['4.00'][0][i] / max(1, cdoc['4.00'][1][i]):.4f}; breaks in {len(BLOW[i])} reads\n")
    md += ["| read | size | KL / row unhealed → read | onset position (into the doc / into the generation) | KL / row before → after | token there (context) | script |", "|---|---|---|---|---|---|---|"]
    r = {}
    for n in sorted(BLOW[i]):
        sz, k = reads[n]; kk, cc = k[q], ctl[sz][q]; a = float(kk[gg].sum() / max(1, gg.sum())); c = float(cc[gg].sum() / max(1, gg.sum()))
        pos = None; ba = ctxt = scr = "-"; jump = "-"
        if len(idx) >= 32:
            ra = np.convolve(kk[idx], np.ones(32) / 32, "valid"); rc = np.convolve(cc[idx], np.ones(32) / 32, "valid"); j = np.where((ra >= 10 * rc) & (ra >= 0.1))[0]
            if len(j):
                pos = int(sf + idx[j[0]]); before = float(kk[idx[:j[0]]].mean()) if j[0] > 0 else float("nan"); after = float(kk[idx[j[0]:]].mean())
                ctxt = tok.decode(toks[t0 + max(0, pos - 6): t0 + pos]).replace("\n", "\\n").replace("|", "/") + " ⟦" + tok.decode(toks[t0 + pos: t0 + pos + 1]).replace("\n", "\\n").replace("|", "/") + "⟧"
                scr = script(tok.decode(toks[t0 + pos: t0 + pos + 1])); jump = f"{pos} of {nt} / {pos - g0}"; ba = f"{before:.3f} → {after:.3f}"
        md.append(f"| {n} | {sz} | {c:.4f} → {a:.4f} | {jump} | {ba} | {ctxt} | {scr} |")
        r[n] = {"size": sz, "unhealed": c, "read": a, "onset_pos": pos, "onset_in_generation": None if pos is None else pos - g0, "script": scr}
    res[int(d["orig"])] = {"category": docs[i].get("category"), "source": d["source"], "kind": docs[i].get("kind"), "tokens": nt, "gen_rows": int(gg.sum()), "gen_start": g0, "reads": r}
# aggregates
byc = collections.Counter((res[o]["category"], res[o]["source"].split("/")[0]) for o in res)
md.append("\n## Aggregates\n"); md.append("Breaking documents by (category, corpus): " + ", ".join(f"{k[0]}/{k[1]}: {v}" for k, v in byc.most_common()))
sc = collections.Counter(v["script"] for o in res.values() for v in o["reads"].values() if v["script"] != "-"); md.append("\nScript of the onset token over all (doc, read) pairs: " + ", ".join(f"{k}: {v}" for k, v in sc.most_common()))
byarm = collections.Counter(re.sub(r"-ckpt-.*", "", n) for v in BLOW.values() for n in v); md.append("\nBreaks per arm (reads x documents): " + ", ".join(f"{k}: {v}" for k, v in byarm.most_common()))
open(outp + ".md", "w").write("\n".join(md) + "\n"); json.dump({"per_read": per_read, "docs": res, "unknown": unknown}, open(outp + ".json", "w"), indent=1); print("\n".join(md))
