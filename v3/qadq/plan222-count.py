#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""plan222-count.py META_DIR OUT_DIR [--tok TOKENIZER.json] -- count every usable QAD gen row and write the extension plan inputs (w222, user 11-22). META_DIR holds per golden dir its kld.json + tokens.i32 (+ corpus.meta.jsonl),
for calib (`calib/`: kld.json + the official gen.u8), j217 (`j217/kld.json`) and the xcorp / trace goldens of Thor's backup.
Calib: pools-v3.json (qad / fit / search / verify) minus the verify pool minus j217's documents minus the ledger's judge slices (variant V2: ledger roles check / final / monitor, plus the QAD `select` slice
qad218-tune); variants V1 (verify + j217 only) and V3 (also every ledger `select` slice) are counted too. xcorp / trace goldens: the challenge's gen mask rebuilt by genmask181.py's rule from the tokens
(checked against w214's masks), documents deduplicated across goldens by their first 64 tokens. Rows are counted as the trainer uses them: the first CAP (8192) tokens of a document, documents with >= 16 gen rows there;
also uncapped. Writes OUT_DIR/counts.json, OUT_DIR/genmask/<golden>.u8 (one byte a row), OUT_DIR/ext-docs.json (the extension's documents: golden, doc index, tokens, rows)."""
import collections, hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import genmask181 as G
from tokenizers import Tokenizer

meta, out = sys.argv[1:3]; tokp = sys.argv[sys.argv.index("--tok") + 1] if "--tok" in sys.argv else f"{meta}/tok/tokenizer.json"
CAP, MINR = 8192, 16
REPO = os.path.join(HERE, "..", "..")
tk = Tokenizer.from_file(tokp); S, E = tk.token_to_id("<|im_start|>"), tk.token_to_id("<|im_end|>")
A, NL = tk.encode("assistant", add_special_tokens=False).ids, tk.encode("\n", add_special_tokens=False).ids; assert len(A) == 1 and len(NL) == 1
A, NL = A[0], NL[0]
os.makedirs(f"{out}/genmask", exist_ok=True)


def capped_rows(d, gen_doc, cap=CAP):
    """gen rows of one document the way Gold.scored(i, 0, min(cap, n)) counts them: positions j in [score_from, min(cap, n, n - 1))"""
    n = d["tokens"]; j0, j1 = d["score_from"], min(cap, n, n - 1)
    return int(gen_doc[j0:j1].sum()) if j1 > j0 else 0


res = {"cap": CAP, "min_rows": MINR}
# ---------------- calib
m = json.load(open(f"{meta}/calib/kld.json")); gen = np.fromfile(f"{meta}/calib/gen.u8", np.uint8); docs = m["docs"]
pools = json.load(open(os.path.join(REPO, "eval/pools/pools-v3.json")))["pool"]; assert len(pools) == len(docs)
orig = [d.get("orig", i) for i, d in enumerate(docs)]
j217 = {int(d["orig"]) for d in json.load(open(f"{meta}/j217/kld.json"))["docs"]}
led = [json.loads(l) for l in open(os.path.join(REPO, "eval/pools/ledger.jsonl"))]
roles = collections.defaultdict(set)
for x in led: roles[x["role"]].add(x["doc"])
sel_qad = {x["doc"] for x in led if x["use"].startswith("qad") and x["role"] == "select"}
v2x = roles["check"] | roles["final"] | roles["monitor"] | sel_qad; v3x = v2x | roles["select"]
pools214 = json.load(open("/workspace/data-ext/pools214.json"))["pool"]; train214 = {i for i, p in enumerate(pools214) if p == "train214"}
rows_doc = []; capped = []
for i, d in enumerate(docs):
    r = np.arange(d["row0"], d["row0"] + max(0, d["tokens"] - 1 - d["score_from"])); g = gen[r] if len(r) else np.zeros(0, np.uint8)
    gd = np.zeros(d["tokens"], np.uint8); gd[d["score_from"]: d["score_from"] + len(g)] = g       # position j's row
    rows_doc.append(int(g.sum())); capped.append(capped_rows(d, gd))
rows_doc, capped = np.array(rows_doc), np.array(capped)
def tally(sel):
    sel = np.array(sel, bool); return {"docs": int(sel.sum()), "gen_rows": int(rows_doc[sel].sum()), "capped_rows": int(capped[sel].sum()), "docs_ge16_capped": int(((capped >= MINR) & sel).sum()), "capped_rows_ge16": int(capped[sel & (capped >= MINR)].sum())}
isp = lambda p: np.array([x == p for x in pools])
res["calib"] = {"all": tally(np.ones(len(docs), bool)), "by_pool": {p: tally(isp(p)) for p in ("qad", "fit", "search", "verify")}}
bad = {"V1": set(i for i, p in enumerate(pools) if p == "verify") | j217, "V2": set(i for i, p in enumerate(pools) if p == "verify") | j217 | v2x, "V3": set(i for i, p in enumerate(pools) if p == "verify") | j217 | v3x}
res["calib"]["excluded_docs"] = {"verify pool": int(isp("verify").sum()), "j217": len(j217), "ledger check/final/monitor": len(roles["check"] | roles["final"] | roles["monitor"]), "qad select (qad218-tune)": len(sel_qad), "ledger select (all)": len(roles["select"])}
for v, b in bad.items():
    keep = np.array([i not in b for i in range(len(docs))]); res["calib"][f"eligible_{v}"] = {"all": tally(keep), "by_pool": {p: tally(keep & isp(p)) for p in ("qad", "fit", "search")}, "of_which_train214": tally(keep & np.array([i in train214 for i in range(len(docs))]))}
keep2 = np.array([i not in bad["V2"] for i in range(len(docs))])
res["calib"]["train214_in_V2"] = int(sum(1 for i in train214 if keep2[i]))
ext_calib = [i for i in range(len(docs)) if keep2[i] and i not in train214 and capped[i] >= MINR]
# ---------------- xcorp / trace goldens
tok_ids = (S, E, A, NL)
seen = {}; ext = []; xs = {}
W14 = {"xc8k-b2-c0", "xc8k-b2-c1", "tr32k-b2-c0"}
for name in ("xc8k-b2-c0", "xc8k-b2-c1", "tr32k-b2-c0", "xc8k-b2-c2", "xc8k-b1", "tr32k-b1"):
    gd = f"{meta}/{name}"; mm, tok, off = G.load(gd)
    mask = G.rows_mask(mm["docs"], tok, off, S, E, A, NL, mm["rows"]); mask.tofile(f"{out}/genmask/{name}.u8")
    st = collections.Counter(); per = []; dup = 0
    for i, d in enumerate(mm["docs"]):
        t = np.asarray(tok[off[i]: off[i] + d["tokens"]]); h = hashlib.sha1(t[:64].tobytes() + str(d["tokens"]).encode()).hexdigest()
        r = np.arange(d["row0"], d["row0"] + max(0, d["tokens"] - 1 - d["score_from"])); g = mask[r] if len(r) else np.zeros(0, np.uint8)
        gdp = np.zeros(d["tokens"], np.uint8); gdp[d["score_from"]: d["score_from"] + len(g)] = g
        c = capped_rows(d, gdp); isdup = h in seen; seen.setdefault(h, (name, i)); dup += isdup
        per.append((c, int(g.sum()), d["tokens"], isdup, d["source"]))
        if name not in W14 and not isdup and c >= MINR: ext.append({"golden": name, "doc": i, "tokens": min(CAP, d["tokens"]), "rows": c, "source": d["source"]})
    xs[name] = {"docs": len(per), "rows_all": int(mm["rows"]), "gen_rows": int(sum(p[1] for p in per)), "capped_gen_rows": int(sum(p[0] for p in per)), "docs_ge16_capped": sum(1 for p in per if p[0] >= MINR),
                "capped_rows_ge16": int(sum(p[0] for p in per if p[0] >= MINR)), "duplicate_docs": int(dup), "w214_base": name in W14,
                "by_source": {k: {"docs": sum(1 for p in per if p[4] == k), "capped_gen_rows": int(sum(p[0] for p in per if p[4] == k))} for k in sorted({p[4].split("/")[0] for p in per} for _ in [0]) } if False else None}
    src = collections.defaultdict(lambda: [0, 0])
    for p in per: s = p[4].split("/")[0]; src[s][0] += 1; src[s][1] += p[0]
    xs[name]["by_source"] = {k: {"docs": v[0], "capped_gen_rows": int(v[1])} for k, v in sorted(src.items())}
res["xcorp"] = xs
ext_x = sum(e["rows"] for e in ext)
res["extension"] = {"calib_docs": len(ext_calib), "calib_capped_rows": int(capped[ext_calib].sum()), "xcorp_docs": len(ext), "xcorp_capped_rows": int(ext_x)}
json.dump(res, open(f"{out}/counts.json", "w"), indent=1)
json.dump({"calib_docs": [{"doc": int(i), "pool": pools[i], "tokens": int(min(CAP, docs[i]["tokens"])), "rows": int(capped[i])} for i in ext_calib], "chunks": ext, "w214_base_chunks": sorted(W14)}, open(f"{out}/ext-docs.json", "w"))
print(json.dumps(res, indent=1))
