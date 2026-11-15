# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""supply195.py XCORP_META TOKENIZER OUT.jsonl USED_NAME=META.jsonl[,META.jsonl...] ... -- the non-Latin supply of
xcorp v1 (RULES-slices195 §1, #195). Run on colo (CPU, nice). For every xcorp v1 document (XCORP_META = kld.json +
tokens.i32 of xcorp/tokenize.py): the label of its first <= 8192 tokens (piece 0, as kldcorpus.py cuts xc8k pieces;
strata186.label_text), the gen rows of that piece under the board's mask (genmask181.doc_gen) and, per USED_NAME, whether
an earlier corpus (its .meta.jsonl "doc" field = the xcorp document index) holds the document. One JSON line per
non-Latin document (numbers and source names only) + a summary on stdout."""
import collections, json, os, sys

import numpy as np
from transformers import AutoTokenizer

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..")); sys.path.insert(0, os.path.join(H, "..", "..", "..", "v3", "qad181")); sys.path.insert(0, H)
import strata186 as ST  # noqa: E402
import genmask181 as GM  # noqa: E402

CHUNK = 8192
meta, tkp, out = sys.argv[1:4]
tk = AutoTokenizer.from_pretrained(tkp); S, E, A, NL = GM.ids(tk)
used = {}
for a in sys.argv[4:]:
    n, fs = a.split("="); used[n] = {json.loads(l)["doc"] for f in fs.split(",") for l in open(f)}
m = json.load(open(f"{meta}/kld.json")); docs = m["docs"]
T = np.memmap(f"{meta}/tokens.i32", np.int32, "r"); off = np.concatenate([[0], np.cumsum([d["tokens"] for d in docs])])
rows = []
for i, d in enumerate(docs):
    t = np.asarray(T[off[i]:min(off[i + 1], off[i] + CHUNK)])
    if len(t) < 64: continue
    s, sc, sh = ST.label_text(tk.decode(t))
    if s != "N": continue
    w = GM.doc_gen(t, S, E, A, NL)
    rows.append({"doc": i, "source": d["source"], "category": d.get("category", ""), "tokens": int(len(t)), "doc_tokens": d["tokens"],
                 "chat": bool((t == S).any()), "gen": int(w[1:].sum()), "script": sc, "nonlatin": round(sh, 4),
                 "used": [n for n, u in used.items() if i in u]})
with open(out + ".part", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
os.replace(out + ".part", out)
F = [r for r in rows if not r["used"]]
def tab(R, key):
    c = collections.defaultdict(lambda: [0, 0, 0])
    for r in R: c[r[key]][0] += 1; c[r[key]][1] += r["tokens"]; c[r[key]][2] += r["gen"]
    return {k: {"pieces": v[0], "tokens": v[1], "gen": v[2]} for k, v in sorted(c.items(), key=lambda kv: -kv[1][0])}
summ = {"docs": len(docs), "N": len(rows), "N_used_by": {n: sum(n in r["used"] for r in rows) for n in used}, "N_free": len(F),
        "free_tokens": sum(r["tokens"] for r in F), "free_gen": sum(r["gen"] for r in F),
        "free_gen_lt_256": sum(r["gen"] < 256 for r in F), "free_by_script": tab(F, "script"), "free_by_source": tab(F, "source"),
        "free_by_category": tab(F, "category"), "all_N_by_category": tab(rows, "category")}
print(json.dumps(summ, indent=1))
