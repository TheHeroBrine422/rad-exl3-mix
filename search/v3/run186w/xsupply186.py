# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""xsupply186.py -- how many xcorp pieces are still unused, and how many of them are non-Latin (RULES-strata §3, #186).

Run on ds8 (CPU, nice; decodes the pieces' tokens with the bf16 tokenizer). For every xcorp batch that has a golden
(kld.json) and a corpus.meta.jsonl, labels each piece (stratum, script; strata186.label_text), marks it used when its
xcorp document is in an earlier x part (every golden-s17*-x, t183-x, final-x, w129's excl files: the same set pickx.py
--exclude-orig reads), and writes one JSON line per piece to OUT.jsonl (numbers, source names, category; no text).
xc8k-b1 / tr32k-b1 have no corpus.meta.jsonl: their used pieces are read from the search27 x parts' orig indices
(the piece index in b1, as w129 built them) and reported separately.
python3 -I xsupply186.py TOKENIZER OUT.jsonl X_DIR GOLDEN_X [GOLDEN_X ...]"""
import glob, json, os, sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import strata186 as ST  # noqa: E402

tk = AutoTokenizer.from_pretrained(sys.argv[1]); OUT, XD = sys.argv[2], sys.argv[3]; GX = sys.argv[4:]
used = set()
for g in GX:                                    # a golden dir, or a kld.json (w129's excl files)
    for d in json.load(open(g if g.endswith(".json") else f"{g}/kld.json"))["docs"]: used.add(d.get("xdoc", d.get("orig")))
rows = []
for f in sorted(glob.glob(f"{XD}/*/corpus.meta.jsonl")):
    b = os.path.basename(os.path.dirname(f))
    M = [json.loads(l) for l in open(f)]
    if not os.path.exists(f"{XD}/{b}/kld.json"):
        print(f"{b}: no golden (kld.json); {len(M)} pieces listed only", file=sys.stderr); continue
    m = json.load(open(f"{XD}/{b}/kld.json")); assert len(m["docs"]) == len(M), b
    tok = np.memmap(f"{XD}/{b}/tokens.i32", np.int32, "r"); off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
    for i, (x, d) in enumerate(zip(M, m["docs"])):
        assert x["tokens"] == d["tokens"] and x["source"] == d["source"], (b, i)
        s, sc, sh = ST.label_text(tk.decode(np.asarray(tok[off[i]:off[i + 1]])))
        rows.append({"batch": b, "idx": i, "doc": x["doc"], "source": x["source"], "category": x["category"], "tokens": x["tokens"],
                     "score_from": d["score_from"], "stratum": s, "script": sc, "nonlatin": round(sh, 4), "used": x["doc"] in used})
with open(OUT + ".part", "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
os.replace(OUT + ".part", OUT)
for b in sorted({r["batch"] for r in rows}):
    R = [r for r in rows if r["batch"] == b]; U = [r for r in R if not r["used"]]
    print(f"{b}: {len(R)} pieces, unused {len(U)}; unused N {sum(r['stratum'] == 'N' for r in U)} "
          f"(scripts {dict(sorted({s: sum(1 for r in U if r['script'] == s) for s in {r['script'] for r in U if r['stratum'] == 'N'}}.items(), key=lambda kv: -kv[1]))})")
