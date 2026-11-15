# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""refs186.py T183_GOLDEN T183_ROWS_DIR T186N_GOLDEN T186N_ROWS_DIR REFS_T183.jsonl TOKENIZER OUT.jsonl -- the non-Latin
supplement's reference readings (RULES-strata §7, #186, w164). For each reference container read on both goldens:
  L    its KL on t183's L documents (gen positions of the documents labelled L; the rest of t183 is N)
  N183 its KL on t183's N documents (what t183 holds of the non-Latin text)
  N    its KL on t186n (every document is N)
  board 0.961 L + 0.039 N (RULES-strata §2: the calib gen-row weights; arithmetic, as the board's KLD is a mean over gen rows)
and the t183 full KL recomputed from the rows (the lead's refs-t183.jsonl value is the check). Rows files: float32, first
column = per-position KL, one row per golden row (radiance --kld-out's .json.rows). Labels: a calib document's label is its
calib label (strata186 on the calib golden: calib-kld-strata.json); an xcorp piece's label comes from its tokens.
Usage on ds8 (CPU): python3 -I refs186.py ... (see the call in README.md, run186w)."""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import strata186 as ST  # noqa: E402

T183G, T183R, T186G, T186R, REFS, TOK, OUT = sys.argv[1:8]
CAL = os.environ.get("CALIB_STRATA", "/data/q27/w164/strata/calib-kld-strata.json")
W = 0.961, 0.039                                                 # RULES-strata §2: (L, N)


def rows_of(path, n):
    a = np.fromfile(path, np.float32)
    return a.reshape(n, -1)[:, 0]


def golden(gdir):
    m = json.load(open(f"{gdir}/kld.json")); gen = np.fromfile(f"{gdir}/gen.u8", np.uint8).astype(bool)
    return m, gen


# ---- strata per row of t183 (c documents: calib label by orig; x documents: from their tokens)
m, gen = golden(T183G)
calib = json.load(open(CAL))["docs"]
lab = {}
xs = [i for i, d in enumerate(m["docs"]) if d.get("part") != "c"]
if xs:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(TOK)
    tok = np.memmap(f"{T183G}/tokens.i32", np.int32, "r"); off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
    for i in xs: lab[i] = ST.label_text(tk.decode(np.asarray(tok[off[i]:off[i + 1]])))[0]
row_N = np.zeros(m["rows"], bool)
for i, d in enumerate(m["docs"]):
    s = calib[d["orig"]]["stratum"] if d.get("part") == "c" else lab[i]
    n = d["tokens"] - 1 - d["score_from"]
    row_N[d["row0"]:d["row0"] + n] = s == "N"
Lm, Nm = gen & ~row_N, gen & row_N
# ---- t186n: every document N
m2, gen2 = golden(T186G)
assert all(d.get("stratum") == "N" for d in m2["docs"]), "t186n: every document must be N"
bpw = {r["name"]: r["bpw"] for r in map(json.loads, open(REFS))}
# a reference without a bpw in REFS: an IQR-V4 name takes the board value of w134's rows (as refs183t.sh does; the HF
# files carry the vision tower and drafter, so their sizes are not the board's bpw)
IQR_BPW = os.environ.get("IQR_BPW_JSONL", "/workspace/rad-local/workers/w134/refs134.jsonl")
if os.path.exists(IQR_BPW):
    for r in map(json.loads, open(IQR_BPW)):
        if r["tag"].startswith("ref-IQR4-"): bpw.setdefault(r["tag"][len("ref-IQR4-"):], r["bpw"])
out = []
names = sorted({f[:-len("-t183.json.rows")] for f in os.listdir(T183R) if f.endswith("-t183.json.rows")})
for n in names:
    a = rows_of(f"{T183R}/{n}-t183.json.rows", m["rows"])
    rec = {"name": n, "t183_full": float(a[gen].mean()), "t183_L": float(a[Lm].mean()), "t183_N": float(a[Nm].mean()) if Nm.any() else None,
           "L_positions": int(Lm.sum()), "N183_positions": int(Nm.sum())}
    rec["bpw"] = bpw.get(n)
    p = f"{T186R}/{n}-t186n.json.rows"
    if os.path.exists(p):
        b = rows_of(p, m2["rows"]); rec["t186n_N"] = float(b[gen2].mean()); rec["N_positions"] = int(gen2.sum())
        rec["board"] = W[0] * rec["t183_L"] + W[1] * rec["t186n_N"]
    out.append(rec)
with open(OUT + ".part", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
os.replace(OUT + ".part", OUT)
print(f"{OUT}: {len(out)} references; t183 L positions {int(Lm.sum())}, N {int(Nm.sum())}; t186n N positions {int(gen2.sum())}")
