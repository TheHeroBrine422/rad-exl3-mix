#!/usr/bin/env python3
# paired_boot.py ON_ROWS OFF_ROWS GOLDEN_DIR [B] -- w178 (#194-02): the relative KL change of OFF (act_two=0) against ON
# (the container's act_two=1) over the golden's gen rows, paired per document: each bootstrap resample uses the same documents
# for both arms. delta_pct = (mean_off / mean_on - 1) * 100, with its 95 % per-document bootstrap interval (B resamples,
# default 2000, seed 194, the same document rows as kl_boot.py). Prints one JSON line.
import json, sys
import numpy as np

on_f, off_f, gold = sys.argv[1], sys.argv[2], sys.argv[3]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
m = json.load(open(gold + "/kld.json")); n = m["rows"]
gen = np.fromfile(gold + "/gen.u8", np.uint8).astype(bool)
assert gen.size == n


def load(p):
    kl = np.fromfile(p, np.float32).reshape(n, -1)[:, 0].astype(np.float64)
    return np.where(gen, kl, 0.0)


on, off = load(on_f), load(off_f)
S_on, S_off, C = [], [], []
for d in m["docs"]:
    k = d["tokens"] - 1 - d["score_from"]
    seg = slice(d["row0"], d["row0"] + k)
    S_on.append(on[seg].sum()); S_off.append(off[seg].sum()); C.append(gen[seg].sum())
S_on, S_off, C = (np.array(x, dtype=np.float64) for x in (S_on, S_off, C))
mon, moff = S_on.sum() / C.sum(), S_off.sum() / C.sum()
rng = np.random.default_rng(194)
idx = rng.integers(0, len(C), size=(B, len(C)))
ratio = (S_off[idx].sum(1) / C[idx].sum(1)) / (S_on[idx].sum(1) / C[idx].sum(1))
lo, hi = np.percentile(ratio, [2.5, 97.5])
print(json.dumps({"on_rows": on_f, "off_rows": off_f, "golden": gold, "docs": len(C), "gen_positions": int(C.sum()),
                  "kl_on": float(mon), "kl_off": float(moff), "delta_pct": float((moff / mon - 1) * 100),
                  "delta_pct_ci95_doc_boot": [float((lo - 1) * 100), float((hi - 1) * 100)], "B": B, "seed": 194}))
