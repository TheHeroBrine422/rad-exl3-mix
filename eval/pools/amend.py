#!/usr/bin/env python3
"""amend.py POOLS.json GOLDEN_CALIB_DIR OUT_AMEND.json OUT_POOLS.json -- pool amendment 2026-11-01 (user decision, option 3).

Why: every verify document longer than 16k tokens and every agentic / long-agentic / long verify document is spent
(provenance-2026-11-01.json), so a clean ver3 / ver4 could not judge agentic or long-context text. QAD (V4) has not
started; the only use of qad documents so far: YAQA collections (dropped) read the first NTOUCH windows of the
shuffled qad window list (collect.py, seed 0, 2048-token windows), and the head-sensitivity Fisher (headsens.npz,
qad pool sums) -- a per-vocab-row diagonal over 5.7M tokens, which a document cannot be recovered from.
Rule: move about half of each scarce qad category into verify, largest first for long-agentic (one long trace per
ver slice), keeping the other half for QAD. YAQA's reads do not exclude a document (it is dropped: no shipped quant
used its Hessians; all gates failed) but are recorded per moved doc; every LATER qad collection must read the amended
pools (collect.py YAQA_POOLS) so the moved documents never feed a quant again. pools.json stays
untouched (read-only); OUT_POOLS.json is the amended copy (POOLS=... for calib_slice.py), OUT_AMEND.json the record."""
import json, sys
import numpy as np

POOLS, G, OA, OP = sys.argv[1:5]
PROV = sys.argv[5] if len(sys.argv) > 5 else None    # provenance.json: documents any eval slice used are not clean
NTOUCH, L, SEED = 200, 2048, 0
P = json.load(open(POOLS)); pool = list(P["pool"])
meta = json.load(open(f"{G}/kld.json")); docs = meta["docs"]
cal = [json.loads(l) for l in open(f"{G}/../../corpus/calib.jsonl")]
off = np.cumsum([0] + [d["tokens"] for d in docs])
wins = []; owner = {}
for i, d in enumerate(docs):
    if pool[i] != "qad": continue
    for a in range(0, d["tokens"] - L + 1, L): wins.append(int(off[i] + a)); owner[int(off[i] + a)] = i
rng = np.random.default_rng(SEED); rng.shuffle(wins)
touched = {owner[w] for w in wins[:NTOUCH]}
spent = {int(k) for k in json.load(open(PROV))["doc_use"]} if PROV else set()
moved = {}
# long-agentic traces were only ever scored on a prefix (8k / 16k, calA/fit/calB slices): their tails are unscored.
# Moved with score_from = the used prefix (ver slices score only positions past it; the prefix is context).
TRACES = {2300: 8192, 4723: 16384}       # 1772 (long64: 64k used) and 2651 stay in qad
for i, sf in TRACES.items():
    assert pool[i] == "qad" and cal[i]["category"] == "long-agentic"
    pool[i] = "verify"; moved[i] = ("long-agentic", int(cal[i]["tokens"]), i in touched)
for cat, take in (("agentic", 7), ("long", 2)):
    cand = sorted((i for i, p in enumerate(pool) if p == "qad" and cal[i]["category"] == cat and i not in spent),
                  key=lambda i: -int(cal[i]["tokens"]))
    pick = cand[:take] if cat != "agentic" else cand[::2][:take]     # agentic: every other by size, a spread
    for i in pick: pool[i] = "verify"; moved[i] = (cat, int(cal[i]["tokens"]), i in touched)
rec = {"date": "2026-11-01", "reason": "verify pool had no unspent agentic / long / >16k documents (provenance)",
       "from": "qad", "to": "verify", "yaqa_touched_qad_docs": sorted(touched),
       "moved": {str(i): {"category": c, "tokens": t, "yaqa_read": r, **({"score_from": TRACES[i]} if i in TRACES else {})}
              for i, (c, t, r) in sorted(moved.items())},
       "caveat": "the moved docs were in the qad-pool head Fisher (headsens.npz) used to rank the slab head's rows"}
json.dump(rec, open(OA, "w"), indent=1)
json.dump({**P, "pool": pool, "amended": "amend-2026-11-01.json"}, open(OP, "w"))
print(f"YAQA touched {len(touched)} qad docs; moved {len(moved)} to verify, {sum(t for _, t, _ in moved.values())} tokens:")
for i, (c, t, r) in sorted(moved.items(), key=lambda x: x[1]): print(f"  doc {i:5d} {c:13s} {t:7d} tok  yaqa-read {r}")
