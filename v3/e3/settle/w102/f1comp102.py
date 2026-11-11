#!/usr/bin/env python3
"""f1comp102.py SNAPROOT GOLDEN POOLS CORPUS COLLECTLOG OUT.json -- w102 #136-05 (../w102.md): category composition of
the 27B Sigma-dy^2 collection's 128-sequence blocks (bn1920's blocks).
  - window order: collect.py's (qad-pool windows of 2048 tokens of the calib golden's token stream, in document order,
    shuffled by numpy default_rng(0); sequence s uses window s mod #windows; YAQA_SHORT / GENMASK off as hdiag27.sh)
  - category: CORPUS line's "category" field if present, else race.py's source -> category map
  - block energy: per tensor, the block's Sigma-dy^2 total per sequence (snapshots differenced as mkbn.py) / the median
    block's; per layer the geometric mean over the layer's tensors; layers 0 / 10 / 39 / 50 / 63 and all-layer
  - loss/tok per sequence from collect.log ("seq N: ... loss/tok X"; the last line of a sequence wins)
gpubox CPU, niced; a few seconds per snapshot set."""
import glob, json, math, os, re, sys
from collections import Counter

import numpy as np
import torch
from safetensors import safe_open

SNAP, G, POOLS, CORPUS, CLOG, OUT = sys.argv[1:7]
L, SEED, NSEQ, B = 2048, 0, 1920, 128
AGENT = {"agentic", "tools", "long-agentic"}
SRC2CAT = {"hermes-fc": "tools", "glaive-fc-v2": "tools", "when2call": "tools", "wildchat": "chat", "smoltalk2": "chat",
           "stackv2-edu": "code", "the-stack-smol-xl": "code", "wikimedia": "encyclopedic", "fineweb-edu": "web",
           "fineweb2-hq": "multilingual", "wikipedia": "multilingual", "stackexchange": "qa", "finemath-4plus": "math",
           "libretexts": "math", "openthoughts3": "reasoning", "doab": "books", "pg19": "books", "gutenberg": "books",
           "pubmed": "science", "arxiv": "science", "peS2o": "science", "caselaw": "legal", "swe-smith": "agentic",
           "nemotron-swe": "agentic", "swe-rebench-openhands": "agentic", "agent-trace": "long-agentic",
           "pg19-long": "long", "arxiv-long": "long"}

meta = json.load(open(f"{G}/kld.json")); docs = meta["docs"]
pools = json.load(open(POOLS))["pool"]
off = np.cumsum([0] + [d["tokens"] for d in docs])
wins = []
for i, d in enumerate(docs):
    if pools[i] != "qad": continue
    for a in range(0, d["tokens"] - L + 1, L): wins.append(int(off[i] + a))
np.random.default_rng(SEED).shuffle(wins)
assert len(wins) == 1350, len(wins)                       # collect.log: "1350 qad windows of 2048 tokens"
cat_line = []
with open(CORPUS) as f:
    for ln in f:
        if ln.strip(): cat_line.append(json.loads(ln).get("category"))
assert len(cat_line) == len(docs), (len(cat_line), len(docs))
cat = [c if c else SRC2CAT.get(d["source"].split("/")[0], "other") for c, d in zip(cat_line, docs)]
src = [d["source"] for d in docs]
seq_doc = [int(np.searchsorted(off, wins[s % len(wins)], side="right") - 1) for s in range(NSEQ)]
loss = {}
for ln in open(CLOG):
    m = re.match(r"seq (\d+):.*loss/tok ([0-9.eE+-]+)", ln)
    if m: loss[int(m[1])] = float(m[2])

# collection-wide shares (by window = equal tokens)
allc = Counter(cat[d] for d in seq_doc)
mean_agent = sum(v for k, v in allc.items() if k in AGENT) / NSEQ

# block energies (mkbn.py's differencing)
ns = sorted(int(m.group(1)) for p in glob.glob(f"{SNAP}/hdiag-snap*-dy") if (m := re.search(r"hdiag-snap(\d+)-dy$", p)))
ns = [n for n in ns if n <= NSEQ]; assert ns[-1] == NSEQ and len(ns) == NSEQ // B, ns
keys = sorted(os.path.basename(f) for f in glob.glob(f"{SNAP}/hdiag-snap{NSEQ}-dy/*.safetensors"))
ratio = {}
for k in keys:
    prev, pn, tot = None, 0, []
    for n in ns:
        with safe_open(f"{SNAP}/hdiag-snap{n}-dy/{k}", "pt") as h: cur = h.get_tensor("hout_diag").double()
        blk = cur * n - (prev * pn if prev is not None else 0); tot.append(float(blk.sum()) / (n - pn)); prev, pn = cur, n
    med = sorted(tot)[len(tot) // 2]; ratio[k] = [t / med for t in tot]
layer_of = lambda k: int(re.search(r"layers\.(\d+)\.", k)[1]) if "layers." in k else -1
def lay_ratio(ls):
    ks = [k for k in keys if layer_of(k) in ls]
    return [math.exp(sum(math.log(ratio[k][b]) for k in ks) / len(ks)) for b in range(len(ns))], len(ks)

out = {"windows": len(wins), "sequences": NSEQ, "block_seqs": B, "mean_agent_share": mean_agent,
       "collection_shares": {k: v / NSEQ for k, v in allc.most_common()}, "blocks": []}
lr = {str(l): lay_ratio([l]) for l in (0, 10, 39, 50, 63)}
lr["all"] = lay_ratio(list(range(64)))
out["energy_ratio_layers"] = {k: {"ratio": [round(x, 3) for x in v[0]], "tensors": v[1]} for k, v in lr.items()}
for b in range(len(ns)):
    ss = range(b * B, (b + 1) * B)
    c = Counter(cat[seq_doc[s]] for s in ss)
    ag = sum(v for k, v in c.items() if k in AGENT) / B
    top = sorted(ss, key=lambda s: -loss.get(s, 0))[:5]
    out["blocks"].append({"block": b, "seqs": [b * B, (b + 1) * B - 1], "agent_share": ag, "agent_x_mean": ag / mean_agent,
                          "shares": {k: round(v / B, 3) for k, v in c.most_common()},
                          "energy_l0": lr["0"][0][b], "energy_l10": lr["10"][0][b], "energy_all": lr["all"][0][b],
                          "loss_mean": float(np.mean([loss[s] for s in ss if s in loss])) if any(s in loss for s in ss) else None,
                          "distinct_docs": len({seq_doc[s] for s in ss}),
                          "top_loss_seqs": [{"seq": s, "loss": loss.get(s), "doc": seq_doc[s], "cat": cat[seq_doc[s]], "source": src[seq_doc[s]]} for s in top]})
json.dump(out, open(OUT, "w"), indent=1)
print(f"mean agent share {mean_agent:.3f}; collection {out['collection_shares']}")
print("block  agent  x-mean  E(l0)  E(l10)  E(all)  loss  top category")
for o in out["blocks"]:
    print(f"{o['block']:5d} {o['agent_share']:6.3f} {o['agent_x_mean']:6.2f} {o['energy_l0']:6.2f} {o['energy_l10']:6.2f} {o['energy_all']:6.2f} "
          f"{(o['loss_mean'] or 0):5.2f}  {list(o['shares'].items())[:3]}")
