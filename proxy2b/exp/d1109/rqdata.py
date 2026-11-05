"""rqdata.py -- calibration windows for d1109 (REAL-Q): qad-pool documents of the calib meta golden cut into SEQLEN-token
windows, shuffled with numpy default_rng(SEED) -- the same windows and order as v3/yaqa/collect.py (c18 / c23 use
seed 0, 2048 tokens) -- and the gen mask of each window (GPOS[t] = the prediction made at token t is scored)."""
import json
import numpy as np

META = "/data/challenge/calib-meta"
POOLS = "/workspace/rad-exl3-mix/eval/pools/pools-v3.json"


def windows(L=2048, seed=0, meta=META, pools=POOLS):
    """-> (tokens int32 memmap, GPOS float32 [tokens], shuffled window starts)"""
    m = json.load(open(f"{meta}/kld.json"))
    toks = np.fromfile(f"{meta}/tokens.i32", np.int32)
    pool = json.load(open(pools))["pool"]
    off = np.cumsum([0] + [d["tokens"] for d in m["docs"]])
    assert off[-1] == len(toks) and len(pool) == len(m["docs"]), (off[-1], len(toks), len(pool))
    gen = np.fromfile(f"{meta}/gen.u8", np.uint8); assert len(gen) == m["rows"]
    gpos = np.zeros(len(toks), np.float32)
    for i, d in enumerate(m["docs"]):
        n = d["tokens"] - 1 - d["score_from"]
        gpos[off[i] + d["score_from"]:off[i] + d["tokens"] - 1] = gen[d["row0"]:d["row0"] + n]
    wins = []
    for i, d in enumerate(m["docs"]):
        if pool[i] != "qad": continue
        for a in range(0, d["tokens"] - L + 1, L): wins.append(int(off[i] + a))
    np.random.default_rng(seed).shuffle(wins)
    return toks, gpos, wins
