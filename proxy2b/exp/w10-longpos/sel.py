#!/usr/bin/env python3
"""sel.py CALIB_KLD.json POOLS.json GEN.u8 [CAP] -- the long 0.8B golden's documents (w10-longpos), printed in
mkgold.py's DOCIDX syntax (idx:cap): every SEARCH-pool document (pools-v3) of >= 8192 tokens, cut at CAP tokens
(default 65536; the two 245k agent traces are the only ones cut). Search pool only: disjoint from the qad windows the
diagonals were collected on and from the fit-pool calibration choices. Then a table to stderr: gen rows by band."""
import json, sys
import numpy as np
kj, pj, gj = sys.argv[1:4]; CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 65536
m = json.load(open(kj)); pool = json.load(open(pj))["pool"]; gen = np.fromfile(gj, np.uint8)
E = [0, 2048, 8192, 32768, 65536]; tot = np.zeros(4, int); out = []
for i, d in enumerate(m["docs"]):
    if pool[i] != "search" or d["tokens"] < 8192: continue
    nt = min(d["tokens"], CAP); n = nt - 1 - d["score_from"]
    g = gen[d["row0"]:d["row0"] + n].astype(bool); p = d["score_from"] + np.arange(n)
    c = [int(g[(p >= E[k]) & (p < E[k + 1])].sum()) for k in range(4)]; tot += c
    out.append(f"{i}:{CAP}" if d["tokens"] > CAP else str(i))
    print(f"{i:5d} {d['source'][:24]:24s} {nt:6d} tokens, gen by band {c}", file=sys.stderr)
print(f"{len(out)} docs; gen rows by band 0-2k/2-8k/8-32k/32-64k {tot.tolist()} = {tot.sum()}", file=sys.stderr)
print(" ".join(out))
