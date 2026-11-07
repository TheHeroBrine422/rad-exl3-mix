#!/usr/bin/env python3
"""plan71.py [colo|thor] > jobs.txt -- w71's job lists (../w71.md, #113): lines "LAYER TENSOR K SEED GROUP ARMS" for
q71.py. Amended 2026-11-07 4:20 AM CST (colo alone was too slow: reported arms trimmed) and 6:50 AM CST (the lead
added Thor: the trimmed arms are back, split by host; the two lists are disjoint).
colo: the gate (S16 / F16 / T16 on attn_qg 3 / 19 / 43, sample 0, K2-4); K2-4 deciding groups g1 + g3, both samples
      (ffn_down first within each K: the long pole); then backfill: g5 (iid-table ceiling) sample 1, h3 (V20 at K5 / K6).
thor: g2 (B16 / B18-4096: gate 3), both samples; g4 (V20) sample 0; K5 / K6 (h1 + h2) sample 0; g5 sample 0; g4
      sample 1; K5 / K6 sample 1."""
import sys
TENSORS = [("ffn_down", (2, 30, 63)), ("ssm_qkv", (2, 30, 50)), ("ssm_z", (10, 34, 62)), ("ssm_out", (10, 30, 50)),
           ("attn_qg", (3, 19, 43)), ("attn_k", (11, 35, 63)), ("attn_v", (7, 43, 59)),
           ("attn_output", (19, 43, 63)), ("ffn_gate_up", (10, 43, 62))]
LO = {"g1": "V16,V18,B16-256,B16-1024,B18-1024,B20-256,B20-1024", "g3": "B20-4096", "g2": "B16-4096,B18-4096",
      "g4": "V20", "g5": "V20r,B20-4096r"}
HI = {"h1": "V16,V18,B16-4096,B20-1024", "h2": "B20-4096", "h3": "V20"}
host = sys.argv[1] if len(sys.argv) > 1 else "colo"
out = []
def add(Ks, seeds, grps, arms):
    for K in Ks:
        for w, layers in TENSORS:
            for L in layers:
                for s in seeds:
                    for g in grps: out.append(f"{L} {w} {K} {s} {g} {arms[g]}")
if host == "colo":
    for L in (3, 19, 43):
        for K in (2, 3, 4): out.append(f"{L} attn_qg {K} 0 gate S16,F16,T16")
    add((2, 3, 4), (0, 1), ("g1", "g3"), LO)
    add((2, 3, 4), (1,), ("g5",), LO)
    add((5, 6), (0, 1), ("h3",), HI)
else:
    add((2, 3, 4), (0, 1), ("g2",), LO)
    add((2, 3, 4), (0,), ("g4",), LO)
    add((5, 6), (0,), ("h1", "h2"), HI)
    add((2, 3, 4), (0,), ("g5",), LO)
    add((2, 3, 4), (1,), ("g4",), LO)
    add((5, 6), (1,), ("h1", "h2"), HI)
print("\n".join(out))
