#!/usr/bin/env python3
"""dktab150.py RESDIR [M] -- dkm.sh's CSVs: per rate and row (tag), the median us over rounds of each family at M rows
(default 1), the 48 GDN : 16 attention layer (GDN: gdn_qkv, gdn_z, ssm_out, gate_up, down; attention: attn_qg, attn_k,
attn_v, attn_o, gate_up, down), the head, and the layer's code read rate (GB/s of trellis bytes). (w150, #177-02)"""
import csv, glob, os, re, statistics as st, sys
from collections import defaultdict
M = int(sys.argv[2]) if len(sys.argv) > 2 else 1
D = defaultdict(lambda: defaultdict(list))   # (rate, tag) -> shape -> [us]
NK = {}
for f in glob.glob(os.path.join(sys.argv[1], "r*-*-k*.csv")):
    m = re.match(r"r(\d+)-(.+)-k([\d.]+)\.csv$", os.path.basename(f))
    _, tag, k = m.groups()
    for r in csv.DictReader(open(f)):
        if int(r["M"]) != M or int(r["rc"]): continue
        D[(float(k), tag)][r["shape"]].append(float(r["us_median"])); NK[r["shape"]] = (int(r["N"]), int(r["K"]))
FAM = ["gate_up", "down", "gdn_qkv", "gdn_z", "attn_qg", "attn_k", "attn_v", "attn_o", "ssm_out"]
GDN = ["gdn_qkv", "gdn_z", "ssm_out", "gate_up", "down"]; ATT = ["attn_qg", "attn_k", "attn_v", "attn_o", "gate_up", "down"]
print(f"M {M}: us (median over rounds); layer = 48 GDN : 16 attention mean; GB/s = the layer's trellis bytes / time")
print(f"{'K':>6} {'row':22s} " + " ".join(f"{s[:7]:>7s}" for s in FAM) + f" {'layer':>7s} {'head':>7s} {'GB/s':>6s}")
for (k, tag) in sorted(D):
    d = D[(k, tag)]; md = {s: st.median(v) for s, v in d.items()}
    if not all(s in md for s in FAM): continue
    lay = (48 * sum(md[s] for s in GDN) + 16 * sum(md[s] for s in ATT)) / 64
    byt = lambda ss: sum(NK[s][0] * NK[s][1] * k / 8 for s in ss)
    gbs = (48 * byt(GDN) + 16 * byt(ATT)) / 64 / (lay * 1e3)
    print(f"{k:6g} {tag:22s} " + " ".join(f"{md[s]:7.1f}" for s in FAM) + f" {lay:7.1f} {md.get('head', float('nan')):7.1f} {gbs:6.0f}")
