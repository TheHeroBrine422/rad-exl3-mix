#!/usr/bin/env python3
"""mkhalf.py SNAPROOT OUT [EST] -- split the 27B H_out collection's cumulative snapshots (SNAPROOT/hdiag-snapN-<est>,
running means over N sequences) into independent 128-sequence blocks; then
  OUT/train-N/  the mean over the EVEN-numbered blocks up to snapshot N (≈ N/2 sequences): what a quant would use
  OUT/test/     the mean over ALL odd-numbered blocks available (independent of every train-N): the reference
as ourq --hout dirs ("hout_diag"). EST: dy (default, Σdy²), xw, or "" (Sketch B). MKHALF_NORM=1: each block scaled to the
same per-tensor total before averaging (robust to the heavy-tailed outlier sequences, 2026-11-04)."""
import glob, os, re, sys, json
import torch
from safetensors import safe_open
from safetensors.torch import save_file
R, O = sys.argv[1], sys.argv[2]; NORM = os.environ.get("MKHALF_NORM") == "1"; est = sys.argv[3] if len(sys.argv) > 3 else "dy"; suf = f"-{est}" if est else ""
ns = sorted(int(m.group(1)) for p in glob.glob(f"{R}/hdiag-snap*{suf}") if (m := re.search(rf"hdiag-snap(\d+){re.escape(suf)}$", p)))
def load(n):
    out = {}
    for f in glob.glob(f"{R}/hdiag-snap{n}{suf}/*.safetensors"):
        with safe_open(f, "pt") as h: out[os.path.basename(f)] = h.get_tensor("hout_diag").double()
    return out
def save(d, acc, cnt, meta):
    os.makedirs(d, exist_ok=True)
    for k, v in acc.items(): save_file({"hout_diag": (v / cnt).float().contiguous()}, f"{d}/{k}", metadata=meta)
prev, pn, ev, od, ne, no = {}, 0, {}, {}, 0, 0
for i, n in enumerate(ns):
    cur = load(n); blk = {k: v * n - prev.get(k, 0) * pn for k, v in cur.items()}; m = n - pn
    if NORM: blk = {k: v / v.sum().clamp_min(1e-300) * m for k, v in blk.items()}   # every block the same total per tensor
    tgt = ev if i % 2 == 0 else od
    for k, b in blk.items(): tgt[k] = tgt.get(k, 0) + b
    if i % 2 == 0: ne += m
    else: no += m
    if os.path.exists(f"{O}/train-{n}/.done"): pass
    else:
        save(f"{O}/train-{n}", ev, ne, {"seqs": str(ne), "half": "even", "upto": str(n), "est": est}); open(f"{O}/train-{n}/.done", "w").close()
    prev, pn = cur, n
if no: save(f"{O}/test", od, no, {"seqs": str(no), "half": "odd", "upto": str(ns[-1]), "est": est})
json.dump({"snaps": ns, "train_seqs_last": ne, "test_seqs": no}, open(f"{O}/mkhalf.json", "w"))
print(f"snapshots {ns}: train (even blocks) up to {ne} seqs, test (odd blocks) {no} seqs")
