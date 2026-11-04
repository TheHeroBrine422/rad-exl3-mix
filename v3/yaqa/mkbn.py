#!/usr/bin/env python3
"""mkbn.py SNAPROOT N OUT -- the block-normalized Σdy² diagonal of the 27B collection over its first N sequences:
the cumulative snapshots SNAPROOT/hdiag-snap<n>-dy (running means over n sequences) are differenced into
128-sequence blocks, each block scaled to the same per-tensor total (its share of the sequences), and averaged
(#33-27: robust to the few blocks that carry 4-27x the median energy in early layers). The result is a running mean
on the same scale as the plain snapshot (its per-tensor total equals the plain mean's average block total), written as
ourq --hout files (OUT/<key>.safetensors, tensor "hout_diag"). Also prints, per layer range, the ratio of the
largest block's raw total to the median block's (the tail it removed)."""
import glob, os, re, sys, json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

R, N, O = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ns = sorted(int(m.group(1)) for p in glob.glob(f"{R}/hdiag-snap*-dy") if (m := re.search(r"hdiag-snap(\d+)-dy$", p)))
ns = [n for n in ns if n <= N]
assert ns and ns[-1] == N, f"no snapshot at {N} (have {ns[-3:]})"
keys = sorted(os.path.basename(f) for f in glob.glob(f"{R}/hdiag-snap{N}-dy/*.safetensors"))
assert len(keys) >= 400, f"only {len(keys)} tensors in the {N} snapshot"
os.makedirs(O, exist_ok=True)
tail = {}
for k in keys:
    prev, pn, acc, totals, ws = None, 0, 0, [], []
    for n in ns:
        with safe_open(f"{R}/hdiag-snap{n}-dy/{k}", "pt") as h: cur = h.get_tensor("hout_diag").double()
        m = n - pn
        blk = cur * n - (prev * pn if prev is not None else 0)          # sum over the block's m sequences
        totals.append(float(blk.sum()) / m); ws.append(m)
        acc = acc + blk / blk.sum().clamp_min(1e-300) * m                # each sequence of the block: equal share
        prev, pn = cur, n
    raw_mean_total = sum(t * w for t, w in zip(totals, ws)) / sum(ws)      # plain mean's per-sequence total
    out = (acc / N) * raw_mean_total                                      # same overall scale as the plain mean
    save_file({"hout_diag": out.float().contiguous()}, f"{O}/{k}",
              metadata={"seqs": str(N), "est": "dy", "blocknorm": "1", "blocks": str(len(ns))})
    t = sorted(totals); tail[k] = max(totals) / t[len(t) // 2]
json.dump({"seqs": N, "blocks": len(ns), "tensors": len(keys)}, open(f"{O}/mkbn.json", "w"))
by = {}
for k, v in tail.items():
    l = int(re.search(r"layers\.(\d+)\.", k)[1]) if "layers." in k else -1
    by.setdefault(l // 16 if l >= 0 else -1, []).append(v)
for b, v in sorted(by.items()):
    print(f"layers {b*16}-{b*16+15}: max block / median block, median over tensors {sorted(v)[len(v)//2]:.2f}, max {max(v):.2f}")
print(f"MKBN DONE {len(keys)} tensors, {len(ns)} blocks, {N} sequences -> {O}")
