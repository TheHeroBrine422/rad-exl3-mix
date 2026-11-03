#!/usr/bin/env python3
"""splice.py MODE CKPT HDIAG KEYS.txt BASEWQ OUTWQ [HIWQ LOWQ | UNIWQ] -- D6's fake-quant slab arms: an evalq.py --wq
dir with every tensor a symlink to BASEWQ, except the KEYS, which become
  sorted   per 128-row output block: HIWQ's rows on the half of the blocks with the highest predicted two-sided cost
           (blk.block_costs of h sv^2, no quantized data used), LOWQ's on the rest
  random   the same, a fixed random half of the blocks (rng seed 0 per key)
  uniform  UNIWQ's tensor (K2.5: the same bytes as half K3 / half K2)
Splicing output blocks of two quants of one tensor is what a per-slab K would produce: the output-side feedback never
crosses a 128-block and the input-side recursion is per output column. Not identical: each donor brings its own
refitted input scales (suh) and g_scale, so a real slab format needs per-slab input scales (2 bytes x in x slabs)."""
import os, sys
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blk, json, glob

mode, ck, hd, kf, bwq, owq = sys.argv[1:7]; rest = sys.argv[7:]
keys = [l.strip() for l in open(kf) if l.strip() and not l.startswith("#")]
idx = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
def link(src, dst):
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.abspath(src), dst)
os.makedirs(owq, exist_ok=True)
for f in glob.glob(f"{bwq}/*.safetensors"): link(f, f"{owq}/{os.path.basename(f)}")
for k in keys:
    dst = f"{owq}/{k}.safetensors"
    if mode == "uniform": link(f"{rest[0]}/{k}.safetensors", dst); continue
    with safe_open(f"{ck}/{idx[k + '.weight']}", "pt") as f: w = f.get_tensor(k + ".weight")
    with safe_open(f"{hd}/{k}.safetensors", "pt") as f: h = f.get_tensor("hout_diag").double().numpy()
    nb = w.shape[0] // blk.HB
    if mode == "sorted":
        c, _ = blk.block_costs(np.maximum(h, 1e-30 * h.max()) * blk.out_scales2(w)); top = np.argsort(-c)[:nb // 2]
    else:
        top = np.random.default_rng(0).permutation(nb)[:nb // 2]
    hi, lo = load_file(f"{rest[0]}/{k}.safetensors")["weight"], load_file(f"{rest[1]}/{k}.safetensors")["weight"]
    m = torch.zeros(nb, dtype=torch.bool); m[torch.from_numpy(top)] = True; m = m.repeat_interleave(blk.HB)
    os.remove(dst); save_file({"weight": torch.where(m[:, None], hi, lo).contiguous()}, dst)
print(f"{mode}: {len(keys)} tensors spliced into {owq}")
