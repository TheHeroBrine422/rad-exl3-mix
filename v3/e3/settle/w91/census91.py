#!/usr/bin/env python3
"""census91.py LEVEL CKPT HESS OUT.jsonl [--layers A-B] [--device cuda:0] -- ../w91.md item 1 census: for every residual
writer (ssm_out / attn_output / ffn_down) of an ourq-layout level, per output row r (= residual channel r) the one-sided
weighted error e_r = [(W - What) H (W - What)^T]_rr and the output energy o_r = [W H W^T]_rr, H the calib input Hessian
of the writer's input group (rxq.quant.hessian: the plain sum, unscaled; shares are scale-free). One JSON line per
tensor (resumable: tensors already in OUT are skipped): row 3994's shares, the top-8 rows of e and of o, the full share
vectors as f32 in OUT.npz-dir (OUT.d/<key>.npy, [2, rows]: e-share, o-share)."""
import argparse, json, os, sys, time

import numpy as np
import torch

from rxq import deq, qwen35
from rxq.quant import Ckpt, hessian

ap = argparse.ArgumentParser()
ap.add_argument("level"); ap.add_argument("ckpt"); ap.add_argument("hess"); ap.add_argument("out")
ap.add_argument("--layers", default="0-63"); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
dev = torch.device(a.device)
lo, hi = map(int, a.layers.split("-"))
order = list(range(0, 4)) + [l for l in range(4, 64)]
done = set()
if os.path.exists(a.out):
    done = {json.loads(x)["key"] for x in open(a.out) if x.strip()}
os.makedirs(a.out + ".d", exist_ok=True)
lev, ck = deq.Level(a.level, dev), Ckpt(a.ckpt)
for l in order:
    if not lo <= l <= hi: continue
    for hn, keys in qwen35.layer_plan(l):
        for key in keys:
            if qwen35.family(key) not in qwen35.WRITERS or key in done: continue
            t0 = time.time()
            W = ck.get(key + ".weight").to(dev, torch.float32)            # (out, in)
            Wh = lev.get(key)                                             # (out, in) f32
            assert W.shape == Wh.shape, (key, W.shape, Wh.shape)
            H = hessian(os.path.join(a.hess, hn + ".safetensors"), key, dev)["H"]
            E = W - Wh
            e = ((E @ H) * E).sum(1).double(); o = ((W @ H) * W).sum(1).double()
            del H, E
            es, os_ = (e / e.sum()).cpu().numpy(), (o / o.sum()).cpu().numpy()
            np.save(os.path.join(a.out + ".d", key + ".npy"), np.stack([es, os_]).astype(np.float32))
            te, to = np.argsort(-es)[:8], np.argsort(-os_)[:8]
            rel = float(((W - Wh) ** 2).sum() / (W ** 2).sum())
            rec = {"key": key, "layer": l, "family": qwen35.family(key), "rows": int(W.shape[0]), "cols": int(W.shape[1]),
                   "e3994": float(es[3994]), "o3994": float(os_[3994]),
                   "top_e": [[int(i), float(es[i])] for i in te], "top_o": [[int(i), float(os_[i])] for i in to],
                   "e_sum": float(e.sum()), "o_sum": float(o.sum()), "rel_mse": rel,
                   "row3994_rel_mse": float(((W[3994] - Wh[3994]) ** 2).sum() / (W[3994] ** 2).sum()),
                   "row3994_absmax": float(W[3994].abs().max()), "absmax_other": float(torch.cat([W[:3994], W[3995:]]).abs().max()),
                   "sec": round(time.time() - t0, 1)}
            with open(a.out, "a") as fh: fh.write(json.dumps(rec) + "\n")
            print(f"{key}: e3994 {rec['e3994']:.4f} o3994 {rec['o3994']:.4f} top_e {rec['top_e'][:3]} ({rec['sec']} s)", flush=True)
            del W, Wh
            torch.cuda.empty_cache() if dev.type == "cuda" else None
print("CENSUS91 DONE", flush=True)
