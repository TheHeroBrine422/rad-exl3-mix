#!/usr/bin/env python3
"""evalm.py PARAMS WQ GOLDEN OUT.f32 -- w79 confounder check: score a healed arm on a golden through heal.py's own
student (stack weights from WQ, PARAMS' multipliers applied in memory by w06-heal's wrappers, no fold / no weight
re-rounding) with evalq.py's arithmetic. PARAMS '-' = the plain stack through the same wrappers (identity)."""
import json, os, sys, time
import numpy as np, torch
P = "/workspace/rad-exl3-mix/proxy2b"
sys.path[:0] = [P, f"{P}/exp/w06-heal"]
from common import load_model, Golden, score_rows, doc_logsoftmax
import heal
params, wq, gd, out = sys.argv[1:5]
cap = float(os.environ.get("VRAM_CAP_GB", 0))
if cap: torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / torch.cuda.get_device_properties(0).total_memory))
model, dec, head = load_model(heal.CKPT)
model.requires_grad_(False)
stu = heal.Student(model, wq)
if params != "-": stu.load_state(torch.load(params, weights_only=False))
model.eval(); g = Golden(gd); KL, T1, t0 = [], [], time.time()
with torch.no_grad():
    for i, d in enumerate(g.docs):
        tk, sf = g.doc_tokens(i), d["score_from"]
        for j, lq in doc_logsoftmax(dec, head, tk, sf):
            r0 = d["row0"] + j; n = lq.shape[0]; top = g.top[r0:r0 + n]
            kl, t1 = score_rows(lq, torch.from_numpy(np.ascontiguousarray(top["id"])).cuda(),
                                torch.from_numpy(np.ascontiguousarray(top["lp"])).cuda(),
                                torch.from_numpy(np.asarray(g.rows[r0:r0 + n, 2], np.float64)).cuda())
            KL.append(kl.cpu().numpy()); T1.append(t1.cpu().numpy())
kl, t1 = np.concatenate(KL), np.concatenate(T1); gen = g.gen[:len(kl)]
assert len(kl) == g.meta["rows"]
with open(out + ".part", "wb") as fh: np.stack([kl, t1, gen, np.zeros_like(kl)], 1).astype(np.float32).tofile(fh)
os.replace(out + ".part", out)
print(json.dumps({"golden": gd, "params": params, "kld": float(kl[gen].mean()), "top1": float(t1[gen].mean()), "sec": round(time.time() - t0, 1)}))
