#!/usr/bin/env python3
"""evalq.py MODEL GOLDEN [--wq DIR] [--out ROWS.f32] -- fake-quant KLD of the 2B against a mkgold.py golden: the bf16
model with each linear whose quantized weight is in DIR (ourq.py --save-wq: DIR/<hf key>.safetensors, tensor "weight",
[out, in] fp16) substituted, scored with the challenge's arithmetic (common.score_rows), gen-masked. Prints one JSON
line {positions, kld mean, top1, n_swapped}; --out writes per-row [kl, top1, gen, 0] f32 for paired.py-style
comparisons."""
import glob, json, os, sys, time
import numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, Golden, score_rows, doc_logsoftmax

a = sys.argv[1:]
def opt(k):
    if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
wq, out = opt("--wq"), opt("--out")
model_path, gd = a[:2]
model, dec, head = load_model(model_path)
nsw = 0
if wq:
    params = {n.split("layers.", 1)[1]: p for n, p in model.named_parameters() if "layers." in n}
    for f in sorted(glob.glob(f"{wq}/*.safetensors")):
        key = os.path.basename(f)[:-len(".safetensors")]
        tgt = params[key.split("layers.", 1)[1] + ".weight"]
        w = load_file(f)["weight"]
        assert tuple(w.shape) == tuple(tgt.shape), (key, w.shape, tgt.shape)
        with torch.no_grad(): tgt.copy_(w.to(tgt.dtype))
        nsw += 1
g = Golden(gd)
KL, T1, GN, t0 = [], [], [], time.time()
for i, d in enumerate(g.docs):
    tk, sf = g.doc_tokens(i), d["score_from"]
    for j, lq in doc_logsoftmax(dec, head, tk, sf):
        r0 = d["row0"] + j; n = lq.shape[0]
        top = g.top[r0:r0 + n]
        kl, t1 = score_rows(lq, torch.from_numpy(np.ascontiguousarray(top["id"])).cuda(),
                            torch.from_numpy(np.ascontiguousarray(top["lp"])).cuda(),
                            torch.from_numpy(np.asarray(g.rows[r0:r0 + n, 2], np.float64)).cuda())
        KL.append(kl.cpu().numpy()); T1.append(t1.cpu().numpy())
kl, t1 = np.concatenate(KL), np.concatenate(T1); gen = g.gen[:len(kl)]
assert len(kl) == g.meta["rows"]
if out: np.stack([kl, t1, gen, np.zeros_like(kl)], 1).astype(np.float32).tofile(out)
print(json.dumps({"golden": gd, "wq": wq, "n_swapped": nsw, "positions": int(gen.sum()), "kld": float(kl[gen].mean()),
                  "top1": float(t1[gen].mean()), "sec": round(time.time() - t0, 1)}))
