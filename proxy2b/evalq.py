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
# EVALQ_AQ="REGEX:BITS:GROUP[:CLIP];..." -- fake activation quantization as the int kernels do it (v3/e1b/aqproxy.py):
# x -> x * suh -> 128-point Hadamard per block (normalised) -> symmetric int BITS per GROUP of features per token,
# scale = CLIP * amax / (2^(BITS-1) - 1), clamped -> back through the exact inverse. suh from the quant shards
# (EVALQ_AQ_SU: ':'-separated quant dirs with q-LNN.safetensors, searched in order; first hit wins).
naq = 0
if os.environ.get("EVALQ_AQ"):
    import re
    from safetensors import safe_open
    def _had(n=128):
        h = torch.ones(1, 1)
        while h.shape[0] < n: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        return (h / n ** 0.5).cuda()
    HAD = _had()
    SU = {}
    for d in [x for x in os.environ.get("EVALQ_AQ_SU", "").split(":") if x]:
        for f in sorted(glob.glob(f"{d}/q-L*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    if k.endswith(".suh") and k[:-4] not in SU: SU[k[:-4]] = h.get_tensor(k).float().cuda()
    rules = []
    for r in os.environ["EVALQ_AQ"].split(";"):
        pat, b, g_, *c = r.split(":"); rules.append((re.compile(pat), int(b), int(g_), float(c[0]) if c else 1.0))
    def aq(x, su, b, grp, clip):
        sh = x.shape; xf = x.float().reshape(-1, sh[-1]) * su
        xr = (xf.reshape(-1, sh[-1] // 128, 128) @ HAD).reshape(-1, sh[-1])
        xg = xr.reshape(-1, sh[-1] // grp, grp); qm = 2 ** (b - 1) - 1
        s = (clip * xg.abs().amax(-1, keepdim=True) / qm).clamp_min(1e-12)
        xq = (torch.round(xg / s).clamp(-qm, qm) * s).reshape(-1, sh[-1] // 128, 128) @ HAD.T
        return (xq.reshape(-1, sh[-1]) / su).reshape(sh).to(x.dtype)
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or "layers." not in n: continue
        hit = next((r for r in rules if r[0].search(n)), None)
        if not hit: continue
        key = "model.language_model.layers." + n.split("layers.", 1)[1]
        su = SU.get(key)
        assert su is not None, f"no suh for {key}"
        m.register_forward_pre_hook(lambda mod, inp, su=su, h=hit: (aq(inp[0], su, h[1], h[2], h[3]),) + inp[1:])
        naq += 1
    print(f"EVALQ_AQ: {naq} linears", file=sys.stderr)
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
if out:   # .part, fsync, rename: a kill never leaves a short X.f32 that the queues' "skip if it exists" would trust
    with open(out + ".part", "wb") as fh:
        np.stack([kl, t1, gen, np.zeros_like(kl)], 1).astype(np.float32).tofile(fh); fh.flush(); os.fsync(fh.fileno())
    os.replace(out + ".part", out)
print(json.dumps({"golden": gd, "wq": wq, "n_swapped": nsw, "n_aq": naq, "positions": int(gen.sum()), "kld": float(kl[gen].mean()),
                  "top1": float(t1[gen].mean()), "sec": round(time.time() - t0, 1)}))
