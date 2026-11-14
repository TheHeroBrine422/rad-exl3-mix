#!/usr/bin/env python3
"""headprobe.py MODEL_DIR WQ HESSIAN OUT.json [N] -- (w155, #180-02) how well a quantized head ranks candidates off the
trunk's hidden distribution: the original lm_head W (MODEL_DIR's bf16 lm_head.weight) vs the head the trunk serves W'
(WQ: ourq's dequantized wq/lm_head.safetensors, f16, rows in vocabulary order -- checked here) on x drawn two ways:
  trunk  x ~ N(0, H), H the calib input Hessian of lm_head (E[x x^T] over calib: the distribution the head was fit on)
  iso    x ~ N(0, s^2 I), s^2 = trace(H) / n_embd (the same mean square, no preferred direction)
Per draw: the logit error ||W'x - Wx|| / ||Wx - mean||, and the rescoring question the drafter asks: among the top 32
of Wx (the exact head's), does W' rank the same top-1 first, and how many of the exact top 8 are in W's top 8 among
them. The same for an int8 g128 symmetric RTN of W (w153's rescoring copy) as the reference. CPU, torch, N draws each (default 64). Not a measurement on drafter hiddens (none are dumped): it shows whether
the served head's error is shaped to the trunk's distribution."""
import json, os, sys, time
import torch
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
md, wq_path, hp, out = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 64
t0 = time.time()
idx = json.load(open(os.path.join(md, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(md, idx["lm_head.weight"]), "pt") as f:
    W = f.get_tensor("lm_head.weight").float()
with safe_open(wq_path, "pt") as f:
    Wq = f.get_tensor(f.keys().__iter__().__next__()).float()
with safe_open(hp, "pt") as f:   # the challenge's packed upper triangle (row-major), k = n_embd
    ks = list(f.keys())
    meta = f.metadata() or {}
    u = f.get_tensor(ks[0]).double()
n = int(meta.get("k", W.shape[1]))
iu = torch.triu_indices(n, n)
H = torch.zeros(n, n, dtype=torch.float64)
H[iu[0], iu[1]] = u
H = H + H.T - torch.diag(torch.diag(H))
info = {"W": list(W.shape), "Wq": list(Wq.shape), "H": list(H.shape), "H_keys": ks}
# wq may hold the rows in R16 order (stored row s = vocabulary row ((s >> 4) & 7) N / 8 + (s >> 7) 16 + (s & 15)):
# take whichever order matches the original's rows
r0 = torch.arange(0, W.shape[0], 997)
c_nat = float(torch.nn.functional.cosine_similarity(W[r0], Wq[r0], dim=1).median())
sv = torch.arange(W.shape[0])
v_of_s = ((sv >> 4) & 7) * (W.shape[0] // 8) + (sv >> 7) * 16 + (sv & 15)
s_of_v = torch.empty_like(v_of_s)
s_of_v[v_of_s] = sv
c_r16 = float(torch.nn.functional.cosine_similarity(W[r0], Wq[s_of_v[r0]], dim=1).median())
info["row_cos_natural"], info["row_cos_r16"] = c_nat, c_r16
if c_r16 > c_nat:
    Wq = Wq[s_of_v]
    info["wq_order"] = "r16"
else:
    info["wq_order"] = "natural"
# row order: the served head's rows must be the original's (cosine of matching rows vs shifted rows)
r = torch.arange(0, W.shape[0], 997)
cos = torch.nn.functional.cosine_similarity(W[r], Wq[r], dim=1)
cos_shift = torch.nn.functional.cosine_similarity(W[r], Wq[(r + 1) % W.shape[0]], dim=1)
info["row_cos_median"] = float(cos.median())
info["row_cos_shift_median"] = float(cos_shift.median())
ev, V = torch.linalg.eigh(H)
ev = ev.clamp(min=0)
L = (V * ev.sqrt()).float()          # x = L z, z ~ N(0, I) -> cov H
s = float((ev.sum() / H.shape[0]).sqrt())
info["H_trace_over_n"] = s * s
info["H_eig_top_share"] = float(ev[-64:].sum() / ev.sum())
# the reference: w153's copy, int8 a (row, 128 inputs) with a symmetric absmax scale (the recipe's i8 g128 clamp=sym;
# its scale is bf16 in the recipe, f32 here: a little better than the copy)
Wg = W.view(W.shape[0], -1, 128)
sc = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-30) / 127.0
W8 = (torch.round(Wg / sc).clamp(-127, 127) * sc).view_as(W)
del Wg, sc
g = torch.Generator().manual_seed(155)
res = {}
for name in ("trunk", "iso"):
    z = torch.randn(H.shape[0], N, generator=g)
    X = L @ z if name == "trunk" else z * s
    Y = W @ X
    for hn, Wh in (("served_K4", Wq), ("int8_copy", W8)):
        Yq = Wh @ X
        rel, top1, top8 = [], [], []
        for j in range(N):
            y, yq = Y[:, j], Yq[:, j]
            rel.append(float((yq - y).norm() / (y - y.mean()).norm()))
            c = torch.topk(y, 32).indices
            order = c[torch.argsort(yq[c], descending=True)]
            top1.append(int(order[0] == c[0]))
            top8.append(len(set(order[:8].tolist()) & set(c[:8].tolist())))
        res[f"{name}/{hn}"] = {"rel_err_median": sorted(rel)[N // 2], "rel_err_mean": sum(rel) / N,
                               "top1_kept": sum(top1) / N, "top8_overlap_mean": sum(top8) / N}
info["draws"] = N
info["result"] = res
info["seconds"] = round(time.time() - t0, 1)
json.dump(info, open(out, "w"), indent=1)
print(json.dumps(info, indent=1))
