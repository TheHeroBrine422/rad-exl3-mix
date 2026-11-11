#!/usr/bin/env python3
"""r14chk.py PIPELINE TDIR CKPT HESS HOUT -- R14 clamp (rxq.refit.clamp_scales, quant.scale_clamp) on stored jobs: for
every t-file whose suh / svh has values under the clamp floor, the one- and two-sided proxy of the decoded weight
before and after clamping (real100.py's arithmetic; no pre-refit values on disk, so the floor value is used)."""
import glob, json, sys
import torch
from safetensors import safe_open
PIPE, TD, CK, HS, HO = sys.argv[1:6]
sys.path.insert(0, PIPE)
from rxq import deq, post, qwen35  # noqa: E402
from rxq.refit import clamp_scales  # noqa: E402
torch.set_num_threads(4)
ix = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
def rel(Q, W, M, hw):
    QM = Q @ M; WM = W @ M
    dd = (QM * Q).sum(1) - 2 * (QM * W).sum(1) + (WM * W).sum(1); ww = (WM * W).sum(1)
    return float(dd.sum() / ww.sum()), float((hw * dd).sum() / (hw * ww).sum())
for f in sorted(glob.glob(f"{TD}/*/t-*.safetensors")):
    with safe_open(f, "pt") as h:
        md = h.metadata(); key = md["key"]
        tr, suh, svh = (h.get_tensor(f"{key}.{t}") for t in ("trellis", "suh", "svh"))
    suc, nu = clamp_scales(suh, suh); svc, nv = clamp_scales(svh, svh)
    if not (nu or nv): continue
    lut = torch.from_numpy(deq.lut(md["codebook"])); K = tr.shape[2] / 16
    Q0 = deq.weight(tr, suh, svh, K, lut).T.double(); Q1 = deq.weight(tr, suc.half(), svc.half(), K, lut).T.double()
    with safe_open(f"{CK}/{ix[key + '.weight']}", "pt") as h: W = h.get_tensor(key + ".weight").double()
    _, _, _, hn, _ = qwen35.locate(key)
    M = post.hessian(f"{HS}/{hn}.safetensors", "cpu")
    with safe_open(f"{HO}/{key}.safetensors", "pt") as h: hw = h.get_tensor("hout_diag").double()
    a1, a2 = rel(Q0, W, M, hw); b1, b2 = rel(Q1, W, M, hw)
    print(json.dumps({"key": key, "K": md["K"], "su_clamped": nu, "sv_clamped": nv, "rel1": [a1, b1], "rel2": [a2, b2],
                      "ratio1": b1 / a1, "ratio2": b2 / a2}), flush=True)
