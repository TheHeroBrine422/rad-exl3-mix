#!/usr/bin/env python3
"""rowerrcheck.py WQDIR [CKPT HIN] -- are ourq --save-wq's <key>.rowerr.npz right? Per tensor: the file's summed
err / ref against ourq's own proxy_err (proxy.jsonl beside WQDIR: the refit's e_after, undamped H: equal to the
true relative error), and with CKPT HIN the per-row Spearman / ratio against a recomputation from the bf16 weight
(stats.py's rowquad). Before the fix (rowerr.patch) the file is computed on the regularized weight quantize_exl3
leaves behind in place, not the original: err/ref ~ 1 instead of ~ proxy_err."""
import glob, json, os, sys
import numpy as np, torch
from scipy.stats import spearmanr
wq = sys.argv[1]; px = {json.loads(l)["key"]: json.loads(l)["proxy_err"] for l in open(f"{wq}/../proxy.jsonl")}
rows = []
for f in sorted(glob.glob(f"{wq}/*.rowerr.npz")):
    k = os.path.basename(f)[:-len(".rowerr.npz")]; z = np.load(f)
    rows.append((k, z["err"].sum() / z["ref"].sum(), px.get(k)))
r = np.array([x[1] / x[2] for x in rows if x[2]])
print(f"{len(rows)} files: file err/ref over proxy_err: median {np.median(r):.3g}, min {r.min():.3g}, max {r.max():.3g} "
      f"({'OK' if np.all(np.abs(r - 1) < 0.01) else 'WRONG'})")
if len(sys.argv) > 3:
    ck, hin = sys.argv[2:4]
    sys.argv = [sys.argv[0], ck, "/dev/null", "-"]
    from safetensors import safe_open
    idx = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
    import struct
    for k, *_ in rows[:3]:                                   # a per-row check on a few tensors
        with safe_open(f"{ck}/{idx[k + '.weight']}", "pt") as h: w = h.get_tensor(k + ".weight").float()
        with safe_open(f"{wq}/{k}.safetensors", "pt") as h: q = h.get_tensor("weight").float()
        g = json.load(open(f"{wq}/../stamp.json"))           # only to show where the run came from
        p = k.rsplit(".", 2)[0]; rest = ".".join(k.rsplit(".", 2)[1:])
        gn = {"mlp.gate_proj": "post_attention.input", "mlp.up_proj": "post_attention.input", "mlp.down_proj": "mlp.down_proj.input",
              "linear_attn.out_proj": "linear_attn.out_proj.input", "self_attn.o_proj": "self_attn.o_proj.input"}.get(rest, "input")
        hp = f"{hin}/{p}.{gn}.safetensors"
        with open(hp, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; kk = int(json.loads(fh.read(n))["__metadata__"]["k"])
        with safe_open(hp, "pt") as h: up = h.get_tensor("hessian_upper")
        H = torch.zeros(kk, kk); iu = torch.triu_indices(kk, kk); H[iu[0], iu[1]] = up; H = H + H.triu(1).T
        E = w - q; e = ((E @ H) * E).sum(1).double().numpy(); fe = np.load(f"{wq}/{k}.rowerr.npz")["err"]
        print(f"  {k}: per-row Spearman file vs recomputed {spearmanr(fe, e).correlation:.3f}, sum ratio {fe.sum() / e.sum():.3g}")
