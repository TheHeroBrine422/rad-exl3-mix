#!/usr/bin/env python3
"""c9w75.py MIX OUT --alpha attn_o=A,gdn_out=B,down=C [--dn DNDIR] [--cmax X] [--model M] [--hess H] -- w75 #112-01
(../w75.md): w64's c9w.py (item 3, #112) unchanged, plus --cmax (clamp every multiplier to <= X; default off) and the
fuller multiplier spread per writer in c9w.jsonl (c quantiles, rows with c > 1.5 / > 2). w64's docstring follows.

c9w.py MIX OUT --alpha attn_o=A,gdn_out=B,down=C [--dn DNDIR] [--model M] [--hess H] -- w64 item 3 (#112, ../w64.md):
C9's per-row svh multiplier (#85, w02 refitq.py --alpha) on the writers of a 0.8B mix, with the row error measured
against the target and metric the writer was actually fitted to:
  plain writers (no --dn):   t = W (bf16 checkpoint), M = calib H_in of the writer's group (raw, mean x x^T)
  asym writers (--dn DNDIR): t = W F = (Ft W^T)^T with DNDIR/wpre/<group>.safetensors "Ft", M = G from
                             DNDIR/hin/<group>.safetensors (mkdenoise's G, challenge layout; raw)
  e_n = (q_n - t_n) M (q_n - t_n)^T / t_n M t_n^T,   c_n = (1 - e_n)^(-A/2)   (A = 1 restores the row's M-norm)
MIX: a dir of 150 (out, in) f16 weight files (<key>.safetensors "weight", evalq --wq's layout). OUT: the same set:
links to MIX for every file but the changed writers, which are written as Q * c (rows), f16. OUT/c9w.jsonl: per
writer e_n mean / max and the multiplier's mean / sd / max. Families with A = 0 are linked unchanged. Codes are
untouched (the multiplier folds into svh). e_n is clipped to 0.5 (never reached at K >= 2: logged)."""
import argparse, json, os, sys
import torch
sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b/exp/w02-fixedscale")
import fsutil as U
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("mix"); ap.add_argument("out")
ap.add_argument("--alpha", required=True)
ap.add_argument("--dn", default=None)
ap.add_argument("--model", default="/data/models/qwen35-08b")
ap.add_argument("--hess", default="/data/h08/calib")
ap.add_argument("--device", default="cpu")
ap.add_argument("--cmax", type=float, default=None)
a = ap.parse_args()
AL = {k: float(v) for k, v in (x.split("=") for x in a.alpha.split(","))}
WR = ("attn_o", "gdn_out", "down")
assert set(AL) <= set(WR), AL
dev = a.device
os.makedirs(a.out + ".part", exist_ok=True)
ck = U.Ckpt(a.model)
keys = sorted(f[:-12] for f in os.listdir(a.mix) if f.endswith(".safetensors"))
assert len(keys) == 150, len(keys)
log = []
for key in keys:
    dst = f"{a.out}.part/{key}.safetensors"
    f = U.fam(key); al = AL.get(f, 0.0)
    if al == 0.0:
        if not os.path.lexists(dst): os.symlink(os.path.realpath(f"{a.mix}/{key}.safetensors"), dst)
        continue
    with safe_open(f"{a.mix}/{key}.safetensors", "pt", device="cpu") as h: Q = h.get_tensor("weight").to(dev).double()  # (out, in)
    W = ck.get(key + ".weight").to(dev).double()
    g = U.group(key)
    if a.dn:
        M = U.hessian(f"{a.dn}/hin/{g}.safetensors", dev).double()
        with safe_open(f"{a.dn}/wpre/{g}.safetensors", "pt", device="cpu") as h: Ft = h.get_tensor("Ft").to(dev).double()
        T = (Ft @ W.T).T
    else:
        M = U.hessian(f"{a.hess}/{g}.safetensors", dev).double(); T = W
    E = Q - T
    e = ((E @ M) * E).sum(1) / ((T @ M) * T).sum(1).clamp(min=1e-300)
    ncl = int((e > 0.5).sum()); e = e.clamp(max=0.5, min=0.0)
    c = (1 - e).pow(-al / 2)
    c0 = c.clone(); ncm = 0
    if a.cmax is not None: ncm = int((c > a.cmax).sum()); c = c.clamp(max=a.cmax)
    qs = torch.quantile(c0.float().cpu(), torch.tensor([0.01, 0.5, 0.99])).tolist()
    save_file({"weight": (Q * c[:, None]).half().contiguous().cpu()}, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    log.append({"key": key, "fam": f, "alpha": al, "e_mean": float(e.mean()), "e_max": float(e.max()), "clipped": ncl,
                "c_mean": float(c.mean()), "c_sd": float(c.std()), "c_max": float(c.max()), "target": "WF/G" if a.dn else "W/H",
                "rows": int(c.numel()), "c_min": float(c.min()), "c_q01_q50_q99": qs, "n_gt15": int((c0 > 1.5).sum()),
                "n_gt2": int((c0 > 2).sum()), "c_max_unclamped": float(c0.max()), "cmax": a.cmax, "n_clamped": ncm})
with open(f"{a.out}.part/c9w.jsonl", "w") as fh:
    for o in log: fh.write(json.dumps(o) + "\n")
os.replace(a.out + ".part", a.out)
print(json.dumps({"changed": len(log), "alpha": AL, "c_mean": sum(o["c_mean"] for o in log) / max(len(log), 1)}))
