#!/usr/bin/env python3
"""hgate.py HXDIR CKPT QUANTIZE.py OUT.json [--sens HEADSENS.npz] [--rows 8192] -- CPU gate for E4 before any GPU quant:
does quantizing the head under the gen-row input Hessian (HXDIR/hin-gen, headx.py) instead of the all-position one
(HXDIR/hin-all, same documents) lower the head's error ON GEN ROWS? exllamav3's own quantizer code on the CPU with a
scalar stand-in for the trellis (w05-head2s/src/cpuq.py), one-sided, mean-row, K2 / K3, on N head rows (64 evenly
spaced runs in vocabulary order). Error = sum_v F_v (w - q)_vᵀ H_gen (w - q)_v (F: --sens fisher_qad_gen, else 1).
In-sample for the gen arm (its own metric), so this is the most the gen Hessian can buy: below 2% -> E4 stops."""
import argparse, json, os, sys
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../w05-head2s/src"))
import cpuq
ap = argparse.ArgumentParser(); ap.add_argument("hx"); ap.add_argument("ckpt"); ap.add_argument("quant"); ap.add_argument("out")
ap.add_argument("--sens"); ap.add_argument("--rows", type=int, default=8192); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args(); torch.set_num_threads(a.threads)
def hess(p):
    with safe_open(p, "pt") as f: up = f.get_tensor("hessian_upper").double(); md = f.metadata()
    k = int(md["k"]); H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up
    return (H + H.triu(1).T).float(), int(md["rows"])
(Hg, ng), (Ha, na) = hess(f"{a.hx}/hin-gen/lm_head.input.safetensors"), hess(f"{a.hx}/hin-all/lm_head.input.safetensors")
idx = json.load(open(f"{a.ckpt}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{a.ckpt}/{idx}", "pt") as f: W = f.get_tensor("lm_head.weight").float()
V = W.shape[0]; W -= W.mean(0, keepdim=True)
run = a.rows // 64; rows = torch.cat([torch.arange(s, s + run) for s in np.linspace(0, V - run, 64).astype(int)])
Wn = W[rows].T.contiguous()
F = torch.from_numpy(np.load(a.sens)["fisher_qad_gen"][rows.numpy()]).float() if a.sens else torch.ones(len(rows))
qm = cpuq.load(a.quant); res = {"gen_rows": ng, "all_rows": na, "F": "fisher_qad_gen" if a.sens else "uniform"}
Hm = Hg / ng                                         # the metric: mean over gen rows
def cost(wq): E = Wn - wq; return float(((Hm @ E) * E).sum(0) @ F)
for K in (2, 3):
    ea = cost(cpuq.quant(qm, Wn, Ha / na * 1000, K=K)[0]); eg = cost(cpuq.quant(qm, Wn, Hg / ng * 1000, K=K)[0])
    res[f"K{K}"] = eg / ea; print(f"K{K}: gen-H quant / all-H quant, error on gen rows: {eg / ea:.4f}", flush=True)
res["go"] = min(res["K2"], res["K3"]) <= 0.98
print("E4 GATE", "GO" if res["go"] else "STOP (< 2% in-sample)"); json.dump(res, open(a.out, "w"), indent=1)
