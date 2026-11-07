#!/usr/bin/env python3
"""gpureport.py OUT SETTLE_DATA CKPT HESS -- gpucheck.sh's check unit, read (CPU only):
  1. off (patched ourq, no --tile-cd) vs orig (today's ourq): every shard byte-identical (codes, scales, code stamp in
     the metadata), stamp.json stamps equal
  2. per tensor and K: cd3 / off proxy error after the scale refit, per output row from --save-wq's wq (f16) and the
     checkpoint weight (sum err / sum ref, the
     undamped calib H in the original basis: settle.py's "rel"), whole tensor and on the 128-row blocks settle.py
     sampled (its seeds 0 and 1); beside it settle.py's cd3 / ldlq for the same tensor, K and blocks
  3. tilecd.jsonl: the quantization-domain ratio and accepted shares; per-tensor seconds (proxy.jsonl "s")"""
import glob, hashlib, json, math, os, sys
import numpy as np
import torch
from safetensors import safe_open
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))

O, SD, CKPT, HESS = sys.argv[1:5]
L = "model.language_model.layers"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
res = {"identity": {}, "tensors": []}
# 1
for kd in sorted(glob.glob(f"{O}/orig/K*")):
    K = os.path.basename(kd); same = []
    for f in sorted(glob.glob(f"{kd}/q-L*.safetensors")):
        g = f"{O}/off/{K}/{os.path.basename(f)}"; same.append(os.path.exists(g) and sha(f) == sha(g))
    st = lambda arm: [e["stamp"] for e in json.load(open(f"{O}/{arm}/{K}/stamp.json"))]
    res["identity"][K] = {"shards": len(same), "identical": sum(same), "stamp_orig": st("orig"), "stamp_off": st("off")}
    print(f"{K}: {sum(same)}/{len(same)} shards byte-identical orig vs off; stamps orig {st('orig')} off {st('off')}")
# settle.py reference
ref = {}
for f in glob.glob(f"{SD}/*/cd/s*/*-attn_qg-K*.jsonl"):
    o = json.loads(open(f).readline())
    ref[(o["layer"], float(o["bits"]), o["seed"])] = (o["rel"]["cd3"] / o["rel"]["ldlq"], o["blocks"])
# ourq's own wq/<key>.rowerr.npz is wrong (err ~ ref: it uses the in-place regularized weight; w46, reval.md), so the
# per-row errors are recomputed here from the dequantized weight it saved
_W, _H = {}, {}
def rows(arm, K, key):
    if key not in _W:
        ix = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
        with safe_open(f"{CKPT}/{ix[key + '.weight']}", "pt") as f: _W.clear(); _W[key] = f.get_tensor(key + ".weight").float()
        hn = key.rsplit(".self_attn", 1)[0] + ".input"
        with safe_open(f"{HESS}/{hn}.safetensors", "pt") as f: up = f.get_tensor("hessian_upper").float()
        k = _W[key].shape[1]; H = torch.zeros(k, k); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up; H = H + H.triu(1).T
        _H.clear(); _H[key] = H / H.diagonal().mean()
    W, H = _W[key], _H[key]
    with safe_open(f"{O}/{arm}/K{K}/wq/{key}.safetensors", "pt") as f: Wq = f.get_tensor("weight").float()
    E = W - Wq
    return ((E @ H) * E).sum(1).double().numpy(), ((W @ H) * W).sum(1).double().numpy()
def prox(arm, K):
    return {j["key"]: j for j in map(json.loads, open(f"{O}/{arm}/K{K}/proxy.jsonl"))}
geo = lambda x: math.exp(sum(map(math.log, x)) / len(x))
print(f"\n{'layer':>5} {'K':>3} | {'GPU whole':>9} {'GPU qdom':>8} | {'GPU blk s0':>10} {'CPU s0':>7} {'GPU blk s1':>10} {'CPU s1':>7} | "
      f"{'CPU geo':>7} {'GPU-CPU':>7} | {'acc cd1/2/3':>17} {'s off':>5} {'s cd3':>5}")
for K in (2, 3, 4):
    po, pc = prox("off", K), prox("cd3", K)
    cdl = {j["key"]: j for j in map(json.loads, open(f"{O}/cd3/K{K}/tilecd.jsonl"))}
    for l in (3, 19, 43):
        key = f"{L}.{l}.self_attn.q_proj"
        eo, r = rows("off", K, key); ec, _ = rows("cd3", K, key)
        whole = ec.sum() / eo.sum()
        own = {"off": eo.sum() / r.sum(), "cd3": ec.sum() / r.sum()}           # vs ourq's proxy_err (refit_scales' e_after)
        blk = {}
        for s in (0, 1):
            cpu, b = ref[(l, float(K), s)]
            idx = np.concatenate([np.arange(128 * i, 128 * i + 128) for i in b])
            blk[s] = (ec[idx].sum() / eo[idx].sum(), cpu)
        cg = math.sqrt(blk[0][1] * blk[1][1]); c = cdl[key]
        t = {"layer": l, "K": K, "gpu_whole": whole, "gpu_qdomain": c["ratio"], "gpu_block_s0": blk[0][0], "cpu_s0": blk[0][1],
             "gpu_block_s1": blk[1][0], "cpu_s1": blk[1][1], "cpu_geo": cg, "proxy_err_ratio": pc[key]["proxy_err"] / po[key]["proxy_err"],
             "accepted": c["accepted"], "cd_sec": c["sec"], "s_off": po[key]["s"], "s_cd3": pc[key]["s"],
             "rel_off": own["off"], "rel_cd3": own["cd3"], "proxy_err_off": po[key]["proxy_err"], "proxy_err_cd3": pc[key]["proxy_err"]}
        res["tensors"].append(t)
        print(f"{l:5d} {K:3d} | {whole:9.4f} {c['ratio']:8.4f} | {blk[0][0]:10.4f} {blk[0][1]:7.4f} {blk[1][0]:10.4f} {blk[1][1]:7.4f} | "
              f"{cg:7.4f} {100 * (whole / cg - 1):+6.2f}% | " + "/".join(f"{v:.3f}" for v in c["accepted"].values())
              + f" {po[key]['s']:5.1f} {pc[key]['s']:5.1f} | rel off {own['off']:.4e} (proxy_err {po[key]['proxy_err']:.4e}) "
              f"cd3 {own['cd3']:.4e} ({pc[key]['proxy_err']:.4e})")
T = res["tensors"]
for K in (2, 3, 4):
    x = [t for t in T if t["K"] == K]
    print(f"K{K} geo over 3 tensors: GPU whole {geo([t['gpu_whole'] for t in x]):.4f}, CPU {geo([t['cpu_geo'] for t in x]):.4f}; "
          f"same blocks GPU {geo([t[f'gpu_block_s{s}'] for t in x for s in (0, 1)]):.4f} vs CPU {geo([t[f'cpu_s{s}'] for t in x for s in (0, 1)]):.4f}")
json.dump(res, open(f"{O}/gpureport.json", "w"), indent=1)
