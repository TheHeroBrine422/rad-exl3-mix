#!/usr/bin/env python3
"""f2cd102.py OUT CKPT HESS HOUT -- w102 #136-03 (../w102.md): w81's f2 (f2calc.py) of every finished unit OUT/<arm>-K<K>/:
f2 = sum_n h_n e_n H e_n^T / sum_n h_n w_n H w_n^T, e = W - Wq (Wq = --save-wq's decoded weight), H = the raw calib H_in
of the tensor's group, h = bn1920; f1 = the same with h = 1. Writes OUT/f2.json {unit: {key: {f2, f1}}} (keys already
there are skipped) and prints r = f2_cd / f2_off per key and K with the geo mean (the rule's R(K)). gpubox CPU, niced."""
import glob, json, math, os, sys

import torch
from safetensors import safe_open

OUT, CKPT, HESS, HOUT = sys.argv[1:5]
sys.path.insert(0, os.environ.get("W102_PIPE", "/data/w102/code/pipeline"))
from rxq import quant, qwen35  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
ck = quant.Ckpt(CKPT); Hc = {}
F = f"{OUT}/f2.json"; res = json.load(open(F)) if os.path.exists(F) else {}


def rowq(A, H):
    r = torch.zeros(A.shape[0], dtype=torch.float64)
    for c in range(0, A.shape[0], 2048): r[c:c + 2048] = ((A[c:c + 2048] @ H) * A[c:c + 2048]).sum(1)
    return r


for ud in sorted(glob.glob(f"{OUT}/*/.done")):
    u = os.path.basename(os.path.dirname(ud)); res.setdefault(u, {})
    for wf in sorted(glob.glob(f"{OUT}/{u}/wq/*.safetensors")):
        key = os.path.basename(wf)[:-len(".safetensors")]
        if key in res[u]: continue
        hname = qwen35.locate(key)[3]
        if hname not in Hc:
            Hc.clear(); Hc[hname] = quant.hessian(f"{HESS}/{hname}.safetensors", hname, "cpu")["H"].double()
        H = Hc[hname]
        with safe_open(wf, "pt") as f: Wq = f.get_tensor("weight").double()
        W = ck.get(key + ".weight").double()
        with safe_open(f"{HOUT}/{key}.safetensors", "pt") as f: h = f.get_tensor("hout_diag").double()
        r, e = rowq(W, H), rowq(W - Wq, H)
        res[u][key] = {"f2": float((h * e).sum() / (h * r).sum()), "f1": float(e.sum() / r.sum())}
        json.dump(res, open(F + ".part", "w"), indent=1); os.replace(F + ".part", F)
        print(f"{u:10s} {key:50s} f2 {res[u][key]['f2']:.6e} f1 {res[u][key]['f1']:.6e}", flush=True)

for K in sorted({u.split("-K")[1] for u in res}, key=float):
    a, b = res.get(f"cd-K{K}", {}), res.get(f"off-K{K}", {})
    ks = sorted(set(a) & set(b))
    if not ks: continue
    rs = {k: a[k]["f2"] / b[k]["f2"] for k in ks}; r1 = {k: a[k]["f1"] / b[k]["f1"] for k in ks}
    g = math.exp(sum(math.log(v) for v in rs.values()) / len(rs))
    for k in ks: print(f"K{K} {k:50s} r_f2 {rs[k]:.5f} r_f1 {r1[k]:.5f}")
    print(f"K{K}: R = geo f2_cd / f2_off over {len(rs)} tensors = {g:.5f} -> {'PASS' if g <= 1.002 and len(rs) == 4 else 'FAIL' if len(rs) == 4 else 'incomplete'}")
