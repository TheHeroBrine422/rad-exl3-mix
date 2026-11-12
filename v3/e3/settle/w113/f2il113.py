#!/usr/bin/env python3
"""f2il113.py OUT CKPT HESS HOUT PERMROOT -- w113 #148-01 (../w113.md): f2il101.py for arms N / P / T / V. Every finished
unit OUT/<arm>-K<K>-s<seed>/ gets f2.json: w81's f2 in the natural basis (Wq un-permuted with PERMROOT/<arm>/perm.json;
the permuted-basis f2 must agree to 1e-9) and f1. OUT/model.json: composable perm2.py's high-rate pivot sum (rng 1 signs,
sigma 0.01) for natural and each arm's order, per key. gpubox CPU, niced; THREADS (default 4)."""
import glob, json, os, sys

import numpy as np
import torch
from safetensors import safe_open

OUT, CKPT, HESS, HOUT, PR = sys.argv[1:6]
sys.path.insert(0, os.environ.get("W113_PIPE", "/data/w101/c/code/pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rxq import quant, qwen35  # noqa: E402
from perm113 import arm_perm, inverse  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
PERM = {a: {k: torch.tensor(v) for k, v in json.load(open(f"{PR}/{a}/perm.json")).items()}
        for a in "PTV" if os.path.exists(f"{PR}/{a}/perm.json")}
ck = quant.Ckpt(CKPT); Hc = {}


def hget(key):
    hname = qwen35.locate(key)[3]
    if hname not in Hc:
        Hc.clear(); Hc[hname] = quant.hessian(f"{HESS}/{hname}.safetensors", hname, "cpu")["H"].double()
    return Hc[hname]


def rowq(A, H):
    r = torch.zeros(A.shape[0], dtype=torch.float64)
    for c in range(0, A.shape[0], 2048): r[c:c + 2048] = ((A[c:c + 2048] @ H) * A[c:c + 2048]).sum(1)
    return r


def piv(v):
    Hd = np.array([[1.0]])
    while Hd.shape[0] < 128: Hd = np.block([[Hd, Hd], [Hd, -Hd]])
    Hd = Hd / np.sqrt(128); sg = np.random.default_rng(1).choice([-1.0, 1.0], 128); Hs = Hd * sg
    v = v / v.mean(); nb = len(v) // 128; s = 0.0
    for b in range(nb):
        M = Hs @ np.diag(v[128 * b:128 * b + 128]) @ Hs.T + 0.01 * np.eye(128)
        s += float((np.diag(np.linalg.cholesky(M[::-1, ::-1])) ** 2).sum())
    return s / (nb * 128)


def model_ratios(key, W, h):
    v = (h * W.pow(2).mean(1)).numpy(); o = {"nat": piv(v)}
    for a in "PTV":
        if a == "V" and not key.endswith(("v_proj", "k_proj")): continue
        o[a] = piv(v[arm_perm(key, a, len(v)).numpy()]); o["ratio_" + a] = o[a] / o["nat"]
    return o


MF = f"{OUT}/model.json"; model = json.load(open(MF)) if os.path.exists(MF) else {}
RF = f"{OUT}/ref.json"; ref = json.load(open(RF)) if os.path.exists(RF) else {}
for ud in sorted(glob.glob(f"{OUT}/*/.done")):
    u = os.path.dirname(ud); arm = os.path.basename(u).split("-")[0]; fp = f"{u}/f2.json"
    res = json.load(open(fp)) if os.path.exists(fp) else {}
    for wf in sorted(glob.glob(f"{u}/wq/*.safetensors")):
        key = os.path.basename(wf)[:-len(".safetensors")]
        if key in res: continue
        H = hget(key)
        with safe_open(wf, "pt") as f: Wq = f.get_tensor("weight").double()
        W = ck.get(key + ".weight").double()
        with safe_open(f"{HOUT}/{key}.safetensors", "pt") as f: h = f.get_tensor("hout_diag").double()
        if key not in ref:
            r = rowq(W, H); ref[key] = [float((h * r).sum()), float(r.sum())]
            json.dump(ref, open(RF + ".part", "w"), indent=1); os.replace(RF + ".part", RF)
        if key not in model:
            model[key] = model_ratios(key, W, h)
            json.dump(model, open(MF + ".part", "w"), indent=1); os.replace(MF + ".part", MF)
        o = {}
        if arm != "N":
            p = PERM[arm][key]; ip = inverse(p)
            assert torch.equal(p, arm_perm(key, arm, W.shape[0]))
            e_perm = rowq(W[p] - Wq, H)                       # permuted basis: stored rows vs permuted W, h
            o["f2_permbasis"] = float((h[p] * e_perm).sum()) / ref[key][0]
            Wq = Wq[ip]
        e = rowq(W - Wq, H)
        o.update(f2=float((h * e).sum()) / ref[key][0], f1=float(e.sum()) / ref[key][1])
        if arm != "N": assert abs(o["f2_permbasis"] / o["f2"] - 1) < 1e-9, (key, o)
        res[key] = o
        json.dump(res, open(fp + ".part", "w"), indent=1); os.replace(fp + ".part", fp)
        print(f"{os.path.basename(u):12s} {key:55s} f2 {o['f2']:.6e} f1 {o['f1']:.6e}", flush=True)
