"""w98: w81's f2calc.py (local/workers/w81/f2calc.py, 11-08) for #138-01's refit arms; the arithmetic is unchanged.

  python f2calc98.py ARMDIR CKPT HESS HOUT PIPELINE     (gpubox CPU, niced; caches ARMDIR/<arm>/f2.json, skips keys done)

ARMDIR/<arm>/wq/<key>.safetensors = each arm's decoded weight (f16, (out, in)), written by the rxq jobs with
RXQ_REFIT_ARMS_WQ=ARMDIR. f2 = sum_n h_n e_n H e_n^T / sum_n h_n w_n H w_n^T (rows n = output channels, H = the raw calib
input Hessian of the tensor's group, h = the bn H_out diagonal, e = W - Wq); f1 = the same with h = 1. Changes vs f2calc:
the glob (ARMDIR/<arm>/wq instead of OUT/<unit>/.done + wq), PIPELINE = the rxq checkout on sys.path, the arm
directories need no .done (each wq file is written .part -> rename by the job)."""
import glob, json, os, sys

import torch
from safetensors import safe_open

OUT, CKPT, HESS, HOUT, PIPE = sys.argv[1:6]
sys.path.insert(0, PIPE)
from rxq import quant, qwen35  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "8")))
ck = quant.Ckpt(CKPT)
Hc = {}
RF = f"{OUT}/ref.json"                                  # per key: sum_n h_n r_n, sum_n r_n (W and H only)
ref = json.load(open(RF)) if os.path.exists(RF) else {}
for u in sorted(d for d in glob.glob(f"{OUT}/*") if os.path.isdir(f"{d}/wq")):
    fp = f"{u}/f2.json"
    res = json.load(open(fp)) if os.path.exists(fp) else {}
    for wf in sorted(glob.glob(f"{u}/wq/*.safetensors")):
        key = os.path.basename(wf)[:-len(".safetensors")]
        if key in res: continue
        hname = qwen35.locate(key)[3]
        if hname not in Hc:
            Hc.clear(); Hc[hname] = quant.hessian(f"{HESS}/{hname}.safetensors", hname, "cpu")["H"].double()
        H = Hc[hname]
        with safe_open(wf, "pt") as f: Wq = f.get_tensor("weight").double()
        W = ck.get(key + ".weight").double()
        with safe_open(f"{HOUT}/{key}.safetensors", "pt") as f: h = f.get_tensor("hout_diag").double()
        e = torch.zeros(W.shape[0], dtype=torch.float64)
        for c in range(0, W.shape[0], 2048):
            E = W[c:c + 2048] - Wq[c:c + 2048]; e[c:c + 2048] = ((E @ H) * E).sum(1)
        if key not in ref:
            r = torch.zeros_like(e)
            for c in range(0, W.shape[0], 2048): r[c:c + 2048] = ((W[c:c + 2048] @ H) * W[c:c + 2048]).sum(1)
            ref[key] = [float((h * r).sum()), float(r.sum())]
            json.dump(ref, open(RF + ".part", "w"), indent=1); os.replace(RF + ".part", RF)
        res[key] = {"f2": float((h * e).sum()) / ref[key][0], "f1": float(e.sum()) / ref[key][1]}
        json.dump(res, open(fp + ".part", "w"), indent=1); os.replace(fp + ".part", fp)
        print(f"{os.path.basename(u):14s} {key:60s} f2 {res[key]['f2']:.6e} f1 {res[key]['f1']:.6e}", flush=True)
