#!/usr/bin/env python3
"""embgpu.py BF16.rad OUTDIR K[,K..] -- trellis-coded token embedding on the GPU, written as fake-quant donors.

Per K: every row of the checkpoint's token table (248320 x 5120) -> fwht128 per 128 columns (orthonormal) -> one bf16
scale per row (RMS x factor, best of 0.9 / 1.05 / 1.2 per row, by the row's squared error) -> 20 tiles of 256 values
quantized by exllamav3's trellis (mul1; fractional K through exl3frac) -> decoded, un-rotated, rounded to bf16 and
written over token_embd.weight of OUTDIR/emb-K<k>.rad, a reflink copy of BF16.rad (plain bf16 table: the engine reads
exactly the values a row-decode trellis gather would produce; only bytes and speed differ). Mix the entry into any
base. Prints per K the relative error sum (w-q)^2 / sum w^2 (cf. embtrellis.py's 1024-row CPU numbers) and bits per
value K + 16/5120. Env EXL3FRAC_SRC (+ exl3frac's build env) for quarter steps; one card (ROCR_VISIBLE_DEVICES)."""
import json, math, os, shutil, subprocess, sys, time
import numpy as np, torch

sys.path.insert(0, "/data/yaqa"); from headslab import entry
BF, OD, KS = sys.argv[1], sys.argv[2], [float(k) for k in sys.argv[3].split(",")]
os.makedirs(OD, exist_ok=True)
if any(not float(k).is_integer() for k in KS):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3frac")); import exl3frac
    exl3frac.install(os.environ["EXL3FRAC_SRC"])
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
dev = torch.device("cuda")
off, nbytes, shape = entry(BF, "token_embd.weight"); V, C = shape
assert nbytes == V * C * 2, (nbytes, shape)
W = torch.from_numpy(np.memmap(BF, np.int16, "r", offset=off, shape=(V, C)).copy()).view(torch.bfloat16)
H = torch.tensor([[1.0]])
while H.shape[0] < 128: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
H = (H / math.sqrt(128)).to(dev)
FAC = (0.9, 1.05, 1.2); CH = 8192
for K in KS:
    K = int(K) if float(K).is_integer() else K
    out = f"{OD}/emb-K{K}.rad"
    if os.path.exists(out + ".done"): continue
    t0 = time.time(); Q = torch.empty(V, C, dtype=torch.bfloat16); num = den = 0.0
    for r0 in range(0, V, CH):
        w = W[r0:r0 + CH].to(dev).float(); n = w.shape[0]
        v = (w.view(n, C // 128, 128) @ H).view(n, C)
        rms = v.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        best_e = best_q = None
        for f in FAC:
            s = (rms * f).bfloat16().float()
            q, _ = quantize_tiles((v / s).reshape(-1, 256).contiguous(), {"K": K, "mul1": True})
            q = q.view(n, C) * s
            e = (q - v).pow(2).sum(1)
            if best_e is None: best_e, best_q = e, q
            else: m = e < best_e; best_e = torch.where(m, e, best_e); best_q = torch.where(m[:, None], q, best_q)
        wq = (best_q.view(n, C // 128, 128) @ H.T).view(n, C)
        num += float((wq - w).pow(2).sum()); den += float(w.pow(2).sum())
        Q[r0:r0 + n] = wq.bfloat16().cpu()
    subprocess.run(["cp", "--reflink=always", BF, out + ".part"], check=True)
    with open(out + ".part", "r+b") as f:
        f.seek(off); f.write(Q.view(torch.int16).numpy().tobytes())
    os.replace(out + ".part", out); open(out + ".done", "w").write("")
    print(json.dumps({"K": K, "bpv": K + 16 / C, "rel": num / den, "sec": round(time.time() - t0, 1), "donor": out}), flush=True)
