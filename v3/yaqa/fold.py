#!/usr/bin/env python3
"""fold.py ROWS OUT LAYERS [SEQ_LO-SEQ_HI] -- output-side Hessians from collect.py's rows.

For every trunk linear of the given layers (e.g. "22,23" or "0-63"), reads ROWS/<l>.<module>/<seq>.bf16 ([NPOS][out])
for the sequences in SEQ_LO..SEQ_HI (default all), and writes H_out = sum_t r_t r_tᵀ (fp32, square) to
OUT/<checkpoint key>.safetensors as "hout", with metadata rows / seqs -- the layout exllamav3's --hessians reads.
Device: env FOLD_DEVICE (default cpu; "cuda:N" for the full model)."""
import os, sys, numpy as np, torch
from safetensors.torch import save_file

ROWS, OUT, LAY = sys.argv[1], sys.argv[2], sys.argv[3]
lo, hi = map(int, sys.argv[4].split("-")) if len(sys.argv) > 4 else (0, 10**9)
dev = os.environ.get("FOLD_DEVICE", "cpu")
NPOS = int(os.environ.get("NPOS", 32))     # rows per sequence (collect.py's NPOS)
torch.set_num_threads(int(os.environ.get("FOLD_THREADS", os.cpu_count())))
layers = set()
for p in LAY.split(","):
    a, b = (p.split("-") + [p])[:2]; layers.update(range(int(a), int(b) + 1))
os.makedirs(OUT, exist_ok=True)
for name in sorted(os.listdir(ROWS)):
    if name.startswith(".") or int(name.split(".")[0]) not in layers: continue
    key = "model.language_model.layers." + name
    if os.path.exists(f"{OUT}/{key}.safetensors"): continue
    fs = sorted(f for f in os.listdir(f"{ROWS}/{name}") if lo <= int(f[:5]) <= hi)
    H = None; n = 0
    for i in range(0, len(fs), 64):                   # 64 sequences per GEMM
        r = np.concatenate([np.fromfile(f"{ROWS}/{name}/{f}", np.int16).reshape(NPOS, -1) for f in fs[i:i + 64]])
        R = torch.from_numpy(r).view(torch.bfloat16).to(dev).float()
        H = R.T @ R if H is None else H.addmm_(R.T, R)
        n += R.shape[0]
    save_file({"hout": H.cpu().contiguous()}, f"{OUT}/{key}.safetensors.part",
              metadata={"rows": str(n), "seqs": str(len(fs)), "source": "collect.py sampled-label Fisher, |x|-weighted dy"})
    os.replace(f"{OUT}/{key}.safetensors.part", f"{OUT}/{key}.safetensors")
    d = H.diagonal()
    print(f"{key}: out {H.shape[0]}, {n} rows from {len(fs)} seqs, diag max/mean {(d.max() / d.mean()).item():.1f}", flush=True)
