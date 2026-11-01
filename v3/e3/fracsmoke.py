#!/usr/bin/env python3
"""fracsmoke.py [NTILES] [NCPU] -- check of the quarter-step quantizer (exllamav3 branch rxm-frac-quarter) on the GPU.
Env EXL3FRAC_SRC = the fork's exllamav3_ext source dir, PYTORCH_ROCM_ARCH; see exl3frac/exl3frac.py.

The same seeded unit-Gaussian 256-value tiles are quantized at every V3 menu rate (integer 1-8, quarter steps to 5,
half steps to 8). Per K:
  mse      GPU trellis error (quantize_tiles: the integer or the fractional kernel)
  interp   mse / log-linear interpolation of the integer neighbours (fractional K only; the CPU sweep put real tensors
           at 0.99-1.00)
  cpu      mse of the CPU Viterbi (trellis.py vit, viterbi_tiles_pat) on the first NCPU tiles / GPU mse on the same
           tiles: both are exact Viterbi on the mul1 codebook, so ~1 (tail-biting and fp16 costs differ slightly)
  pack     indices -> pack_trellis -> unpack -> identical (the round trip the format depends on)
  dec      returned values vs the mul1 codebook at the returned indices (correlation)
Exit 1 if pack fails anywhere, dec < 0.9999, a fractional K's cpu/gpu differs > 2% from its integer
neighbours', or interp outside 0.95-1.05 (K <= 5; above, the GPU's integer K8 sits on an fp16 cost floor)."""
import math, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trellis as T
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3frac"))
import exl3frac
Q = exl3frac.install(os.environ["EXL3FRAC_SRC"])     # the fork's exllamav3_ext sources
ext = Q.ext
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles, pack_trellis, trellis_words
frac_k = Q.frac_k

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 256
MENU = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.5, 6, 6.5, 7, 7.5, 8]
dev = torch.device("cuda:0")
x = torch.from_numpy(np.random.default_rng(7).standard_normal((N, 256)).astype(np.float32)).to(dev)
cb = np.ascontiguousarray(T.codebook("mul1"), dtype=np.float32)
res, bad, CPU = {}, [], {}
for K in MENU:
    qa = {"K": K, "mul1": True}
    torch.cuda.synchronize(); t0 = time.time()
    q, idx = quantize_tiles(x, qa)
    torch.cuda.synchronize(); dt = time.time() - t0
    mse = float(((q - x) ** 2).mean())
    packed = pack_trellis(idx.view(1, N, 256), qa)
    assert packed.shape[-1] == trellis_words(K)
    un = torch.zeros_like(idx.view(1, N, 256))
    fk = frac_k(K)
    if fk: ext.unpack_trellis_frac(un, packed, fk[0], fk[1])
    else: ext.unpack_trellis(un, packed, K)
    pack_ok = bool(torch.equal(un, idx.view(1, N, 256)))
    iv = idx.cpu().numpy().view(np.uint16).astype(np.int64).ravel()
    dec = float(np.corrcoef(q.cpu().numpy().ravel(), cb[iv])[0, 1])
    xc = x[:NC].cpu().numpy()
    cpu = float(((T.vit(xc, cb, K) - xc) ** 2).mean()) / float(((q[:NC] - x[:NC]) ** 2).mean())
    res[K] = mse
    print(f"K{K:<5} words {trellis_words(K):3d}  mse {mse:.4e}  cpu/gpu {cpu:.3f}  pack {'ok' if pack_ok else 'FAIL'}  "
          f"dec {dec:.6f}  {dt:.2f}s", flush=True)
    if not pack_ok: bad.append(f"K{K} pack")
    if dec < 0.9999: bad.append(f"K{K} dec {dec:.6f}")
    CPU[K] = cpu
# cpu/gpu: the GPU's fp16-cost search is a few % worse than the CPU reference at EVERY K (integer K = exllamav3's
# own kernel), so a fractional K is judged against its integer neighbours' offset (2026-10-31: 0.94-0.99 for all K)
for K in MENU:
    if float(K).is_integer(): continue
    a, b = math.floor(K), math.ceil(K); off = (CPU[a] + CPU[b]) / 2
    if abs(CPU[K] / off - 1) > 0.02: bad.append(f"K{K} cpu/gpu {CPU[K]:.3f} vs integer {off:.3f}")
for K in MENU:
    if float(K).is_integer(): continue
    a, b = math.floor(K), math.ceil(K)
    r = res[K] / math.exp(math.log(res[a]) + (K - a) * (math.log(res[b]) - math.log(res[a])))
    print(f"K{K:<5} vs log-linear interpolation of K{a}/K{b}: {r:.3f}")
    if K < 5 and not 0.95 <= r <= 1.05: bad.append(f"K{K} interp {r:.3f}")
print("FRACSMOKE " + ("OK" if not bad else "FAIL: " + ", ".join(bad)))
sys.exit(1 if bad else 0)
