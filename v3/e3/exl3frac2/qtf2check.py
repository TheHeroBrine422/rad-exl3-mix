#!/usr/bin/env python3
"""qtf2check.py SRC [N] -- the V3 kernel pass's gate for qtf2 (GPU box, one card).
  1. exactness: for every rate the old kernel has (fork's quantize_tiles_frac: exl3frac = plain mul1, exl3cb_i8 =
     int8-rounded), old vs qtf2 on the same N tiles (Gaussian, heavy-tailed, ties, zeros, large): indices and values
     must be identical, both codebooks. Any difference = FAIL (the pass is output-identical by rule).
  2. speed: tiles / s for old, qtf2 and the integer kernel at the neighbouring K (exllamav3's quantize_tiles).
  3. every rate in qtf2 (1/8 steps, sub-1 included): pack -> unpack round trip is exact, the trellis structure holds (window overlap: state(i) >> D(i) == state(i - 1) &
     (2^(16 - D(i)) - 1), around the ring) and the error sits on the curve between its neighbours (a salt mismatch
     between search and traceback would show as a jump in rel_mse).
Env as exl3frac2 (PYTORCH_ROCM_ARCH, TORCH_EXTENSIONS_DIR, ROCM_HOME). Writes one JSON line per rate to stdout."""
import json, math, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "exl3frac"), os.path.join(HERE, "..", "exl3cb")]
import exl3frac, exl3cb, exl3frac2
import exllamav3.modules.quant.exl3_lib.quantize as Q

SRC = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
dev = torch.device("cuda", 0)
torch.manual_seed(0)
g = torch.randn(N, 256, device=dev)
tiles = torch.cat([g, torch.distributions.Laplace(0., 0.7).sample((N // 4, 256)).to(dev),
                   torch.round(torch.randn(N // 8, 256, device=dev) * 2) / 2,          # many exact ties
                   torch.zeros(8, 256, device=dev), torch.full((8, 256), 3.5, device=dev),
                   torch.randn(8, 256, device=dev) * 12])                              # far outside the codebook
tiles = tiles.contiguous()
OLD = {"mul1": exl3frac.load(SRC), "i8": exl3cb.load(SRC, "i8")}
NEW = {"mul1": exl3frac2.load(SRC, "mul1"), "i8": exl3frac2.load(SRC, "i8")}
OLD_RATES = [(ka, m) for ka in (1, 2, 3, 4) for m in (0x8888, 0xAAAA, 0xEEEE)] + [(ka, 0xAAAA) for ka in (5, 6, 7)]

def run(ext, t, ka, mask):
    e = 65536 >> ka
    costs = torch.zeros((128, 2, e), dtype=torch.half, device=dev); edges = torch.zeros((128, 256, e), dtype=torch.short, device=dev)
    out = torch.zeros_like(t); idx = torch.zeros_like(t, dtype=torch.short)
    ext.quantize_tiles_frac(t, out, idx, costs, edges, ka, mask)
    torch.cuda.synchronize(); return out, idx, (costs, edges)

def tput(fn, n):
    fn(); torch.cuda.synchronize(); t0 = time.time(); fn(); torch.cuda.synchronize(); return n / (time.time() - t0)

def rate(ka, mask): return ka + bin(mask).count("1") / 16
fails = 0
for v in ("mul1", "i8"):
    for ka, m in OLD_RATES:
        o_out, o_idx, _ = run(OLD[v], tiles, ka, m); n_out, n_idx, _ = run(NEW[v], tiles, ka, m)
        di = int((o_idx != n_idx).sum()); dv = int((o_out != n_out).sum()); fails += di + dv
        r = {"check": "exact", "cb": v, "K": rate(ka, m), "idx_diff": di, "val_diff": dv, "tiles": len(tiles)}
        if v == "i8":
            T = tiles[:1024].contiguous()
            r["old_tps"] = round(tput(lambda: run(OLD[v], T, ka, m), len(T)))
            r["new_tps"] = round(tput(lambda: run(NEW[v], T, ka, m), len(T)))
        print(json.dumps(r), flush=True)
# integer kernel reference speed (exllamav3's, i8 codebook, the ladder's)
exl3cb.install(SRC, "i8")
for K in range(1, 8):
    T = tiles[:1024].contiguous()
    print(json.dumps({"check": "int_speed", "K": K, "tps": round(tput(lambda: Q.quantize_tiles(T, {"K": K, "mul1": True}), len(T)))}), flush=True)
# new rates: structure, salt, error on the curve
def D(i, ka, m): return ka + ((m >> (i % 16)) & 1)
NEWR = [tuple(x) for x in NEW["i8"].qtf2_instances()]
G = tiles[:N].contiguous()
for ka, m in NEWR:
    out, idx, _ = run(NEW["i8"], G, ka, m)
    s = idx.to(torch.int64) & 0xFFFF
    bad = 0
    for i in range(256):
        d = D(i, ka, m); p = (i - 1) % 256
        bad += int(((s[:, i] >> d) != (s[:, p] & ((1 << (16 - d)) - 1))).sum())
    bpb = 16 * ka + bin(m).count("1")
    packed = torch.zeros((1, len(G), bpb), dtype=torch.short, device=dev); back = torch.zeros_like(idx).view(1, len(G), 256)
    NEW["i8"].pack_trellis_frac(packed, idx.view(1, len(G), 256).contiguous(), ka, m)
    NEW["i8"].unpack_trellis_frac(back, packed, ka, m); torch.cuda.synchronize()
    rt = int((back.view_as(idx) != idx).sum()); bad += rt
    mse = float(((out - G) ** 2).mean() / (G ** 2).mean())
    old = (ka, m) in OLD_RATES
    print(json.dumps({"check": "rate", "K": rate(ka, m), "ka": ka, "mask": f"{m:04x}", "struct_violations": bad - rt, "pack_roundtrip_diff": rt,
                      "rel_mse": round(mse, 5), "has_old": old,
                      "tps": round(tput(lambda: run(NEW["i8"], G[:1024].contiguous(), ka, m), 1024))}), flush=True)
    fails += bad
print(json.dumps({"check": "summary", "fails": fails, "verdict": "PASS" if fails == 0 else "FAIL"}), flush=True)
