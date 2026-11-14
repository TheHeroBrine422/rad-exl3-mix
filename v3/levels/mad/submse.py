# submse.py VARIANT SRC E3 [K ...] -- #177-03 (w151): random tiles through exllamav3's quantize_tiles with VARIANT's build
# (i8: exl3cb + exl3frac2 i8; mad: v3/e3/exl3hash) from the kernel dirs under E3; MSE per K. Regression for the KA = 0
# byte-sum table bug (11-14): a working mad build has i8's MSE at sub-1 K (the broken one ~2.0, worse than zeros)
import os, sys, torch
E3 = sys.argv[3]; sys.path.insert(0, f"{E3}/exl3frac"); sys.path.insert(0, f"{E3}/exl3hash"); sys.path.insert(0, f"{E3}/exl3cb"); sys.path.insert(0, f"{E3}/exl3frac2")
import exl3frac
v, src = sys.argv[1], sys.argv[2]; Ks = [float(k) for k in sys.argv[4:]]
exl3frac.install(src)
if v == "mad":
    import exl3hash; Q = exl3hash.install(src, True, a=0xD0474B, b=0x2545F491); ref = torch.from_numpy(exl3hash.values())
else:
    import exl3cb, exl3frac2; exl3cb.install(src, "i8"); Q = exl3frac2.install(src, "i8"); ref = None
torch.manual_seed(0); tiles = torch.randn(512, 256, device="cuda")
for K in Ks:
    Kq = int(K) if K.is_integer() else K
    out, idx = Q.quantize_tiles(tiles, {"K": Kq, "mul1": True})
    mse = (out - tiles).pow(2).mean().item()
    print(f"{v} K{K:g}: mse {mse:.5f}", flush=True)
