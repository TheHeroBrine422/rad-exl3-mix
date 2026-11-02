#!/usr/bin/env python3
"""qtf2bench.py SRC N OUT.jsonl -- kernel time split for speed pass 2: tiles/s of the real qtf2 build and of the
timing-only builds QTF2_NOBACK (no traceback chain) and QTF2_NOHIST (no history writes; both: forward compute only),
at the ladder's rates, plus exllamav3's integer kernel (i8 codebook) at K1-6. N random tiles a rate (a multiple of the
resident batch). Run as: one process per build (QTF2_DEFS env), this script spawns them."""
import json, os, subprocess, sys, time
SRC, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
RATES = [0.5, 1.5, 2.125, 2.5, 3.5, 4.5, 5.5]
if os.environ.get("QTF2_CHILD") is None:
    for defs in ["", "QTF2_NOBACK", "QTF2_NOHIST", "QTF2_NOBACK QTF2_NOHIST", "INT"]:
        env = dict(os.environ, QTF2_CHILD="1", QTF2_DEFS="" if defs == "INT" else defs, QTF2_INT="1" if defs == "INT" else "0")
        subprocess.run([sys.executable, "-I", os.path.abspath(__file__), SRC, str(N), OUT], env=env, check=False)
    sys.exit(0)
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3frac2
def tput(fn, n):
    fn(); torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    return n / (time.perf_counter() - t)
g = torch.Generator(device="cuda").manual_seed(0)
T = torch.randn(N, 256, device="cuda", generator=g)
out = open(OUT, "a")
if os.environ["QTF2_INT"] == "1":
    import exllamav3.modules.quant.exl3_lib.quantize as Q
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exl3cb"))
    import exl3cb; exl3cb.install(SRC, "i8")
    for K in (1, 2, 3, 4, 5, 6):
        r = {"build": "int", "K": K, "tps": round(tput(lambda: Q.quantize_tiles(T, {"K": K, "mul1": True}), N))}
        print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n")
    sys.exit(0)
ext = exl3frac2.load(SRC, "i8")
for K in RATES:
    ka, mask = exl3frac2.frac_k(K)
    from exllamav3.modules.quant.exl3_lib import quantize as Q
    em = 65536 >> ka
    oq, oi = torch.zeros_like(T), torch.zeros_like(T, dtype=torch.short)
    bs = min(N, 256 if ka >= 1 else 64)
    tc = torch.empty(bs, 2, em, dtype=torch.half, device="cuda"); te = torch.empty(bs, 256, em, dtype=torch.short, device="cuda")
    r = {"build": os.environ["QTF2_DEFS"] or "real", "K": K, "batch": bs,
         "tps": round(tput(lambda: ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask), N))}
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n")
