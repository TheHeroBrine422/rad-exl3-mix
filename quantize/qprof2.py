#!/usr/bin/env python3
"""qprof2.py K LAYER OUT.json -- like qprof.py, but every function defined in exllamav3's quantize module (and the
torch.linalg / tensor-transfer calls it makes) gets a synchronized inclusive timer, to find the time qprof.py left
unaccounted. Nested calls are inclusive; "self" = inclusive minus wrapped children."""
import json, sys, time, shutil, collections, inspect
import torch
import exllamav3.modules.quant.exl3_lib.quantize as Q
K, L, out = sys.argv[1], sys.argv[2], sys.argv[3]
tot = collections.Counter(); child = collections.Counter(); cnt = collections.Counter(); stack = []
def wrap(owner, name, label):
    f = getattr(owner, name)
    def w(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); stack.append(0.0)
        try: return f(*a, **k)
        finally:
            torch.cuda.synchronize(); dt = time.perf_counter() - t; ch = stack.pop()
            tot[label] += dt; child[label] += ch; cnt[label] += 1
            if stack: stack[-1] += dt
    setattr(owner, name, w)
for n, f in list(vars(Q).items()):
    if inspect.isfunction(f) and f.__module__ == Q.__name__: wrap(Q, n, n)
for n in ["eigh", "cholesky", "inv", "solve", "qr", "svd"]:
    if hasattr(torch.linalg, n): wrap(torch.linalg, n, "torch.linalg." + n)
for n in ["cholesky_inverse", "cholesky", "linalg_cholesky"]:
    if hasattr(torch, n): wrap(torch, n, "torch." + n)
shutil.rmtree(f"/tmp/qprof-K{K}", ignore_errors=True)
sys.argv = ["ourq.py", "-K", K, "-o", f"/tmp/qprof-K{K}", "--layers", f"{L}-{L}"]
src = open("ourq/ourq.py").read().replace("from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3",
          "quantize_exl3 = __import__('exllamav3.modules.quant.exl3_lib.quantize', fromlist=['x']).quantize_exl3")
t0 = time.perf_counter()
exec(compile(src, "ourq/ourq.py", "exec"), {"__name__": "__main__"})
wall = time.perf_counter() - t0
rows = sorted(tot, key=lambda n: -(tot[n] - child[n]))
rep = {"K": int(K), "wall_s": round(wall, 1), "stages": {n: {"incl": round(tot[n], 2), "self": round(tot[n] - child[n], 2), "calls": cnt[n]} for n in rows}}
json.dump(rep, open(out, "w"), indent=1)
print(f"wall {wall:.1f}s");
for n in rows[:25]: print(f"  {n:40} self {tot[n]-child[n]:7.2f}s  incl {tot[n]:7.2f}s  calls {cnt[n]}")
