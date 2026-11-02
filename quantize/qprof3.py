#!/usr/bin/env python3
"""qprof3.py K LAYERS OUT.json [ourq args...] -- stage profile of one ourq run with the real ladder settings (run from
RXM_WORK, beside ourq/). Every function of exllamav3's quantize module, torch.linalg's factorizations and every call into
the quantizer extension (after ourq's installs: exl3cb / exl3frac / exl3frac2) get a synchronized inclusive timer;
"self" = inclusive minus wrapped children. Extension calls also record their tile counts. Wall time and the sum of
extension (kernel) time give the share left to the host and the torch glue."""
import json, sys, time, shutil, collections, inspect
import torch
import exllamav3.modules.quant.exl3_lib.quantize as Q
K, LAY, out, *extra = sys.argv[1:]
tot = collections.Counter(); child = collections.Counter(); cnt = collections.Counter(); stack = []
tiles = collections.defaultdict(collections.Counter)

def timed(f, label, ntiles=False):
    def w(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); stack.append(0.0)
        try: return f(*a, **k)
        finally:
            torch.cuda.synchronize(); dt = time.perf_counter() - t; ch = stack.pop()
            tot[label] += dt; child[label] += ch; cnt[label] += 1
            if ntiles and a and hasattr(a[0], "shape"): tiles[label][int(a[0].shape[0])] += 1
            if stack: stack[-1] += dt
    return w

import os
PLAIN = os.environ.get("QPROF_PLAIN") == "1"     # no timers: true wall time (the timers sync after every call)
for n, f in ([] if PLAIN else list(vars(Q).items())):
    if inspect.isfunction(f) and f.__module__ == Q.__name__: setattr(Q, n, timed(f, n))
for n in ([] if PLAIN else ["eigh", "cholesky", "inv", "solve", "qr", "svd"]):
    if hasattr(torch.linalg, n): setattr(torch.linalg, n, timed(getattr(torch.linalg, n), "torch.linalg." + n))

class Proxy:
    def __init__(self, base): self._base = base
    def __getattr__(self, name):
        v = getattr(self._base, name)
        return timed(v, "ext." + name, ntiles=name.startswith("quantize_tiles") and "scratch" not in name) if callable(v) else v

def hook():                     # after ourq's extension installs, before any quantization
    if not PLAIN: Q.ext = Proxy(Q.ext)

o = f"/tmp/qprof3-K{K}-L{LAY}"
shutil.rmtree(o, ignore_errors=True)
sys.argv = ["ourq.py", "-K", K, "-o", o, "--layers", LAY, "--device", "0"] + extra
src = open("ourq/ourq.py").read()
a = "from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3"
b = 'dev = torch.device("cuda", a.device)\n'
assert src.count(a) == 1 and src.count(b) == 1
src = src.replace(a, "quantize_exl3 = lambda *x, **k: __import__('exllamav3.modules.quant.exl3_lib.quantize', fromlist=['x']).quantize_exl3(*x, **k)")
src = src.replace(b, b + "__qprof_hook__()\n")
t0 = time.perf_counter(); c0 = os.times()
exec(compile(src, "ourq/ourq.py", "exec"), {"__name__": "__main__", "__qprof_hook__": hook})
wall = time.perf_counter() - t0; c1 = os.times()
cpu_u, cpu_s = c1.user - c0.user, c1.system - c0.system   # all threads of this process
rows = sorted(tot, key=lambda n: -(tot[n] - child[n]))
ext_s = sum(tot[n] for n in tot if n.startswith("ext."))
rep = {"plain": PLAIN, "K": K, "layers": LAY, "args": extra, "wall_s": round(wall, 1), "cpu_user_s": round(cpu_u, 1), "cpu_sys_s": round(cpu_s, 1), "ext_s": round(ext_s, 1),
       "stages": {n: {"incl": round(tot[n], 2), "self": round(tot[n] - child[n], 2), "calls": cnt[n]} for n in rows},
       "tiles_per_call": {n: dict(sorted(c.items())) for n, c in tiles.items()}}
json.dump(rep, open(out, "w"), indent=1)
print(f"K{K} L{LAY}: wall {wall:.1f}s, CPU user {cpu_u:.1f}s sys {cpu_s:.1f}s, extension calls {ext_s:.1f}s ({ext_s / wall:.0%})")
for n in rows[:22]: print(f"  {n:34} self {tot[n] - child[n]:7.2f}s  incl {tot[n]:7.2f}s  calls {cnt[n]}")
for n, c in tiles.items(): print(f"  {n} tiles/call: " + ", ".join(f"{k}x{v}" for k, v in sorted(c.items())[:8]))
