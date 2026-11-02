#!/usr/bin/env python3
"""qtint5check.py SRC N -- integer quantizer (exl3cb i8, optimized kernels, EXL3_QT_OPTIMIZED=1 as the ladder) built
from SRC with the LDS-only step barrier vs the same source with -DQT_FULLSYNC (the old barrier): indices and values at
K1-8 must be identical; tiles / s of both. Each build in a child. One JSON line a K."""
import json, os, subprocess, sys, time
SRC, N = sys.argv[1], int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("INT5_CHILD"):
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "exl3cb")); import exl3cb
    Q = exl3cb.install(SRC, "i8")
    torch.manual_seed(0); d = torch.device("cuda", 0)
    T = torch.cat([torch.randn(N, 256, device=d), torch.distributions.Laplace(0., 0.7).sample((N // 4, 256)).to(d),
                   torch.round(torch.randn(N // 8, 256, device=d) * 2) / 2, torch.randn(8, 256, device=d) * 12]).contiguous()
    res = {}
    for K in range(1, 9):
        q, i = Q.quantize_tiles(T, {"K": K, "mul1": True}); torch.cuda.synchronize()
        T2 = T[:2048].contiguous(); Q.quantize_tiles(T2, {"K": K, "mul1": True}); torch.cuda.synchronize()
        t = time.perf_counter(); Q.quantize_tiles(T2, {"K": K, "mul1": True}); torch.cuda.synchronize()
        res[K] = (q.cpu(), i.cpu(), len(T2) / (time.perf_counter() - t))
    torch.save(res, sys.argv[3]); sys.exit(0)
import torch
R = {}
for name, defs in (("old", "QT_FULLSYNC"), ("new", "")):
    out = f"/tmp/qtint5-{name}.pt"
    subprocess.run([sys.executable, "-I", os.path.abspath(__file__), SRC, str(N), out], check=True,
                   env=dict(os.environ, INT5_CHILD="1", EXL3CB_DEFS=defs, EXL3_QT_OPTIMIZED="1"))
    R[name] = torch.load(out)
fails = 0
for K in range(1, 9):
    (qo, io, to), (qn, inew, tn) = R["old"][K], R["new"][K]
    di, dv = int((io != inew).sum()), int((qo != qn).sum()); fails += di + dv
    print(json.dumps({"check": "int5_exact", "K": K, "idx_diff": di, "val_diff": dv, "old_tps": round(to), "new_tps": round(tn)}), flush=True)
print(json.dumps({"check": "int5_summary", "fails": fails, "result": "PASS" if fails == 0 else "FAIL"}), flush=True)
