#!/usr/bin/env python3
"""qtf2v9bench.py SRC OUT.jsonl -- v9 (QTF2_HIOPT: inlined forward, bounded unroll at KOUT >= 6) vs the same source
without it, at KA 5-7 (every instance) and integer K6 / K7: outputs must be identical; tiles / s of both (i8, batch
128 as the ladder; plus mul1 identity). Each build in a child."""
import json, os, subprocess, sys, time
SRC, OUT = sys.argv[1:3]
HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("V9_CHILD"):
    import torch
    sys.path.insert(0, HERE); import exl3frac2
    v, out = os.environ["V9_CB"], os.environ["V9_OUT"]; ext = exl3frac2.load(SRC, v)
    torch.manual_seed(0); d = torch.device("cuda", 0)
    T = torch.cat([torch.randn(1024, 256, device=d), torch.distributions.Laplace(0., 0.7).sample((256, 256)).to(d),
                   torch.round(torch.randn(128, 256, device=d) * 2) / 2, torch.randn(8, 256, device=d) * 12]).contiguous()
    res = {}
    for ka, m in ext.qtf2_instances():
        if ka < 5: continue
        em = 65536 >> ka
        tc = torch.empty(128, 2, em, dtype=torch.half, device=d); te = torch.empty(128, 256, em, dtype=torch.short, device=d)
        oq, oi = torch.zeros_like(T), torch.zeros_like(T, dtype=torch.short)
        ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, m); torch.cuda.synchronize()
        t = time.perf_counter(); ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, m); torch.cuda.synchronize()
        res[(ka, m)] = (oq.cpu(), oi.cpu(), len(T) / (time.perf_counter() - t))
    torch.save(res, out); sys.exit(0)
import torch
fails = 0
for v in ("i8", "mul1"):
    R = {}
    for name, defs in (("v8", ""), ("v9", "QTF2_HIOPT")):
        o = f"/tmp/v9-{name}-{v}.pt"
        subprocess.run([sys.executable, "-I", os.path.abspath(__file__), SRC, OUT], check=True,
                       env=dict(os.environ, V9_CHILD="1", V9_CB=v, V9_OUT=o, QTF2_DEFS=defs))
        R[name] = torch.load(o)
    for k in R["v8"]:
        a, b = R["v8"][k], R["v9"][k]
        di, dv = int((a[1] != b[1]).sum()), int((a[0] != b[0]).sum()); fails += di + dv
        r = {"cb": v, "ka": k[0], "mask": hex(k[1]), "K": k[0] + bin(k[1]).count("1") / 16, "diff": di + dv,
             "v8_tps": round(a[2]), "v9_tps": round(b[2])}
        print(json.dumps(r), flush=True); open(OUT, "a").write(json.dumps(r) + "\n")
print(json.dumps({"v9_summary": "PASS" if fails == 0 else "FAIL", "fails": fails}), flush=True)
