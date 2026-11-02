#!/usr/bin/env python3
"""qtf2race.py SRC REPS OUT_PREFIX -- determinism / race check at the rates given by RACE_RATES (default KA 1: masks 0,
0x8080, 0x8888, 0xAAAA): one build (QTF2_DEFS) runs each rate REPS times on fixed tiles; saves the first run, reports
run-to-run differences inside the build, and differences vs OUT_PREFIX-ref.pt when it exists."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3frac2
SRC, REPS, PRE = sys.argv[1], int(sys.argv[2]), sys.argv[3]
name = os.environ.get("QTF2_DEFS", "").replace(" ", "+") or "default"
ext = exl3frac2.load(SRC, "i8")
rates = [tuple(int(x, 0) for x in r.split(":")) for r in os.environ.get("RACE_RATES", "1:0,1:0x8080,1:0x8888,1:0xaaaa").split(",")]
torch.manual_seed(0); d = torch.device("cuda", 0); N = 1024
T = torch.cat([torch.randn(N, 256, device=d), torch.distributions.Laplace(0., 0.7).sample((N // 4, 256)).to(d),
               torch.round(torch.randn(N // 8, 256, device=d) * 2) / 2, torch.zeros(8, 256, device=d),
               torch.full((8, 256), 3.5, device=d), torch.randn(8, 256, device=d) * 12]).contiguous()
ref = torch.load(f"{PRE}-ref.pt") if os.path.exists(f"{PRE}-ref.pt") else None
first = {}
for ka, m in rates:
    em = 65536 >> ka
    tc = torch.empty(128, 2, em, dtype=torch.half, device=d); te = torch.empty(128, 256, em, dtype=torch.short, device=d)
    runs = []
    for _ in range(REPS):
        oq, oi = torch.zeros_like(T), torch.zeros_like(T, dtype=torch.short)
        ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, m); torch.cuda.synchronize(); runs.append((oq.cpu(), oi.cpu()))
    within = [int((r[1] != runs[0][1]).sum()) for r in runs[1:]]
    tiles_bad = sorted({int(t) for r in runs[1:] for t in (r[1] != runs[0][1]).any(1).nonzero().flatten()})[:10]
    vs = [int((r[1] != ref[(ka, m)][1]).sum()) for r in runs] if ref else None
    first[(ka, m)] = runs[0]
    print(json.dumps({"build": name, "ka": ka, "mask": hex(m), "within_build_idx_diff": within, "vs_ref_idx_diff": vs,
                      "tiles_varying": tiles_bad}), flush=True)
if name == "QTF2_BYTEHIST+QTF2_FULLSYNC" and ref is None: torch.save(first, f"{PRE}-ref.pt")
