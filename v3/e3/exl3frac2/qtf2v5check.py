#!/usr/bin/env python3
"""qtf2v5check.py SRC N -- v5 (packed history + LDS-only step barrier) vs the v4 behaviour built from the same source
(QTF2_BYTEHIST QTF2_FULLSYNC), plus the single-change builds (QTF2_FULLSYNC: packing only; QTF2_BYTEHIST: barrier
only) for the speed split, every
(KA, MASK) instance (1/8 steps and sub-1 included), both codebooks: indices and values must equal the reference's
(any difference = FAIL); plus tiles / s of every build (i8) at the ladder's batch (128 tiles a launch). Each build runs
in a child (one extension name per build). One JSON line per instance."""
import json, os, subprocess, sys, time
SRC, N = sys.argv[1], int(sys.argv[2])
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3frac2
def tiles():
    torch.manual_seed(0); d = torch.device("cuda", 0); g = torch.randn(N, 256, device=d)
    return torch.cat([g, torch.distributions.Laplace(0., 0.7).sample((N // 4, 256)).to(d),
                      torch.round(torch.randn(N // 8, 256, device=d) * 2) / 2, torch.zeros(8, 256, device=d),
                      torch.full((8, 256), 3.5, device=d), torch.randn(8, 256, device=d) * 12]).contiguous()
def run(ext, T, ka, mask, bs=128):
    em = 65536 >> ka; bs = min(bs, 32 if ka == 0 else bs)
    oq, oi = torch.zeros_like(T), torch.zeros_like(T, dtype=torch.short)
    tc = torch.empty(bs, 2, em, dtype=torch.half, device=T.device); te = torch.empty(bs, 256, em, dtype=torch.short, device=T.device)
    ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask); return oq, oi, (tc, te)
def tput(fn, n):
    fn(); torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); return n / (time.perf_counter() - t)
if os.environ.get("V5_CHILD"):             # byte build: dump outputs + speed
    v, out = sys.argv[3], sys.argv[4]; ext = exl3frac2.load(SRC, v); T = tiles(); res = {}
    for ka, m in ext.qtf2_instances():
        q, i, _ = run(ext, T, ka, m)
        res[(ka, m)] = (q.cpu(), i.cpu(), tput(lambda: run(ext, T[:1024].contiguous(), ka, m), 1024) if v == "i8" else None)
    torch.save(res, out); sys.exit(0)
fails = 0
BUILDS = {"ref": "QTF2_BYTEHIST QTF2_FULLSYNC QTF2_SCANMIN"} if os.environ.get("V6") else \
         {"ref": "QTF2_BYTEHIST QTF2_FULLSYNC", "pack_only": "QTF2_FULLSYNC", "bar_only": "QTF2_BYTEHIST"}
for v in (("i8",) if os.environ.get("V5_I8ONLY") else ("mul1", "i8")):
    res = {}
    for name, defs in BUILDS.items():
        if v == "mul1" and name != "ref": continue
        out = f"/tmp/qtf2v5-{name}-{v}.pt"
        subprocess.run([sys.executable, "-I", os.path.abspath(__file__), SRC, str(N), v, out],
                       env=dict(os.environ, V5_CHILD="1", QTF2_DEFS=defs), check=True)
        res[name] = torch.load(out)
    ref = res["ref"]; ext = exl3frac2.load(SRC, v); T = tiles()
    for ka, m in ext.qtf2_instances():
        q, i, _ = run(ext, T, ka, m); rq, ri, rt = ref[(ka, m)]
        di = int((i.cpu() != ri).sum()); dv = int((q.cpu() != rq).sum())
        for name in res:
            if name != "ref": di += int((res[name][(ka, m)][1] != ri).sum()); dv += int((res[name][(ka, m)][0] != rq).sum())
        fails += di + dv
        r = {"check": "v5_exact", "cb": v, "ka": ka, "mask": hex(m), "K": ka + bin(m).count("1") / 16, "idx_diff": di, "val_diff": dv}
        if v == "i8":
            r.update({f"{n}_tps": round(res[n][(ka, m)][2]) for n in res})
            r["v5_tps"] = round(tput(lambda: run(ext, T[:1024].contiguous(), ka, m), 1024))
        print(json.dumps(r), flush=True)
print(json.dumps({"check": "v5_summary", "fails": fails, "result": "PASS" if fails == 0 else "FAIL"}), flush=True)
