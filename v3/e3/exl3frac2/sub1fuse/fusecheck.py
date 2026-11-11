#!/usr/bin/env python3
"""fusecheck.py -- #142 judge (a): qtf2 with the fused KA = 0 forward pass (v10) vs today's kernel, bit for bit.

  fusecheck.py run TREE SRC VARIANT OUT     (child) load exl3frac2 from TREE (its own build dir), run every instance on
                                            the tile set, save {(ka, mask, lut, batch): (values, indices)} + rerun digests
  fusecheck.py cmp REF.pt NEW.pt            compare: one JSON line per (instance, read path, batch), then a summary

Tile set (SUB1FUSE.md): qtf2check's (seed 0, N = 1024: Gaussian, Laplace(0.7) N/4, half-integer ties N/8, 8 zero,
8 x 3.5, 8 x 12) + the fallback second set (seed 1: 512 Gaussian x 0.5, 512 x 2). KA = 0 instances: both 0-bit read
paths (QTF2_LUT = 1 / 0) at launch batches 32 and 128, plus 3 more runs (LUT, batch 32) for determinism; KA >= 1: batch
128 (regression). Env as exl3frac2 (PYTORCH_ROCM_ARCH, QTF2_DEFS); W108_STUB = a dir holding a stub exllamav3 package
with util/cuda_flags.py only (exl3frac2.load needs nothing else)."""
import hashlib, json, os, sys


def tiles(torch, dev, N=1024):
    torch.manual_seed(0)
    g = torch.randn(N, 256, device=dev)
    t = torch.cat([g, torch.distributions.Laplace(0., 0.7).sample((N // 4, 256)).to(dev),
                   torch.round(torch.randn(N // 8, 256, device=dev) * 2) / 2,
                   torch.zeros(8, 256, device=dev), torch.full((8, 256), 3.5, device=dev),
                   torch.randn(8, 256, device=dev) * 12])
    torch.manual_seed(1)
    t2 = torch.cat([torch.randn(512, 256, device=dev) * 0.5, torch.randn(512, 256, device=dev) * 2])
    return torch.cat([t, t2]).contiguous()


def run(torch, ext, T, ka, mask, bs):
    em = 65536 >> ka
    oq, oi = torch.zeros_like(T), torch.zeros_like(T, dtype=torch.short)
    tc = torch.empty(bs, 2, em, dtype=torch.half, device=T.device)
    te = torch.empty(bs, 256, em, dtype=torch.short, device=T.device)
    ext.quantize_tiles_frac(T, oq, oi, tc, te, ka, mask)
    torch.cuda.synchronize()
    return oq.cpu(), oi.cpu()


def digest(q, i):
    return hashlib.sha256(q.numpy().tobytes() + i.numpy().tobytes()).hexdigest()[:16]


def child(tree, src, variant, out):
    sys.path[:0] = [os.environ["W108_STUB"], tree]
    import torch
    import exl3frac2
    ext = exl3frac2.load(src, variant)
    dev = torch.device("cuda", 0)
    T = tiles(torch, dev)
    res, dig = {}, {}
    for ka, m in ext.qtf2_instances():
        if ka == 0:
            for lut in ("1", "0"):
                os.environ["QTF2_LUT"] = lut
                for bs in (32, 128):
                    res[(ka, m, lut, bs)] = run(torch, ext, T, ka, m, bs)
            os.environ["QTF2_LUT"] = "1"
            dig[(ka, m)] = [digest(*run(torch, ext, T, ka, m, 32)) for _ in range(3)]
        else:
            res[(ka, m, "1", 128)] = run(torch, ext, T, ka, m, 128)
        print(json.dumps({"done": [ka, hex(m)]}), flush=True)
    torch.save({"res": res, "dig": dig, "tiles": len(T)}, out)


def cmp(ref_p, new_p):
    import torch
    R, N = torch.load(ref_p), torch.load(new_p)
    fails = 0
    for key in sorted(R["res"]):
        rq, ri = R["res"][key]; nq, ni = N["res"][key]
        di = int((ri != ni).sum()); dv = int((rq != nq).sum())
        dt = int(((ri != ni) | (rq != nq)).any(1).sum())
        fails += di + dv
        ka, m, lut, bs = key
        print(json.dumps({"K": ka + bin(m).count("1") / 16, "ka": ka, "mask": f"{m:04x}", "lut": lut, "batch": bs,
                          "idx_diff": di, "val_diff": dv, "tiles_diff": dt, "tiles": R["tiles"]}), flush=True)
    for (ka, m), ds in sorted(N["dig"].items()):
        first = digest(*N["res"][(ka, m, "1", 32)])
        bad = sum(d != first for d in ds)
        fails += bad
        print(json.dumps({"check": "determinism", "K": ka + bin(m).count("1") / 16, "runs": 1 + len(ds), "differing_runs": bad}), flush=True)
    print(json.dumps({"check": "summary", "fails": fails, "verdict": "PASS" if fails == 0 else "FAIL"}), flush=True)
    return fails


if __name__ == "__main__":
    if sys.argv[1] == "run": child(*sys.argv[2:6])
    else: sys.exit(1 if cmp(sys.argv[2], sys.argv[3]) else 0)
