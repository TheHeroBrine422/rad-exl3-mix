"""qtf2i4check.py QTF2DIR OUTDIR -- the qtf2 i4 build (w53's script, 11-06; copied into pipeline/tests by w76 with the
paths from the environment: RXQ_RXM = the rad-exl3-mix tree (kernels), RXQ_EXL3_SRC = the fork's exllamav3_ext; QTF2DIR
= <RXQ_RXM>/v3/e3/exl3frac2 now that qtf2-i4.patch is applied there).
  1. exactness, as qtf2check.py: old kernel (exl3cb "i4": quantize_tiles_frac at the fork's quarter / half rates,
     quantize_tiles at integer K1-4)
     vs qtf2 i4 (MASK 0 for integer K, as exl3frac2.install routes them) on the same tiles: indices and values identical
  2. every qtf2 rate (1/8 steps, sub-1 included): pack -> unpack exact, values on the int4 grid; packed / values / indices
     dumped to OUTDIR/k<K>.{packed,vals,idx}.bin for tilecheck (radiance's oracle; sub-1: the salted 0-bit steps)
One JSON line per rate; the last line is the summary."""
import json, os, sys
import torch

QD, OUT = sys.argv[1], sys.argv[2]
R = os.environ.get("RXQ_RXM", "/workspace/rad-exl3-mix")
sys.path[:0] = [QD, f"{R}/v3/e3/exl3frac", f"{R}/v3/e3/exl3cb"]
import exl3cb, exl3frac2
SRC = os.environ.get("RXQ_EXL3_SRC", "/data/src/exllamav3/exllamav3/exllamav3_ext")
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda", 0)
torch.manual_seed(1)
N = 2048
tiles = torch.cat([torch.randn(N, 256, device=dev), torch.randn(N // 4, 256, device=dev) * 3,
                   torch.round(torch.randn(N // 8, 256, device=dev) * 2) / 2, torch.zeros(8, 256, device=dev)]).contiguous()
OLD, NEW = exl3cb.load(SRC, "i4"), exl3frac2.load(SRC, "i4")
ST = 2.6 / 7.5


def run(ext, t, ka, mask, frac=True):
    e = 65536 >> ka
    costs = torch.zeros((128, 2, e), dtype=torch.half, device=dev); edges = torch.zeros((128, 256, e), dtype=torch.short, device=dev)
    out = torch.zeros_like(t); idx = torch.zeros_like(t, dtype=torch.short)
    ext.quantize_tiles_frac(t, out, idx, costs, edges, ka, mask)
    torch.cuda.synchronize()
    return out, idx


def run_int(t, K):   # exllamav3's integer kernel, i4 build (what exl3cb.install routes integer K to)
    import exllamav3.modules.quant.exl3_lib.quantize as Q
    out, idx = torch.zeros_like(t), torch.zeros_like(t, dtype=torch.short)
    temp_costs, temp_edges = Q.get_temp_buffers(t.device, K, 256, 2)
    OLD.quantize_tiles(t, out, idx, temp_costs, temp_edges, K, False, True)
    torch.cuda.synchronize()
    return out, idx


def on_grid(v):
    q = v / ST - 0.5
    return bool(((q - q.round()).abs() < 1e-2).all() and q.round().min() >= -8 and q.round().max() <= 7)


import exllamav3.modules.quant.exl3_lib.quantize as Q
Q.ext = exl3cb._Ext(Q.ext._base if hasattr(Q.ext, "_base") else Q.ext, OLD)   # scratch sizing from the i4 build
fails = 0
mask = exl3frac2.frac_mask
OLD_RATES = [(ka, m) for ka in (1, 2, 3, 4) for m in (0x8888, 0xAAAA, 0xEEEE)] + [(ka, 0xAAAA) for ka in (5, 6, 7)]   # the fork's
for ka in range(0, 8):
    for n in range(2, 16, 2):
        m = mask(n); K = ka + n / 16
        if (ka, m) not in [tuple(x) for x in NEW.qtf2_instances()]:
            continue
        n_out, n_idx = run(NEW, tiles, ka, m)
        r = {"K": K, "grid": on_grid(n_out)}
        fails += not r["grid"]
        if (ka, m) in OLD_RATES:
            o_out, o_idx = run(OLD, tiles, ka, m)
            r["idx_diff"] = int((o_idx != n_idx).sum()); r["val_diff"] = int((o_out != n_out).sum())
            fails += r["idx_diff"] + r["val_diff"]
        if True:   # every qtf2 rate dumped for tilecheck (the ladder's kernel); sub-1 also round-trips pack / unpack
            bpb = 16 * ka + n
            packed = torch.zeros((1, len(tiles), bpb), dtype=torch.short, device=dev)
            back = torch.zeros_like(n_idx).view(1, len(tiles), 256)
            NEW.pack_trellis_frac(packed, n_idx.view(1, len(tiles), 256).contiguous(), ka, m)
            NEW.unpack_trellis_frac(back, packed, ka, m); torch.cuda.synchronize()
            r["pack_roundtrip_diff"] = int((back.view_as(n_idx) != n_idx).sum()); fails += r["pack_roundtrip_diff"]
            tag = f"k{K:g}"
            packed.cpu().numpy().tofile(f"{OUT}/{tag}.packed.bin")
            n_out.half().view(torch.int16).cpu().numpy().tofile(f"{OUT}/{tag}.vals.bin")
            n_idx.cpu().numpy().tofile(f"{OUT}/{tag}.idx.bin")
            r.update(dump=tag, m=bpb, tiles=len(tiles))
        r["rel_mse"] = round(float(((n_out - tiles) ** 2).mean() / (tiles ** 2).mean()), 5)
        print(json.dumps(r), flush=True)
for K in (1, 2, 3, 4):   # integer K: qtf2 MASK 0 (install's route) vs the integer kernel
    o_out, o_idx = run_int(tiles, K); n_out, n_idx = run(NEW, tiles, K, 0)
    r = {"K": K, "route": "int", "idx_diff": int((o_idx != n_idx).sum()), "val_diff": int((o_out != n_out).sum()), "grid": on_grid(n_out)}
    fails += r["idx_diff"] + r["val_diff"] + (not r["grid"])
    print(json.dumps(r), flush=True)
print(json.dumps({"summary": "qtf2 i4", "fails": fails, "verdict": "PASS" if fails == 0 else "FAIL"}), flush=True)
