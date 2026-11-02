#!/usr/bin/env python3
"""dumpvec.py SRC OUT.npz -- test vectors for the exl3i oracle from the GPU quantizer (qtf2, both codebooks): per rate,
64 random tiles -> quantizer indices (ring order), output values (f32) and the packed tile words (pack_trellis_frac,
or exllamav3's pack_trellis for integer K)."""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = ["/root/ourq/exl3frac2", "/root/ourq/exl3frac", "/root/ourq/exl3cb"]
import exl3frac2, exl3cb
SRC, OUT = sys.argv[1:3]
dev = torch.device("cuda", 0); torch.manual_seed(1)
tiles = torch.randn(64, 256, device=dev)
res = {}
for v in ("mul1", "i8"):
    q = exl3frac2.load(SRC, v)
    for ka, m in [tuple(x) for x in q.qtf2_instances()]:
        e = 65536 >> ka
        costs = torch.zeros((128, 2, e), dtype=torch.half, device=dev); edges = torch.zeros((128, 256, e), dtype=torch.short, device=dev)
        out = torch.zeros_like(tiles); idx = torch.zeros_like(tiles, dtype=torch.short)
        q.quantize_tiles_frac(tiles, out, idx, costs, edges, ka, m)
        bpb = 16 * ka + bin(m).count("1")
        packed = torch.zeros((1, 64, bpb), dtype=torch.short, device=dev)
        q.pack_trellis_frac(packed, idx.view(1, 64, 256).contiguous(), ka, m); torch.cuda.synchronize()
        K = ka + bin(m).count("1") / 16
        res[f"{v}_{K:g}_idx"] = idx.cpu().numpy().view(np.uint16); res[f"{v}_{K:g}_out"] = out.cpu().numpy()
        res[f"{v}_{K:g}_packed"] = packed.cpu().numpy().view(np.uint16).reshape(64, bpb); res[f"{v}_{K:g}_ka_mask"] = np.array([ka, m])
np.savez_compressed(OUT, **res); print(len(res) // 4, "rate x codebook sets")
