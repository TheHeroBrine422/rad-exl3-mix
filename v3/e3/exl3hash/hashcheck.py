#!/usr/bin/env python3
"""hashcheck.py SRC [K ...] -- card self-test of exl3hash (seconds): random tiles through exllamav3's quantize_tiles
with the hash build installed (integer K: qtf2 for K1-4 / 6 / 7, exl3cb for K5 / 8; fractional K: qtf2): every
output value must equal the numpy hash value of its returned state (exl3hash.values), and the MSE must be in mul1's
range (a broken build gives garbage). Prints one line a K; exit 1 on a mismatch. Default K: 2 3 4 5 2.5."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for d in ("exl3frac",): sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d))
import exl3frac, exl3hash

src = sys.argv[1]
Ks = [float(k) if "." in k else int(k) for k in sys.argv[2:]] or [2, 3, 4, 5, 2.5]
exl3frac.install(src)
Q = exl3hash.install(src)
ref = torch.from_numpy(exl3hash.values())
torch.manual_seed(0)
tiles = torch.randn(512, 256, device="cuda")
bad = 0
for K in Ks:
    out, idx = Q.quantize_tiles(tiles, {"K": K, "mul1": True})
    st = idx.cpu().long() & 0xFFFF
    want = ref[st]
    ok = torch.equal(out.cpu().float(), want)
    mse = (out - tiles).pow(2).mean().item()
    print(f"K{K}: values == hash(state): {ok}  mse {mse:.5f}  distinct values {len(torch.unique(out))}", flush=True)
    bad += not ok
sys.exit(1 if bad else 0)
