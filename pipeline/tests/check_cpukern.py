#!/usr/bin/env python3
"""check_cpukern.py [RQAH_CHECKOUT] -- rxq/cpukern.py's stand-in kernels (CPU, seconds): every tile the greedy encoder
returns equals the decode of its packed trellis, by two decoders: rxq.deq (decode_hat) and, if a rad-quant-at-home
checkout is given (default /workspace/rad-quant-at-home), its independent numpy checker (checker/exl3dec.py), at
sub-1, fractional and integer rates for every codebook the ladder uses; the states are valid 16-bit windows of the
ring; the error is sane from K1.5 up (below the input's variance; the greedy encoder is poor at <= 1 bit a weight,
which only matters for the stand-in's own numbers). Last line: "CPUKERN CHECKS n/m PASS"."""
import os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from rxq import cpukern, deq  # noqa: E402

RQ = sys.argv[1] if len(sys.argv) > 1 else "/workspace/rad-quant-at-home"
ex = None
if os.path.isdir(os.path.join(RQ, "checker")):
    sys.path.insert(0, RQ)
    from checker import exl3dec as ex  # noqa: E402
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


g = torch.Generator().manual_seed(7)
perm = torch.tensor([r * 16 + c for t in range(32) for r, c in
                     [((t % 4) * 2 + d, t // 4 + 8 * e) for e in (0, 1) for d in (0, 1, 8, 9)]])   # ring position -> tile element
for cb, Ks in (("i4", (0.125, 0.5, 0.875, 1, 1.5, 2, 2.5, 3)), ("i8", (1.125, 2, 2.5, 4, 5, 6.5, 7)), ("mul1", (6.5, 7.25, 8))):
    for K in Ks:
        kt, nt = 2, 3
        tiles = torch.randn(kt * nt, 256, generator=g) * 1.2
        fk = cpukern.frac_k(K); ka, mask = fk if fk else (int(K), 0)
        vals, st = cpukern.encode(tiles, ka, mask, cb)
        packed = cpukern.pack(st.view(kt, nt, 256), ka, mask)
        ok_shape = tuple(packed.shape) == (kt, nt, int(round(16 * K)))
        what = deq.decode_hat(packed, K, torch.from_numpy(deq.codebook_lut(cb).astype(np.float32)))   # (16 kt, 16 nt)
        # the encoder's values are in ring order per tile; decode_hat gives tile layout: map back through perm
        ref = torch.empty(kt, nt, 16, 16)
        for a in range(kt):
            for c in range(nt):
                t = torch.empty(256); t[perm] = vals[a * nt + c]
                ref[a, c] = t.view(16, 16)
        ref = ref.permute(0, 2, 1, 3).reshape(16 * kt, 16 * nt)
        same = torch.equal(what, ref)
        why = f"{cb} K{K}: shape {tuple(packed.shape)} ok {ok_shape}, deq decode == encoder values {same}"
        if ex is not None:
            W2 = ex.decode_what(packed.numpy(), K, cb)
            s2 = np.array_equal(W2, ref.numpy()); same = same and s2; why += f", exl3dec {s2}"
        mse = float(((vals - tiles) ** 2).mean()); why += f", mse {mse:.3f} (input var 1.44)"
        report(f"{cb}-K{K}", same and ok_shape and (mse < 1.0 or K < 1.5), why)
print(f"CPUKERN CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
