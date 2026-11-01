#!/usr/bin/env python3
"""b19k1.py TD_DIR ORACLE.so [N] -- mining merge B19: is our K1 decode bit-exact with exllamav3's?

For up to N tensors per K in turboderp's export TD_DIR (e.g. td-SC_1.40bpw_H3: K1 and K2 tensors), W_hat (the trellis
values, f16, exllamav3's [k, n]) from exllamav3's own GPU decode (ext.reconstruct) vs from the plugin's C oracle
(plugin/exl3import/common/exl3_decode.c, exl3_trellis_decode -- what our reference kernels and the importer's checks
use). Prints per tensor the number of differing f16 values; any nonzero count at K1 is a bug. Build the oracle with
  gcc -O2 -ffp-contract=off -shared -fPIC exl3_decode.c -o exl3_decode.so
One card (a few MB plus the context); run with ROCR_VISIBLE_DEVICES set."""
import ctypes, json, os, sys
import numpy as np, torch
from safetensors import safe_open

TD, SO = sys.argv[1:3]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
lib = ctypes.CDLL(SO)
lib.exl3_trellis_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.exl3_trellis_decode.restype = ctypes.c_int
from exllamav3.ext import exllamav3_ext as ext
if os.path.exists(f"{TD}/model.safetensors.index.json"):
    ix = json.load(open(f"{TD}/model.safetensors.index.json"))["weight_map"]
else:                                           # single-file exports
    import glob as _g
    ix = {}
    for fn in sorted(_g.glob(f"{TD}/*.safetensors")):
        with safe_open(fn, "pt") as f: ix.update({k: os.path.basename(fn) for k in f.keys()})
keys = sorted(k[:-len(".trellis")] for k in ix if k.endswith(".trellis"))
byK = {}
for key in keys:
    with safe_open(f"{TD}/{ix[key + '.trellis']}", "pt") as f: K = f.get_slice(key + ".trellis").get_shape()[2] // 16
    byK.setdefault(K, []).append(key)
print({K: len(v) for K, v in sorted(byK.items())}, flush=True)
bad = 0
for K in sorted(byK):
    for key in byK[K][:N]:
        def g(sub):
            k = f"{key}.{sub}"
            if k not in ix: return None
            with safe_open(f"{TD}/{ix[k]}", "pt") as f: return f.get_tensor(k)
        tr = g("trellis"); cb = 2 if g("mul1") is not None else 1 if g("mcg") is not None else 0
        kt, nt = tr.shape[0], tr.shape[1]
        # exllamav3 (GPU)
        w = torch.empty(kt * 16, nt * 16, dtype=torch.half, device="cuda")
        ext.reconstruct(w, tr.cuda(), K, cb == 1, cb == 2)
        ref = w.cpu().view(torch.int16).numpy().view(np.uint16)
        # oracle (CPU)
        trn = np.ascontiguousarray(tr.numpy())
        out = np.empty((kt * 16, nt * 16), np.uint16)
        rc = lib.exl3_trellis_decode(trn.ctypes.data, kt, nt, K, cb, out.ctypes.data)
        n = int((out != ref).sum()) if rc == 0 else -1
        bad += n != 0
        print(f"K{K} cb{cb} {key}: {kt * 16}x{nt * 16}, rc {rc}, differing f16 values {n}", flush=True)
print("B19 OK: bit-exact" if bad == 0 else f"B19 FAIL: {bad} tensors differ")
