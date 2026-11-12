#!/usr/bin/env python3
"""tpcheck113.py UNIT PERMJSON OUT.json -- w113 #148-03 (../w113.md): TP functional check of P16-TP on real quantized
tensors, CPU, no production kernel.
UNIT = an rxq T-arm unit (UNIT/wq/<key>.safetensors: the decoded Wq in STORED row order, fp16). Per key, x = 4 random
tokens (seed 148), float64:
  kernel model  per stored 128-row block b: U_b = H128 W_s[b] (codes live in the output-Hadamard basis), the GEMM gives
                u = x U_bᵀ and the epilogue y_b = u H128 (the block's output Hadamard).
  TP 1 ref      every stored block, outputs un-permuted by the TABLE (perm.json), then cut by natural rank rows.
  TP t rank r   the loader's cut of the stored rows (rank_rows: per part, the rank's slice), U_b from the rank's own rows,
                each local 128-row block asserted to be one whole stored block, outputs written through the closed form
                (p16tp_map.c, compiled here) into the rank's local natural columns.
PASS iff every rank at TP 2 and TP 4 equals the TP 1 ref bit for bit, and the C map equals the table for every family,
TP and tile. Also reported: max |y - x W_natᵀ| / max |x W_natᵀ| (float64, a sanity number)."""
import ctypes, json, os, subprocess, sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from perm113 import PARTS, family, rank_rows, tpperm, local_map, arm_perm  # noqa: E402

UNIT, PJ, OUT = sys.argv[1:4]
so = os.path.join(os.path.dirname(os.path.abspath(OUT)), "libp16tp.so")
subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "p16tp_map.c")], check=True)
L = ctypes.CDLL(so)


class P16TP(ctypes.Structure):
    _fields_ = [("np", ctypes.c_int), ("m", ctypes.c_int), ("nb", ctypes.c_int), ("c", ctypes.c_int * 3), ("C", ctypes.c_int * 4)]


def cmap(parts, t, n_tiles):
    c = [P // (16 * 4) for P in parts]; C = [0] + list(np.cumsum(c))
    a = P16TP(len(parts), 4 // t, C[-1] // 8, (ctypes.c_int * 3)(*(c + [0] * (3 - len(c)))), (ctypes.c_int * 4)(*(C + [0] * (4 - len(C)))))
    out = (ctypes.c_int * n_tiles)(); L.p16tp_all(ctypes.byref(a), n_tiles, out); return np.array(out[:], dtype=np.int64)


res = {"map": {}, "keys": {}}; ok = True
# (1) the C map == the table, every family, TP, rank, tile
for fam, parts in PARTS.items():
    p = tpperm(parts, 4).numpy(); N = sum(parts)
    for t in (1, 2, 4):
        bad = 0
        for r in range(t):
            S = rank_rows(parts, t, r); pos = {int(x): i for i, x in enumerate(S)}
            tab = np.array([pos[int(p[S[16 * k]])] // 16 for k in range(len(S) // 16)])
            # every row of a tile moves with it (the map is a tile map)
            assert all(pos[int(p[S[16 * k + c]])] == 16 * tab[k] + c for k in range(0, len(S) // 16, 7) for c in range(16))
            cm = cmap(parts, t, len(S) // 16); bad += int((cm != tab).sum())
            assert all(local_map(parts, 4, t, k) == cm[k] for k in range(0, len(cm), 5))
        res["map"][f"{fam} TP{t}"] = bad; ok &= bad == 0
        print(f"map {fam:12s} TP{t}: C map vs table: {bad} tiles differ", flush=True)

# (2) functional check on the unit's tensors
Hd = np.array([[1.0]])
while Hd.shape[0] < 128: Hd = np.block([[Hd, Hd], [Hd, -Hd]])
H = Hd / np.sqrt(128)
perms = json.load(open(PJ)); rng = np.random.default_rng(148)


def blocks_out(x, Ws):   # Ws: [n, K] stored rows (n % 128 == 0) -> [M, n] in the same (stored) order
    y = np.empty((x.shape[0], Ws.shape[0]))
    for b in range(Ws.shape[0] // 128):
        U = H @ Ws[128 * b:128 * b + 128]
        y[:, 128 * b:128 * b + 128] = (x @ U.T) @ H
    return y


for f in sorted(x for x in os.listdir(f"{UNIT}/wq") if x.endswith(".safetensors")):
    key = f[:-len(".safetensors")]; parts = PARTS[family(key)]
    with safe_open(f"{UNIT}/wq/{f}", "pt") as g: Ws = g.get_tensor("weight").double().numpy()
    N, K = Ws.shape; p = np.array(perms[key]); assert np.array_equal(p, arm_perm(key, "T", N).numpy())
    x = rng.standard_normal((4, K))
    ys = blocks_out(x, Ws)
    ynat = np.empty_like(ys); ynat[:, p] = ys                      # TP 1 reference, through the table
    Wnat = np.empty_like(Ws); Wnat[p] = Ws; yd = x @ Wnat.T
    rel = float(np.abs(ynat - yd).max() / np.abs(yd).max())
    out = {"rel_vs_dense": rel}
    for t in (2, 4):
        nbad = 0
        for r in range(t):
            S = rank_rows(parts, t, r)                               # the loader's cut = the rank's stored positions
            for lb in range(len(S) // 128):                          # each local block is one whole stored block
                assert S[128 * lb] % 128 == 0 and np.array_equal(S[128 * lb:128 * lb + 128], np.arange(S[128 * lb], S[128 * lb] + 128))
            yl = blocks_out(x, np.ascontiguousarray(Ws[S]))         # rank-local GEMM + per-block output Hadamard
            cm = cmap(parts, t, len(S) // 16)
            cols = (16 * cm[:, None] + np.arange(16)[None, :]).reshape(-1)
            yr = np.empty_like(yl); yr[:, cols] = yl                 # the store remap (closed form)
            ref = ynat[:, S]                                         # natural rank-local rows: same set, natural order
            nbad += int((yr != ref).sum())
        out[f"TP{t}_mismatch"] = nbad; ok &= nbad == 0
    res["keys"][key] = out
    print(f"{key:48s} N {N:5d}: TP2 {out['TP2_mismatch']} / TP4 {out['TP4_mismatch']} outputs differ from TP1 (bit for bit); "
          f"rel vs x·W_natᵀ {rel:.2e}", flush=True)
res["pass"] = bool(ok)
json.dump(res, open(OUT, "w"), indent=1)
print(f"W113 #148-03 TPCHECK {'PASS' if ok else 'FAIL'}")
