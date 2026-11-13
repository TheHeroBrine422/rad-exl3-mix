#!/usr/bin/env python3
"""abcheck147.py RAD ABSRC ABT LIB [OUT.safetensors] -- (w147, #178-02) the container's ssm_ab against its source and w142's simulated arm.
Per GDN layer: RAD's blk.N.ssm_ab.weight planes (record, suh, svh) byte-equal to ABSRC's trellis (tiles in k order) /
suh / svh; THE ORACLE's decode of the record (exl3import/common/exl3i_decode.c, LIB = libexl3iref.so), RNE bf16, rows
0-47 / 48-95 against ABT's blk.N.ssm_a / ssm_b (w142's ab-T4.safetensors: rxq --save-wq, exllamav3's fp16 decode, then
bf16): equal values, max |diff|, and the rel. Frobenius error of the difference. Prints one line a layer and a summary.
OUT: the oracle's bf16 a / b ({blk.N.ssm_a.weight, blk.N.ssm_b.weight} [48, 5120], abpatch.py's input): the served weight's
values as a dequantized splice (separates the decode's rounding from the A8 activation in the served KL)."""
import ctypes, json, struct, sys
import numpy as np
from safetensors import safe_open

RAD, SRC, ABT, LIB = sys.argv[1:5]
f = open(RAD, "rb"); h = f.read(4096)
str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt, pl_off, pl_cnt = struct.unpack_from("<8Q", h, 16 + 5 * 8)
f.seek(str_off); S = f.read(str_bytes)
ES = 136
f.seek(dir_off); D = f.read(ES * dir_cnt)
f.seek(pl_off); PL = np.frombuffer(f.read(16 * pl_cnt), np.uint64).reshape(-1, 2)
ent = {}
for i in range(dir_cnt):
    e = D[i * ES:(i + 1) * ES]
    nm = struct.unpack_from("<Q", e, 0)[0]
    rank = struct.unpack_from("<I", e, 28)[0]
    ent[S[nm:S.index(b"\0", nm)].decode()] = (struct.unpack_from("<6q", e, 32)[:rank], struct.unpack_from("<I", e, 92)[0],
                                              struct.unpack_from("<Q", e, 96)[0])
def plane(name, p):
    _, n, first = ent[name]; o, b = PL[first + p]
    f.seek(int(o)); return f.read(int(b))

class View(ctypes.Structure):
    _fields_ = [("rows", ctypes.c_int64), ("cols", ctypes.c_int64), ("m", ctypes.c_int), ("cb", ctypes.c_int),
                ("record", ctypes.c_void_p), ("suh", ctypes.c_void_p), ("suh1", ctypes.c_void_p), ("svh", ctypes.c_void_p)]
lib = ctypes.CDLL(LIB)
def bf16(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

src = safe_open(f"{SRC}/model.safetensors", "np"); abt = safe_open(ABT, "pt")
import torch
OUT = sys.argv[5] if len(sys.argv) > 5 else None; keep = {}
n_bad, tot_eq, tot, worst, worst_rel = 0, 0, 0, 0.0, 0.0
for l in range(64):
    nm = f"blk.{l}.ssm_ab.weight"
    if l % 4 == 3:
        assert nm not in ent; continue
    shape, npl, _ = ent[nm]
    assert tuple(shape) == (128, 5120) and f"blk.{l}.ssm_a.weight" not in ent and f"blk.{l}.ssm_b.weight" not in ent, (l, shape)
    P = f"model.language_model.layers.{l}.linear_attn.in_proj_ab"
    tr = src.get_tensor(f"{P}.trellis"); kt, nt, m = tr.shape
    rec_src = np.ascontiguousarray(tr.view(np.uint32).reshape(kt, nt, m // 2).transpose(1, 0, 2))
    rec = np.frombuffer(plane(nm, 1), np.uint32).copy(); suh = np.frombuffer(plane(nm, 2), np.uint16).copy()
    svh = np.frombuffer(plane(nm, 3), np.uint16).copy()
    same = (rec.tobytes() == rec_src.tobytes() and suh.tobytes() == src.get_tensor(f"{P}.suh").tobytes()
            and svh.tobytes() == src.get_tensor(f"{P}.svh").tobytes())
    out = np.empty((128, 5120), np.float32)
    v = View(128, 5120, m, 1, rec.ctypes.data, suh.ctypes.data, None, svh.ctypes.data)
    assert lib.exl3i_decode_rows(ctypes.byref(v), ctypes.c_int64(0), ctypes.c_int64(128), out.ctypes.data_as(ctypes.c_void_p)) == 0
    got = bf16(out[:96])
    want = np.concatenate([abt.get_tensor(f"blk.{l}.ssm_a.weight").view(dtype=__import__("torch").int16).numpy().view(np.uint16),
                           abt.get_tensor(f"blk.{l}.ssm_b.weight").view(dtype=__import__("torch").int16).numpy().view(np.uint16)])
    if OUT:
        keep[f"blk.{l}.ssm_a.weight"] = torch.from_numpy(got[:48].astype(np.int16)).view(torch.bfloat16).contiguous()
        keep[f"blk.{l}.ssm_b.weight"] = torch.from_numpy(got[48:96].astype(np.int16)).view(torch.bfloat16).contiguous()
    eq = int((got == want).sum())
    tof = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
    d = tof(got) - tof(want); mx = float(np.abs(d).max()); rel = float(np.linalg.norm(d) / np.linalg.norm(tof(want)))
    n_bad += not same; tot_eq += eq; tot += got.size; worst = max(worst, mx); worst_rel = max(worst_rel, rel)
    print(f"L{l:02d} planes {'=' if same else 'DIFF'} source | oracle vs w142 T4: {eq}/{got.size} equal, max |d| {mx:.3g}, rel {rel:.2e}")
if OUT:
    from safetensors.torch import save_file
    save_file(keep, OUT); print("wrote", OUT, len(keep))
print(json.dumps({"layers": 48, "planes_differ": n_bad, "equal": tot_eq, "values": tot, "max_abs": worst, "max_rel": worst_rel}))
