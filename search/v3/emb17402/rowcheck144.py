#!/usr/bin/env python3
"""rowcheck144.py SRC CB REF.rad N BIN -- (w144, #174-02) the served row against the dequantized reference on the REAL table:
N token ids (seeded: 1/2 uniform over the vocabulary, 1/2 the low ids < 32768 where text sits) gathered by embed_exl3i's
host row (rowcheck144, == the device row bit for bit) from SRC's record, compared with REF.rad's token_embd rows (the
oracle's bf16 decode of the same record, embref.py). Prints elements differing, by how many bf16 ulps, and the largest
|difference| / row RMS. BIN: the rowcheck144 binary."""
import json, os, struct, subprocess, sys, tempfile
import numpy as np
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SRC, CB, REF, N, BIN = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
V, E = 248320, 5120
P = "model.language_model.embed_tokens"
with safe_open(f"{SRC}/q-embed.safetensors", "np") as f:
    tr = f.get_tensor(f"{P}.trellis"); suh = f.get_tensor(f"{P}.suh"); svh = f.get_tensor(f"{P}.svh")
kt, nt, m = tr.shape
rec = np.ascontiguousarray(tr.view(np.uint32).reshape(kt, nt, m // 2).transpose(1, 0, 2)); del tr
rng = np.random.default_rng(144)
tok = np.concatenate([rng.integers(0, V, N // 2), rng.integers(0, 32768, N - N // 2)]).astype(np.int32)
def entry(path, want):
    f = open(path, "rb"); h = f.read(4096)
    str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt = struct.unpack_from("<6Q", h, 16 + 5 * 8)
    f.seek(str_off); S = f.read(str_bytes)
    es = 3 * 8 + 4 + 4 + 8 * 6 + 4 * 3 + 4 + 8 + 8 + 8 + 16
    f.seek(dir_off); D = f.read(es * dir_cnt)
    for i in range(dir_cnt):
        e = D[i * es:(i + 1) * es]
        name = struct.unpack_from("<Q", e, 0)[0]
        if S[name:S.index(b"\0", name)].decode() != want: continue
        off, nbytes = struct.unpack_from("<QQ", e, es - 32)
        return off, nbytes
    raise KeyError(want)
with tempfile.TemporaryDirectory() as d:
    for n, a in (("rec", rec), ("suh", suh), ("svh", svh), ("tok", tok)): a.tofile(f"{d}/{n}")
    subprocess.run([BIN, f"{d}/rec", f"{d}/suh", f"{d}/svh", f"{d}/tok", f"{d}/out", str(m), CB], check=True)
    x = np.fromfile(f"{d}/out", np.uint16).reshape(N, E)
off, nb = entry(REF, "token_embd.weight"); assert nb == V * E * 2
ref = np.memmap(REF, np.uint16, "r", offset=off, shape=(V, E))[tok]
key = lambda u: np.where(u >> 15, -(u & 0x7FFF).astype(np.int64), (u & 0x7FFF).astype(np.int64))
d = key(x) - key(ref)
f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
rms = np.sqrt((f32(ref) ** 2).mean(1, keepdims=True)); rel = np.abs(f32(x) - f32(ref)) / np.where(rms > 0, rms, 1)
print(json.dumps({"src": SRC, "tokens": N, "elements": int(d.size), "differ": int((d != 0).sum()),
                  "differ_frac": float((d != 0).mean()), "max_ulps": int(np.abs(d).max()), "gt1ulp": int((np.abs(d) > 1).sum()),
                  "max_abs_over_rms": float(rel.max()), "rows_identical": int((d == 0).all(1).sum())}))
if os.environ.get("ROWCHECK_WORST"):
    idx = np.argsort(-rel.ravel())[:12]
    for i in idx:
        r, c = divmod(int(i), E)
        print(json.dumps({"tok": int(tok[r]), "col": c, "x": float(f32(x[r:r+1, c:c+1])[0, 0]), "ref": float(f32(ref[r:r+1, c:c+1])[0, 0]),
                          "rms": float(rms[r, 0]), "ulps": int(d[r, c]), "row_differ": int((d[r] != 0).sum())}))
