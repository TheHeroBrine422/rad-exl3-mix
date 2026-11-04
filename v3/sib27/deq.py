#!/usr/bin/env python3
"""deq.py -- dequantize an EXL3 level (ourq.py's export: q-L*.safetensors + model.safetensors.index.json + stamp.json)
to the weights the level actually holds, in pure torch (CPU or GPU, no exllamav3 import). A transcription of the
plugin's C oracle (radiance-hero plugin/exl3import/common/exl3_decode.c, bit-exact with exllamav3's reconstruct, B19):

  W = diag(suh) . Hb . W_hat . Hb . diag(svh)     (exllamav3's [k, n] = (in, out); Hb Sylvester 128, 1/sqrt(128))
  W_hat: 16x16 tiles, tile = 16 K int16 = 8 K little-endian u32 words read as one MSB-first stream; position p owns
         bits [pK, (p+1)K), its 16-bit state is the window ending at bit (p+1)K (circular in the tile), value =
         codebook(state); p = 8 L + j sits at tile row 2 (L % 4) + {0,1,8,9}[j % 4], column L / 4 + 8 (j / 4).
  mul1:  fp16_fma(fp16(1024 + bytesum(s * 0x83DCD12D)), K_INV, K_BIAS)
  i8:    the int8-rounded mul1 the ladder levels are searched with (exl3cb, v3/nf/check/i8round.py):
         fp16(clamp(rint(v * (1 / (4 K_INV))), -127, 127) * 4 K_INV) -- a function of the mul1 value v, so exact.
The packing is the same for both (marker "mul1"); which one a level uses is in its stamp.json ("codebook"). i8c, mcg,
3INST and fractional K (qtf2 non-integer rates) are refused. W is formed in f32 in exllamav3's order (had over k, x suh,
had over n, x svh); --save-wq's files differ from it by the fp16 rounding of suh / svh (ourq keeps the refit scales
in f32 there) and their own fp16 storage: ~1e-3 relative.

CLI:
  deq.py wq LEVEL OUT KEYFILE      write OUT/<key>.safetensors "weight" (out, in) f16, ourq --save-wq's layout
  deq.py check LEVEL WQDIR [N]     decode N (default all) tensors that have a --save-wq file and compare
  deq.py oracle LEVEL SO [N]       W_hat vs the C oracle (exl3_decode.so: gcc -O2 -ffp-contract=off -shared -fPIC)
Env DEV (default cuda if available)."""
import json, os, struct, sys
import numpy as np, torch
from safetensors import safe_open

K_INV, K_BIAS, MARK = 0x1EEE, 0xC931, 0x83DCD12D
CODEBOOKS = ("mul1", "i8")


def _f16(bits): return float(np.array([bits], np.uint16).view(np.float16)[0])


def lut(codebook):
    """codebook value of each of the 65,536 states, f32 [65536] (exact fp16 values)"""
    assert codebook in CODEBOOKS, f"codebook {codebook}: only {CODEBOOKS}"
    s = np.arange(65536, dtype=np.uint64)
    x = (s * MARK) & 0xFFFFFFFF
    bs = sum(((x >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
    v = ((1024 + bs) * _f16(K_INV) + _f16(K_BIAS)).astype(np.float16)      # exact in f64, one rounding = fp16 fma
    if codebook == "i8":
        s4 = np.float32(4.0) * np.float32(_f16(K_INV)); inv = np.float32(1.0) / s4
        q = np.clip(np.rint(v.astype(np.float32) * inv), -127, 127).astype(np.float32)
        v = (q * s4).astype(np.float16)
    return torch.from_numpy(v.astype(np.float32))


def _perm():
    """P[r * 16 + c] = the stream position of tile element (r, c)"""
    P = np.empty(256, np.int64)
    off = (0, 1, 8, 9)
    for p in range(256):
        L, j = divmod(p, 8)
        P[(2 * (L % 4) + off[j % 4]) * 16 + L // 4 + 8 * (j // 4)] = p
    assert sorted(P) == list(range(256))
    return torch.from_numpy(P)


_P = _perm()


def hadamard128(dev):
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < 128: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / 128 ** 0.5).to(dev)


def decode_hat(tr, table, chunk=8192):
    """W_hat f32 [k, n] from a trellis int16 [k/16, n/16, 16 K] (on tr's device; table = lut(...) on that device)"""
    kt, nt, w16 = tr.shape
    assert w16 % 16 == 0, f"trellis words {w16}: not an integer K (fractional rates are not supported)"
    K = w16 // 16; nb = 256 * K; dev = tr.device
    t = tr.reshape(kt * nt, 8 * K, 2).to(torch.int64) & 0xFFFF
    words = t[..., 0] | (t[..., 1] << 16)                                  # little-endian u32 words
    sh = torch.arange(31, -1, -1, device=dev)
    pos = torch.arange(256, device=dev)[:, None] * K + K - 16 + torch.arange(16, device=dev)[None]   # window bits
    idx = (pos % nb).reshape(-1)
    wt = (1 << torch.arange(15, -1, -1, device=dev)).to(torch.int64)
    out = torch.empty(kt * nt, 256, dtype=torch.float32, device=dev)
    P = _P.to(dev)
    for c in range(0, kt * nt, chunk):
        bits = ((words[c:c + chunk, :, None] >> sh) & 1).reshape(-1, nb)   # MSB-first stream
        st = (bits[:, idx].reshape(-1, 256, 16) * wt).sum(-1)              # 16-bit states
        out[c:c + chunk] = table[st][:, P]                                 # tile order (r, c)
    return out.reshape(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def weight(tr, suh, svh, table):
    """W f32 [k, n] = (in, out): exllamav3's get_weight_tensor in f32"""
    w = decode_hat(tr, table)
    k, n = w.shape; H = hadamard128(w.device)
    w = (H @ w.view(k // 128, 128, n)).view(k, n)
    w *= suh.float()[:, None]
    w = (w.view(k, n // 128, 128) @ H).view(k, n)
    w *= svh.float()[None, :]
    return w


class Level:
    """an ourq.py export; .get(key) -> W (out, in) f32 on dev, the checkpoint's orientation"""
    def __init__(self, d, codebook=None, dev="cpu"):
        self.d, self.dev = d, torch.device(dev)
        self.ix = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
        st, asm = f"{d}/stamp.json", f"{d}/assembled.json"            # an assemble.py level: its base's codebook
        cb = json.load(open(st))[-1]["args"].get("codebook", "mul1") if os.path.exists(st) else \
            json.load(open(asm))["codebook"] if os.path.exists(asm) else None
        if codebook and cb and codebook != cb: sys.exit(f"{d}: stamp.json says codebook {cb}, asked for {codebook}")
        self.codebook = codebook or cb
        if self.codebook not in CODEBOOKS: sys.exit(f"{d}: codebook {self.codebook!r} (stamp.json) not decodable here: {CODEBOOKS}")
        self.table = lut(self.codebook).to(self.dev)
    def keys(self): return sorted(k[:-len(".trellis")] for k in self.ix if k.endswith(".trellis"))
    def _t(self, k):
        with safe_open(f"{self.d}/{self.ix[k]}", "pt", device="cpu") as f: return f.get_tensor(k)
    def get(self, key):
        assert f"{key}.mul1" in self.ix, f"{key}: no mul1 marker (mcg / 3INST codebook): not decodable here"
        tr, suh, svh = (self._t(f"{key}.{s}").to(self.dev) for s in ("trellis", "suh", "svh"))
        return weight(tr, suh, svh, self.table).T


def _wq(d, key):
    with safe_open(f"{d}/{key}.safetensors", "pt", device="cpu") as f: return f.get_tensor("weight")


if __name__ == "__main__":
    from safetensors.torch import save_file
    dev = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")
    cmd = sys.argv[1]
    if cmd == "wq":
        lv, out, kf = Level(sys.argv[2], dev=dev), sys.argv[3], sys.argv[4]
        os.makedirs(out, exist_ok=True)
        keys = [l.strip() for l in open(kf) if l.strip() and not l.startswith("#")]
        for key in keys:
            save_file({"weight": lv.get(key).half().contiguous().cpu()}, f"{out}/{key}.safetensors.part",
                      metadata={"source": f"deq.py {lv.d} codebook {lv.codebook}"})
            os.replace(f"{out}/{key}.safetensors.part", f"{out}/{key}.safetensors")
        print(f"DEQ DONE {len(keys)} tensors -> {out}")
    elif cmd == "check":
        lv, wqd = Level(sys.argv[2], dev=dev), sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
        keys = [k for k in lv.keys() if os.path.exists(f"{wqd}/{k}.safetensors")][:N]
        other = Level(sys.argv[2], codebook=lv.codebook, dev=dev)
        other.table = lut("mul1" if lv.codebook == "i8" else "i8").to(lv.dev)   # the wrong codebook, for contrast
        worst = 0.0
        for k in keys:
            q = _wq(wqd, k).float().to(lv.dev); w = lv.get(k); wo = other.get(k)
            r = float((w - q).norm() / q.norm()); ro = float((wo - q).norm() / q.norm())
            mx = float(((w - q).abs() / q.abs().clamp_min(float(q.abs().mean()))).max())
            worst = max(worst, r)
            print(f"{k:60s} rel {r:.2e} (max elem {mx:.2e}); with the other codebook {ro:.2e}", flush=True)
        print(f"DEQ CHECK {len(keys)} tensors, worst rel {worst:.2e} ({lv.codebook}): {'PASS' if keys and worst < 2e-3 else 'FAIL'}")
    elif cmd == "oracle":
        import ctypes
        lv, so = Level(sys.argv[2], dev="cpu"), sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        lib = ctypes.CDLL(so)
        lib.exl3_trellis_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.exl3_trellis_decode.restype = ctypes.c_int
        bad = 0
        for k in lv.keys()[:N]:
            tr = lv._t(f"{k}.trellis"); kt, nt, w16 = tr.shape
            ref = np.empty((kt * 16, nt * 16), np.uint16); trn = np.ascontiguousarray(tr.numpy())
            rc = lib.exl3_trellis_decode(trn.ctypes.data, kt, nt, w16 // 16, 2, ref.ctypes.data)   # 2 = EXL3_CB_MUL1
            mine = decode_hat(tr, lut("mul1")).numpy().astype(np.float16).view(np.uint16)
            n = int((mine != ref).sum()) if rc == 0 else -1; bad += n != 0
            print(f"{k}: K{w16 // 16} {kt * 16}x{nt * 16} rc {rc}, differing W_hat values {n}", flush=True)
        print("DEQ ORACLE: bit-exact" if bad == 0 else f"DEQ ORACLE FAIL: {bad} tensors differ")
