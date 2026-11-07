# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""An ourq-layout level back to the weights it holds, in torch (CPU or GPU, no exllamav3 import): what the level passes
(S4 swj, S5 sibling moments) and the writers' C9 pass read. Port of rad-exl3-mix v3/yaqa/swj27/exl3deq.py (decode) with
rad-quant-at-home checker/exl3dec.py's codebook tables and ring plan (mul1 / i8 / i8c / i4, every rate in 1/8 steps,
the sub-1 salt), vendored here so rxq needs no other checkout (tests/check_deq.py: bit-identical to exl3deq.py and to
v3/yaqa/asym27/deq.py on a real 27B level).

  W (k = in, n = out) = diag(suh) . Hb . W_hat . Hb . diag(svh)   (exllamav3's order: had over k, x suh, had over n, x svh)
  W_hat: 16x16 tiles, each a tail-biting ring of 16 K bits in 32-bit little-endian words read MSB first; ring position
  i reads D(i) = KA + bit (i mod 16) of MASK new bits, its state is the 16-bit window ending at S(i) = D(0) + .. + D(i);
  a 0-bit step (sub-1 K) reads the codebook at state ^ salt(i), salt(i) = 0x9E37 (i mod 16 + 1) mod 2^16.
W is formed in f32 (no f16 rounding: what the kernels compute with fp16 suh / svh)."""
import json, math, os

import numpy as np
import torch
from safetensors import safe_open

MUL1 = 0x83DCD12D
CODEBOOKS = ("mul1", "i8", "i8c", "i4")


def frac_mask(n):
    return sum(1 << i for i in range(16) if ((i + 1) * n) // 16 - (i * n) // 16)


def ka_mask(K):
    """(KA, MASK) of bitrate K (16K must be an even integer for fractional K)"""
    if float(K).is_integer(): return int(K), 0
    ka = int(K); n = round((K - ka) * 16)
    assert abs((K - ka) * 16 - n) < 1e-9 and n % 2 == 0, K
    return ka, frac_mask(n)


def _f16(bits): return np.array([bits], dtype=np.uint16).view(np.float16)[0]


def codebook_lut(cb):
    """the fp16 value of every 16-bit state, as float32 [65536] (exl3dec.codebook_lut)"""
    s = np.arange(65536, dtype=np.uint64)
    x = (s * MUL1) & 0xFFFFFFFF
    bs = sum(((x >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
    k_inv, k_bias = float(_f16(0x1EEE)), float(_f16(0xC931))
    v = (1024.0 + bs) * k_inv + k_bias                      # exact in f64: one rounding to f16, as __hfma
    v = v.astype(np.float16)
    if cb == "mul1": return v.astype(np.float32)
    s4 = np.float32(4.0) * np.float32(_f16(0x1EEE)); inv = np.float32(1.0) / s4
    if cb == "i8":
        q = np.clip(np.rint(v.astype(np.float32) * inv), np.float32(-127), np.float32(127))
        return (q * s4).astype(np.float16).astype(np.float32)
    if cb == "i8c":                                         # #78: C = 0.02, SD = 1.0546900 (exl3frac2.load)
        C, SD, sd = np.float32(0.02), np.float32(1.0546900), np.float32(36.950135)
        x = (v.astype(np.float32) * inv) * (np.float32(1.0) / sd)
        a = (C * x) * x
        y = np.array([math.fma(float(p), float(q), float(q)) if hasattr(math, "fma") else float(p) * float(q) + float(q)
                      for p, q in zip(a, x)], dtype=np.float64).astype(np.float32) * (sd / SD)
        q = np.clip(np.rint(y), np.float32(-127), np.float32(127))
        return (q * s4).astype(np.float16).astype(np.float32)
    if cb == "i4":                                          # the i4 ladder's int4 grid (codebook.cuh EXL3_MUL1_I4)
        st, iv = np.float32(2.6) / np.float32(7.5), np.float32(7.5) / np.float32(2.6)
        q = np.clip(np.rint(v.astype(np.float32) * iv - np.float32(0.5)), -8, 7) + np.float32(0.5)
        return (q * st).astype(np.float16).astype(np.float32)
    raise ValueError(f"codebook {cb!r}: not one of {CODEBOOKS}")


_LUT = {}


def lut(cb):
    if cb not in _LUT: _LUT[cb] = codebook_lut(cb)
    return _LUT[cb]


def ring_plan(K):
    """per ring position: first word, next word, shift, salt (0 unless a 0-bit step), the tile element it lands on"""
    ka, mask = ka_mask(K)
    D = np.array([ka + ((mask >> (i & 15)) & 1) for i in range(256)])
    S = np.cumsum(D)
    nw = int(round(8 * K)); R = nw * 32
    assert S[-1] == R, (K, S[-1], R)
    start = (S - 16) % R
    wi, o = start >> 5, start & 31
    salt = np.where(D == 0, (0x9E37 * ((np.arange(256) & 15) + 1)) & 0xFFFF, 0).astype(np.uint64)
    t, j = np.arange(256) >> 3, np.arange(256) & 7
    r = (t & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1); c = (t >> 2) + 8 * ((j >> 2) & 1)
    return nw, wi, (wi + 1) % nw, (48 - o).astype(np.uint64), salt, r * 16 + c


def had128():
    i = np.arange(128)
    pc = np.vectorize(lambda x: bin(x).count("1"))(i[:, None] & i[None, :])
    return (np.where(pc % 2, -1.0, 1.0) / math.sqrt(128)).astype(np.float32)


def plan(K, dev):
    """ring_plan as torch tensors: hi / lo word index, the shifts, salt, element -> ring position"""
    nw, w0, w1, sh, salt, elem = ring_plan(K)
    sh = sh.astype(np.int64)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)
    return nw, t(w0), t(w1), t(sh), t(salt.astype(np.int64)), t(np.argsort(elem))


def decode_hat(tr, K, table, chunk=1 << 16):
    """trellis int16 [kt, nt, 16 K] -> W_hat f32 [16 kt, 16 nt] on tr's device; table = the codebook's f32 [65536]"""
    kt, nt, words = tr.shape
    assert words == round(16 * K), (tuple(tr.shape), K)
    dev = tr.device
    nw, w0, w1, sh, salt, inv = plan(K, dev)
    hi_s = (sh - 32).clamp(min=0); lo_l = (32 - sh).clamp(min=0); lo_r = sh.clamp(max=31); big = sh >= 32
    u = tr.reshape(kt * nt, words).to(torch.int64) & 0xFFFF
    out = torch.empty(kt * nt, 256, dtype=torch.float32, device=dev)
    for c in range(0, kt * nt, chunk):
        t = u[c:c + chunk]
        w = t[:, 0::2] | (t[:, 1::2] << 16)                                  # little-endian u32 words, < 2^32
        a, b = w[:, w0], w[:, w1]                                            # the window's first and next word
        v = torch.where(big, a >> hi_s, (a << lo_l) | (b >> lo_r))           # = ((a << 32) | b) >> sh, low 16 bits
        st = (v & 0xFFFF) ^ salt
        out[c:c + chunk] = table[st][:, inv]                                 # tile elements r * 16 + c (r = k)
    return out.reshape(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def hadamard128(dev):
    return torch.from_numpy(had128()).to(dev)


def weight(tr, suh, svh, K, table):
    """W f32 [k, n] = (in, out), exl3dec.decode's order"""
    w = decode_hat(tr, K, table)
    k, n = w.shape; H = hadamard128(w.device)
    w = (H @ w.reshape(k // 128, 128, n)).reshape(k, n)
    w *= suh.float()[:, None]
    w = (w.reshape(k, n // 128, 128) @ H).reshape(k, n)
    w *= svh.float()[None, :]
    return w


def level_codebook(d):
    """a level's codebook: ourq's stamp.json (last run's args), an rxq-assembled level's stamp.json, or assembled.json"""
    st = os.path.join(d, "stamp.json")
    if os.path.exists(st):
        e = json.load(open(st))[-1]
        return e["codebook"] if "codebook" in e else (e.get("args", {}).get("codebook") or "mul1")
    asm = os.path.join(d, "assembled.json")
    if os.path.exists(asm): return json.load(open(asm))["codebook"]
    raise SystemExit(f"{d}: no stamp.json / assembled.json (which codebook?)")


class Level:
    """an ourq-layout level; .get(key) -> W (out, in) f32 on dev (the checkpoint's orientation)"""
    def __init__(self, d, dev="cpu", codebook=None):
        self.d, self.dev = d, torch.device(dev)
        self.ix = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.codebook = codebook or level_codebook(d)
        if self.codebook not in CODEBOOKS: raise SystemExit(f"{d}: codebook {self.codebook!r} not decodable ({CODEBOOKS})")
        self.table = torch.from_numpy(lut(self.codebook)).to(self.dev)
        self.h = {}

    def keys(self): return sorted(k[:-len(".trellis")] for k in self.ix if k.endswith(".trellis"))

    def _t(self, name):
        f = self.ix[name]
        if f not in self.h: self.h[f] = safe_open(os.path.join(self.d, f), "pt", device="cpu")
        return self.h[f].get_tensor(name)

    def raw(self, key):
        assert f"{key}.mul1" in self.ix, f"{key}: no mul1 marker (mcg / 3INST): not decodable here"
        return {s: self._t(f"{key}.{s}") for s in ("trellis", "suh", "svh")}

    def K(self, key):
        return self.raw(key)["trellis"].shape[2] / 16

    def get(self, key):
        t = self.raw(key)
        K = t["trellis"].shape[2] / 16
        return weight(t["trellis"].to(self.dev), t["suh"].to(self.dev), t["svh"].to(self.dev), K, self.table).T
