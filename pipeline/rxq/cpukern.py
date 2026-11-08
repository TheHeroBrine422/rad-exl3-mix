# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""CPU stand-in for exllamav3's GPU kernels: TEST ONLY (dry runs of the job / queue / assembly wiring without a card).

What runs is the fork's real quantize.py (finalize, regularize, scale search, scaled LDL, LDLQ / ldlq_2hess, refit, pack)
on CPU tensors, with the extension's five entry points replaced:
  quantize_tiles / quantize_tiles_frac   a GREEDY trellis encoder instead of the Viterbi search: ring position by ring
                                         position, the D(i) new bits that bring the state's codebook value nearest the
                                         target. The codes are valid (a tail-biting ring of 16 K bits per tile, the
                                         codebook's values, the sub-1 salt) but worse than the real search's.
  pack_trellis / pack_trellis_frac       the ring's words, the exact inverse of rxq.deq's decode
  quantize_tiles_scratch                 unused (the temp buffers are dummies)
and torch.cuda's streams / events / synchronize made no-ops; block_ldl keeps a copy of H on the CPU as its card path
does (on a CPU tensor its H.cpu() aliases the factor it writes into H's storage). Codebooks: mul1 / i8 / i4 tables of rxq.deq (the ones the
real kernels and the exl3i decoder use). So every quantized tile equals the decode of its packed codes (tests:
check_cpukern.py), and everything downstream (C9, assembly, the queue's scorer, rad-convert's exl3i import) sees
well-formed codes. The trellis search differs, so the codes and proxy errors are NOT the real quantizer's: results are
stamped (code stamp over "cpu-standin", tensor metadata rxq_kernels = cpu-standin) and must never enter a level.

load(fork_root): the fork's quantize.py with its package stubbed (no extension build, no GPU), registered under its
module name so `import exllamav3.modules.quant.exl3_lib.quantize` returns it. install(Q, codebook): the stand-ins."""
import contextlib, importlib.util, math, os, sys, types

import numpy as np
import torch

from . import deq

MARK = "cpu-standin"
_STATE = {"loaded": None, "installed": None}


def _sylvester(n):
    h = torch.ones(1, 1, dtype=torch.float16)
    while h.shape[0] < n: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class _PB:
    """exllamav3's ProgressBar, silent"""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def update(self, *a, **k): pass
    def new_task(self, *a, **k): pass


def load(fork_root):
    """the fork's quantize.py on CPU (package stubbed); idempotent per fork_root"""
    name = "exllamav3.modules.quant.exl3_lib.quantize"
    if _STATE["loaded"] is not None:
        assert _STATE["loaded"] == fork_root, f"cpukern: already loaded from {_STATE['loaded']}"
        return sys.modules[name]
    qpy = os.path.join(fork_root, "exllamav3", "modules", "quant", "exl3_lib", "quantize.py")
    if not os.path.isfile(qpy): raise SystemExit(f"cpukern: no {qpy} (RXQ_EXL3_SRC must be the fork's exllamav3_ext dir)")

    def stub(n, **kw):
        m = types.ModuleType(n); m.__path__ = []; m.__dict__.update(kw); sys.modules[n] = m; return m
    stub("exllamav3"); stub("exllamav3.ext", exllamav3_ext=types.SimpleNamespace())
    stub("exllamav3.util", cuda_sync_active=lambda *a, **k: None)
    stub("exllamav3.util.progress", ProgressBar=_PB)
    stub("exllamav3.util.memory", free_mem=lambda *a, **k: None, list_gpu_tensors=lambda *a, **k: None)
    stub("exllamav3.util.hadamard", get_hadamard_dt=lambda n, device, dtype, scale=1.0:
         (_sylvester(n).to(device=device, dtype=dtype) * scale))     # had_k = had_n = 128: Sylvester (hadamard_1.txt up)
    stub("exllamav3.util.tensor", save_tensor_image=lambda *a, **k: None)
    for p in ("exllamav3.modules", "exllamav3.modules.quant", "exllamav3.modules.quant.exl3_lib"): stub(p)
    spec = importlib.util.spec_from_file_location(name, qpy)
    Q = importlib.util.module_from_spec(spec); sys.modules[name] = Q; spec.loader.exec_module(Q)
    _STATE["loaded"] = fork_root
    return Q


def frac_k(K):
    """exl3frac2's frac_k: (KA, MASK) for K = KA + n / 16, n even, KA >= 0; None for integer K"""
    if isinstance(K, int) or float(K).is_integer(): return None
    ka = int(K); n = round((K - ka) * 16)
    assert abs((K - ka) * 16 - n) < 1e-9 and n % 2 == 0 and K > 0, f"fractional bitrate {K}"
    return ka, deq.frac_mask(n)


def _plan(ka, mask):
    D = np.array([ka + ((mask >> (i & 15)) & 1) for i in range(256)], dtype=np.int64)
    S = np.cumsum(D)
    salt = np.where(D == 0, (0x9E37 * ((np.arange(256) & 15) + 1)) & 0xFFFF, 0).astype(np.int64)
    return D, S, salt


_LUT = {}


def _lut(cb):
    if cb not in _LUT: _LUT[cb] = torch.from_numpy(deq.codebook_lut(cb).astype(np.float32))
    return _LUT[cb]


def encode(tiles, ka, mask, cb):
    """tiles (N, 256) f32 in ring order -> (values (N, 256) f32, states (N, 256) int64): greedy tail-biting trellis"""
    D, S, salt = _plan(ka, mask)
    R = int(S[-1]); assert R == 256 * ka + 16 * bin(mask).count("1"), (R, ka, mask)
    lut = _lut(cb)
    x = tiles.detach().to("cpu", torch.float32)
    N = x.shape[0]
    st = torch.zeros(N, dtype=torch.int64)
    gst = torch.empty((N, 256), dtype=torch.int64)
    cand = {d: torch.arange(1 << d, dtype=torch.int64) for d in set(D.tolist())}
    for i in range(256):
        d = int(D[i])
        if d:
            c = ((st.unsqueeze(1) << d) | cand[d].unsqueeze(0)) & 0xFFFF               # (N, 2^d)
            v = lut[c ^ int(salt[i])]
            j = (v - x[:, i:i + 1]).abs().argmin(1)
            st = c.gather(1, j.unsqueeze(1)).squeeze(1)
        gst[:, i] = st
    # the true windows: positions whose 16-bit window starts before ring bit 0 wrap into the ring's tail, which the
    # last position's window ends with (the greedy pass assumed zeros there)
    for i in range(256):
        s = int(S[i])
        if s >= 16: break
        tail = st & ((1 << (16 - s)) - 1)
        gst[:, i] = (tail << s) | (gst[:, i] & ((1 << s) - 1))
    vals = lut[gst ^ torch.from_numpy(salt).unsqueeze(0)]
    return vals, gst


def pack(states, ka, mask):
    """states (..., 256) int (ring order) -> trellis words (..., 16 K) int16, rxq.deq's layout: the ring's bits
    MSB first in 32-bit words, each word as (low uint16, high uint16)"""
    D, S, _ = _plan(ka, mask)
    R = int(S[-1]); nw = R // 32
    sh = states.shape[:-1]
    s = states.reshape(-1, 256).to("cpu", torch.int64).numpy() & 0xFFFF
    bits = np.zeros((s.shape[0], R), dtype=np.uint8)
    for i in range(256):
        d = int(D[i])
        for b in range(d):                      # ring bit S - d + b = bit (d - 1 - b) of the state (the newest bits)
            bits[:, int(S[i]) - d + b] = (s[:, i] >> (d - 1 - b)) & 1
    w = np.zeros((s.shape[0], nw), dtype=np.uint64)
    for p in range(32):
        w |= bits[:, p::32].astype(np.uint64) << np.uint64(31 - p)
    u16 = np.empty((s.shape[0], 2 * nw), dtype=np.uint16)
    u16[:, 0::2] = (w & np.uint64(0xFFFF)).astype(np.uint16); u16[:, 1::2] = (w >> np.uint64(16)).astype(np.uint16)
    return torch.from_numpy(u16.view(np.int16).reshape(*sh, 2 * nw).copy())


class _Ext:
    """the extension's entry points the quantizer calls, on the CPU"""
    def __init__(self, cb): self.cb = cb

    def quantize_tiles_scratch(self, *a, **k): return False, 0

    def _q(self, tiles, out, idx, ka, mask):
        v, s = encode(tiles, ka, mask, self.cb)
        out.copy_(v.to(out.device)); idx.copy_(s.to(torch.int16).to(idx.device))   # int16 view of the 16-bit state

    def quantize_tiles(self, tiles, out, idx, temp_costs, temp_edges, K, mcg, mul1):
        assert mul1 and not mcg, "cpukern: mul1-family codebooks only"
        self._q(tiles, out, idx, int(K), 0)

    def quantize_tiles_frac(self, tiles, out, idx, temp_costs, temp_edges, ka, mask):
        self._q(tiles, out, idx, int(ka), int(mask))

    def pack_trellis(self, packed, encoded, K):
        packed.copy_(pack(encoded, int(K), 0).to(packed.device))

    def pack_trellis_frac(self, packed, encoded, ka, mask):
        packed.copy_(pack(encoded, int(ka), int(mask)).to(packed.device))

    def pack_signs(self, packed, signs):
        raise NotImplementedError("cpukern: pack_signs is not used by quantize_exl3")


def install(Q, codebook):
    """the stand-ins into quantize module Q (after rxq.engine's own patches); codebook i8 / i4 / mul1"""
    assert codebook in ("mul1", "i8", "i4"), f"cpukern: codebook {codebook}"
    if _STATE["installed"] is not None:
        assert _STATE["installed"] == codebook, "cpukern: one codebook a process"
        return Q
    tc = torch.cuda
    tc.synchronize = lambda *a, **k: None
    tc.stream = lambda *a, **k: contextlib.nullcontext()
    tc.Stream = lambda *a, **k: None
    tc.Event = lambda *a, **k: types.SimpleNamespace(record=lambda *x, **y: None, synchronize=lambda: None,
                                                     wait=lambda *x: None, elapsed_time=lambda *x: 0.0)
    orig_ldl, orig_pho = Q.block_ldl, Q.prepare_H_out
    keep = [False]

    def block_ldl(H, b, quant_args, verbose, debug_info=None):
        # the fork returns (L, H.cpu()) after overwriting H's storage with L (VRAM reuse); on a CPU tensor .cpu() is H
        # itself, so the "H" it returns would be L. Only prepare_H_out reads that H (its matrix: the two-sided proxy and
        # tile CD's M; scaled_ldl takes L only): there, keep a copy, as the card path does (one n^2 copy, not one a call)
        if H.device.type != "cpu" or not keep[0]: return orig_ldl(H, b, quant_args, verbose, debug_info)
        kept = H.clone()
        L, _ = orig_ldl(H, b, quant_args, verbose, debug_info)
        return L, kept

    def prepare_H_out(*a, **k):
        keep[0] = True
        try: return orig_pho(*a, **k)
        finally: keep[0] = False
    Q.block_ldl, Q.prepare_H_out = block_ldl, prepare_H_out
    Q.ext = _Ext(codebook)
    Q.frac_k = frac_k
    Q.get_quant_stream = lambda device: None
    Q.get_temp_buffers = lambda *a, **k: (None, None)
    Q.get_temp_buffers_frac = lambda *a, **k: (None, None)
    Q.quantize_tiles_multigpu = lambda tiles, quant_args: Q.quantize_tiles(tiles, quant_args)
    Q._rxq_kernels = MARK
    _STATE["installed"] = codebook
    return Q


def active():
    return _STATE["installed"] is not None
