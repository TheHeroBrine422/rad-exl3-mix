# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""exllamav3's quantizer, set up exactly as rad-exl3-mix quantize/ourq.py sets it up (ported verbatim, same order):

1. block_trace -> 1.0 unless block_trace = true (it only feeds a logged proxy the refit overwrites: output-neutral)
2. refit_scales wrapped: empty the cache and warm the refit kernels first; with RXQ_REFIT_CPU_FALLBACK=1 (or ourq's
   OURQ_REFIT_CPU_FALLBACK=1) an out-of-memory refit runs on the CPU in fp32 (small cards, 27B down_proj)
3. LDLQ drift interpolated log-linearly between exllamav3's table points (drift = "interp"; "exact" = the lookup)
4. kernel swaps: exl3frac (fractional rates), exl3cb (rounded codebooks), exl3frac2 (qtf2), all from rad-exl3-mix
   v3/e3/ built at run time against the fork's exllamav3_ext sources
5. optional: tile coordinate descent (rxq/tilecd.py: w50's module merged, plus the H_out objective), the gain wrapper

Paths (env): RXQ_RXM = rad-exl3-mix checkout (default: the checkout this package lives in), RXQ_EXL3_SRC = the fork's
exllamav3_ext source dir (ourq's --frac-ext). tile_cd = (sweeps, families[, module path]): the path loads another
tilecd.py instead of rxq's (settings quant.tile_cd.module; comparisons against the research copy only).
setup() is idempotent per (codebook, qtf2, drift, block_trace, tile_cd) and refuses a second, different setup in one
process (the swaps are global to the exllamav3 module)."""
import glob, hashlib, math, os, sys

import torch

_STATE = {}


def rxm_root():
    return os.environ.get("RXQ_RXM", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def exl3_src():
    s = os.environ.get("RXQ_EXL3_SRC")
    if not s: raise SystemExit("RXQ_EXL3_SRC: the exllamav3 fork's exllamav3_ext source dir (ourq --frac-ext)")
    return s


def _warm_refit(dev_):
    t = torch.rand(4, 4, device=dev_) + torch.eye(4, device=dev_)
    r = torch.linalg.solve(t, torch.ones(4, device=dev_))
    r = torch.where(torch.isfinite(r) & (r > 0), r, torch.ones_like(r)); _ = (t.diagonal().mean(), (t * t).sum(0), t @ t.T)
    torch.cuda.synchronize(dev_)


def _kernel_dirs():
    e3 = os.path.join(rxm_root(), "v3", "e3")
    return {m: os.path.join(e3, m) for m in ("exl3frac", "exl3cb", "exl3frac2")}


def setup(device, codebook="i8", qtf2=True, drift="interp", block_trace=False, tile_cd=None, frac=True):
    """returns the exllamav3 quantize module, patched. tile_cd: None or (sweeps, families list[, module path])"""
    key = (codebook, bool(qtf2), drift, bool(block_trace), None if tile_cd is None else (tile_cd[0], tuple(tile_cd[1]), *tile_cd[2:]), bool(frac))
    if _STATE.get("key") is not None:
        if _STATE["key"] != key: raise RuntimeError(f"engine already set up as {_STATE['key']}, asked for {key}: one setup a process")
        _STATE["dev"][0] = device
        return _STATE["Q"]
    import exllamav3.modules.quant.exl3_lib.quantize as Q
    dev = [device]
    if not block_trace: Q.block_trace = lambda A, B, block_size=1024: 1.0
    orig_refit = Q.refit_scales

    def _refit(*args, **kw):
        torch.cuda.empty_cache(); _warm_refit(dev[0]); torch.cuda.empty_cache()
        try:
            return orig_refit(*args, **kw)
        except torch.OutOfMemoryError:
            if not (os.environ.get("RXQ_REFIT_CPU_FALLBACK") or os.environ.get("OURQ_REFIT_CPU_FALLBACK")): raise
            torch.cuda.empty_cache(); print("refit_scales: OOM on the card, refit on the CPU", flush=True)
            mv = lambda x, d: x.to(d) if torch.is_tensor(x) else x
            r = orig_refit(*[mv(x, "cpu") for x in args], **{k: mv(v, "cpu") for k, v in kw.items()})
            return tuple(mv(x, dev[0]) for x in r) if isinstance(r, tuple) else mv(r, dev[0])
    Q.refit_scales = _refit
    if drift == "interp":
        _DT = sorted({**Q.LDLQ_DRIFT, 4: 1.0}.items())

        def _drift(K):
            K = float(K)
            if K in dict(_DT): return dict(_DT)[K]
            if K <= _DT[0][0]: return _DT[0][1]
            if K >= _DT[-1][0]: return 1.0
            (k0, d0), (k1, d1) = next((p, q) for p, q in zip(_DT, _DT[1:]) if p[0] < K < q[0])
            return math.exp(math.log(d0) + (math.log(d1) - math.log(d0)) * (K - k0) / (k1 - k0))
        Q.ldlq_drift = _drift
    kd = _kernel_dirs()
    if frac:
        sys.path.insert(0, kd["exl3frac"]); import exl3frac
        exl3frac.install(exl3_src())
    if codebook != "mul1":
        assert frac, "a rounded codebook needs the frac kernels (ourq: --codebook needs --frac-ext)"
        sys.path.insert(0, kd["exl3cb"]); import exl3cb
        exl3cb.install(exl3_src(), codebook)
    if qtf2:                                             # i4: w53's qtf2-i4.patch (exl3frac2 "i4" variant, 11-07)
        assert frac and codebook in ("mul1", "i8", "i8c", "i4"), "qtf2 needs the frac kernels and the mul1, i8, i8c or i4 codebook"
        sys.path.insert(0, kd["exl3frac2"]); import exl3frac2
        exl3frac2.install(exl3_src(), codebook)
    tcd = None
    if tile_cd is not None:                              # rxq's own (merged 11-07, H_out-aware); a module path = another copy
        mod = tile_cd[2] if len(tile_cd) > 2 else ""     # (comparisons: the research module v3/e3/tilecd/tilecd.py)
        if mod:
            import importlib.util
            sp = importlib.util.spec_from_file_location("tilecd_ext", mod)
            tcd = importlib.util.module_from_spec(sp); sp.loader.exec_module(tcd)
        else:
            from . import tilecd as tcd
        tcd.install(Q, int(tile_cd[0]), ",".join(tile_cd[1]))
    _warm_refit(device)                                  # load the refit kernels while VRAM is empty (ourq does it here)
    _STATE.update(key=key, Q=Q, dev=dev, tilecd=tcd, gain=[1.0])
    return Q


def set_gain(Q, g):
    """ourq --gain: a multiplier on exllamav3's g_scale for the next tensor (wrapper installed on first use)"""
    if g != 1.0 and not getattr(Q.g_scale_gss, "_rxq_gain", False):
        orig = Q.g_scale_gss

        def _gss(*x, **k):
            g_, m_ = orig(*x, **k); return g_ * _STATE["gain"][0], m_
        _gss._rxq_gain = True
        Q.g_scale_gss = _gss
    _STATE["gain"][0] = g


def tilecd():
    return _STATE.get("tilecd")


def code_stamp(Q):
    """hash of the code that decides the results: rxq's own modules, exllamav3's quantize.py, the kernel sources
    (the fork's exllamav3_ext and rad-exl3-mix's run-time builds) and tilecd.py if installed"""
    h = hashlib.sha256(); files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py")))
    files.append(Q.__file__)
    src = os.environ.get("RXQ_EXL3_SRC")
    if src: files += sorted(f for e in ("cu", "cuh", "cpp", "h", "py") for f in glob.glob(f"{src}/**/*.{e}", recursive=True))
    for d in _kernel_dirs().values(): files += sorted(glob.glob(os.path.join(d, "*.*")))
    t = tilecd()
    if t is not None and not t.__file__.startswith(os.path.dirname(os.path.abspath(__file__))): files.append(t.__file__)
    for f in files:
        if os.path.isfile(f) and not f.endswith(".pyc"): h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    return h.hexdigest()[:16]
