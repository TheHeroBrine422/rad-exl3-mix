# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Stage timers for one quant job: exclusive wall time per stage, the device synchronized at every boundary (so GPU
work is charged to the stage that queued it). Timing only: the wrapped functions get the same arguments and return
the same values, so codes are unchanged with the profiler on (checked: tests/check_quant.py runs with it on).

Stages (labels): hessian_load, weight_load, hout_load, finalize (H scaling, damping, sign + Hadamard rotation, and
the shared block LDL without scaled_hessian), regularize (output / input scales, Hadamards), scale_search
(g_scale_gss minus its trellis calls), trellis[scale_search], scaled_ldl (unrotate H, per-tensor block LDL),
prepare_hout, ldlq (LDLQ feedback GEMMs; ldlq_2hess for H_out), trellis[ldlq], tile_cd + trellis[tile_cd], refit,
pack, other (rest of quantize_exl3: proxy metrics, rotations back, scale vectors), save.
The trellis search is exllamav3's quantize_tiles_multigpu wherever it is called; it is charged as trellis[<caller>]."""
import time
from collections import defaultdict

import torch


class Prof:
    def __init__(self, device=None, sync=True):
        self.dev, self.sync = device, sync
        self.t = defaultdict(float); self.n = defaultdict(int); self.stack = []

    def _sync(self):
        if self.sync and self.dev is not None and torch.device(self.dev).type == "cuda": torch.cuda.synchronize(self.dev)

    def enter(self, label):
        self._sync()
        if label == "trellis" and self.stack: label = f"trellis[{self.stack[-1][0]}]"
        self.stack.append([label, time.perf_counter(), 0.0])

    def exit(self):
        self._sync()
        label, t0, child = self.stack.pop()
        dt = time.perf_counter() - t0
        self.t[label] += dt - child; self.n[label] += 1
        if self.stack: self.stack[-1][2] += dt

    def stage(self, label):
        p = self

        class _C:
            def __enter__(self): p.enter(label)
            def __exit__(self, *e): p.exit()
        return _C()

    def wrap(self, mod, name, label):
        orig = getattr(mod, name)
        if getattr(orig, "_rxq_prof", None) is not None: orig = orig._rxq_orig
        p = self

        def f(*a, **k):
            p.enter(label)
            try: return orig(*a, **k)
            finally: p.exit()
        f._rxq_prof, f._rxq_orig = self, orig
        setattr(mod, name, f)

    def install(self, Q):
        """wrap exllamav3's stage functions in the quantize module's namespace (they call each other through it)"""
        for name, label in (("finalize_capture_H", "finalize"), ("regularize", "regularize"), ("g_scale_gss", "scale_search"),
                            ("scaled_ldl", "scaled_ldl"), ("prepare_H_out", "prepare_hout"), ("ldlq", "ldlq"),
                            ("ldlq_2hess", "ldlq"), ("quantize_tiles_multigpu", "trellis"), ("refit_scales", "refit"),
                            ("pack_trellis", "pack")):
            if hasattr(Q, name): self.wrap(Q, name, label)
        return self

    def reset(self):
        self.t.clear(); self.n.clear()

    def record(self):
        return {k: round(v, 4) for k, v in sorted(self.t.items())}
