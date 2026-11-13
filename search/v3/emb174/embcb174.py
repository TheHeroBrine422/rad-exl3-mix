#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""embcb174.py E3DIR CODEBOOK ROWS OUT.jsonl -- (w135, #174; lead note 11-13: the embedding's code on KL per byte
only) v3/e3/embtrellis.py's trellis embedding sweep with another trellis codebook (trellis.py's names: mul1, mul1_i8,
gauss_i8, ...) instead of its hard-wired mul1: trellis.codebook("mul1") is redirected before embtrellis runs, nothing
else changes (same seeded rows, rotation, per-row scale search, metric rel = sum (w - q)^2 / sum w^2, H = I). Env as
embtrellis.py (E3_CKPT, EMB_KS, EMB_LLOYD=0, OMP_NUM_THREADS)."""
import os, runpy, sys

e3dir, cbname = sys.argv[1], sys.argv[2]
sys.path.insert(0, e3dir)
import trellis as T          # noqa: E402

_orig = T.codebook
T.codebook = lambda name: _orig(cbname if name == "mul1" else name)
sys.argv = [os.path.join(e3dir, "embtrellis.py")] + sys.argv[3:]
runpy.run_path(sys.argv[0], run_name="__main__")
