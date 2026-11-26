#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkstate233.py SRC_STATE.pt STEP OUT_STATE.pt -- w233 profiling: a copy of a qad220 state.pt that resumes at plan step STEP (parameters / Adam as
saved; the trainer rebuilds every stream's carried state from its tokens: a mid-run window of a plan without training up to it). Profiling only."""
import sys
import torch
src, step, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
z = torch.load(src, weights_only=False); z["st"]["step"] = step; z["st"]["done_br"] = []
torch.save(z, out + ".part"); import os; os.replace(out + ".part", out); print(f"{src} -> {out}: resumes at step {step}")
