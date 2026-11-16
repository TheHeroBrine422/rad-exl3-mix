#!/usr/bin/env python3
"""cmp200.py A_DIR B_DIR -- #200's unit check: the trainer states (state.pt: the student's scales and the optimizer state)
and the per-step train records (train.jsonl: rows, lr, train_kl, per-corpus KL and rows) of two runs, compared bitwise.
Timing and memory fields are ignored. Exit 0 = identical."""
import json, os, sys
import torch

def walk(a, b, path, out):
    if isinstance(a, dict):
        if set(a) != set(b): out.append(f"{path}: keys differ"); return
        for k in a: walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b): out.append(f"{path}: length differs"); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]", out)
    elif torch.is_tensor(a):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()): out.append(f"{path}: tensor differs")
    elif a != b: out.append(f"{path}: {a!r} != {b!r}")

A, B = sys.argv[1], sys.argv[2]
out = []
sa = torch.load(os.path.join(A, "state.pt"), weights_only=False, map_location="cpu")
sb = torch.load(os.path.join(B, "state.pt"), weights_only=False, map_location="cpu")
walk(sa["snap"], sb["snap"], "snap", out)
n_t = 0
KEEP = ("step", "rows", "lr", "train_kl", "ema", "by", "n", "prefix_tok", "what")
ra = [json.loads(x) for x in open(os.path.join(A, "train.jsonl"))]
rb = [json.loads(x) for x in open(os.path.join(B, "train.jsonl"))]
for i, (x, y) in enumerate(zip(ra, rb)):
    n_t += 1
    for k in KEEP:
        if x.get(k) != y.get(k): out.append(f"train.jsonl[{i}] {k}: {x.get(k)!r} != {y.get(k)!r}")
if len(ra) != len(rb): out.append(f"train.jsonl lengths {len(ra)} vs {len(rb)}")
print(json.dumps({"A": A, "B": B, "train_records": n_t, "steps_A": sa["st"]["step"], "steps_B": sb["st"]["step"],
                  "differences": out[:20], "n_differences": len(out), "identical": not out}, indent=1))
sys.exit(0 if not out else 1)
