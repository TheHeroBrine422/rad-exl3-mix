#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""gdncheck.py dump OUT.pt MODEL LEVEL GOLDEN POOLS [--layers 4] [--docs 6]  /  gdncheck.py cmp A.pt B.pt
The GDN core path check: run once with transformers' torch chunk rule and once with flash-linear-attention on
PYTHONPATH (the import decides which one transformers binds), dump per-document KL (qad.py's loss) and the summed
gradient of every trainable vector; cmp prints the relative KL difference and the gradient cosine per vector
(PASS: |dKL| <= max(1% of KL, 1e-5) per document and gradient cosine >= 0.99 overall)."""
import json, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if sys.argv[1] == "cmp":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    print(f"paths: {a['gdn']} vs {b['gdn']}")
    worst = 0.0
    for (i, ka), (_, kb) in zip(a["kl"], b["kl"]):
        r = abs(ka - kb) / max(abs(ka), 1e-12); worst = max(worst, min(r, abs(ka - kb) / 1e-5 * 0.01))
        print(f"  doc {i}: KL {ka:.6g} vs {kb:.6g} (rel {r:.2e})")
    ga = torch.cat([a["g"][n].flatten() for n in sorted(a["g"])]); gb = torch.cat([b["g"][n].flatten() for n in sorted(a["g"])])
    cos = float((ga * gb).sum() / (ga.norm() * gb.norm()))
    low = sorted((float((a["g"][n] * b["g"][n]).sum() / (a["g"][n].norm() * b["g"][n].norm() + 1e-30)), n) for n in a["g"])[:5]
    print(f"gradient cosine overall {cos:.5f}; lowest vectors: " + ", ".join(f"{n.split('layers.')[-1]} {c:.4f}" for c, n in low))
    print("GDNCHECK " + ("PASS" if worst <= 0.01 and cos >= 0.99 else "FAIL"))
    sys.exit(0)

import qmodel  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as q35  # noqa: E402
from qad import Golden, pick_docs, window_loss  # noqa: E402

out, model, level, golden, pools = sys.argv[2:7]
L = int(os.environ.get("GC_LAYERS", 4)); nd = int(os.environ.get("GC_DOCS", 6))
import importlib.util
path = "fla" if importlib.util.find_spec("fla") else "torch"
g = Golden(golden); pl = json.load(open(pools))
docs = pick_docs(g, pl, "qad", nd, 0, 2048)
stu = qmodel.Student(model, level, ["cuda:0"], layers=L)
kl = []
for i in docs:
    k, m, _ = window_loss(stu, g, i, 2048)
    (k / m).backward(); kl.append((i, float(k) / m))
torch.save({"gdn": path, "kl": kl, "g": {n: p.grad.detach().cpu().double() for n, p in stu.P.items()}}, out)
print(f"dumped {len(docs)} docs, path {path}")
