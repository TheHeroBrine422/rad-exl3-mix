#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""eq215.py REF RUN [RUN ...] -- w215's equivalence gate for trainer-speed changes: each RUN's final trained log
multipliers (OUT/state.pt, snap.cur) vs REF's: max |delta|, relative L2 of delta (|d| / |ref|), the largest per-tensor
relative L2; the per-step training KL (train.jsonl) side by side (max relative step difference); speed from speed215.json
(positions / s and gen rows / s over steps 3.., peak VRAM)."""
import json, os, sys

import torch


def load(d):
    z = torch.load(os.path.join(d, "state.pt"), weights_only=False, map_location="cpu")
    tr = [json.loads(l) for l in open(os.path.join(d, "train.jsonl"))]
    sp = json.load(open(os.path.join(d, "speed215.json"))) if os.path.exists(os.path.join(d, "speed215.json")) else {}
    if sp:                                              # rates over steps 3.. (steps 1-2: kernel tuning / warm-up)
        w = sum(sp["walls"][2:]); sp["pos_s"] = sum(sp["pos"][2:]) / w; sp["rows_s"] = sum(sp["rows"][2:]) / w
    return z["snap"]["cur"], [r["train_kl"] for r in tr if r["what"] == "main"], sp


def main():
    ref, rkl, rsp = load(sys.argv[1])
    rn = torch.sqrt(sum((v.double() ** 2).sum() for v in ref.values()))
    print(f"ref {sys.argv[1]}: {len(ref)} tensors, |theta| {float(rn):.6e}, {len(rkl)} steps, "
          f"{rsp.get('pos_s', 0):.1f} pos/s, {rsp.get('rows_s', 0):.1f} rows/s, peak {rsp.get('peak_gib', 0):.2f} GiB")
    for d in sys.argv[2:]:
        cur, kl, sp = load(d)
        assert cur.keys() == ref.keys(), d
        mx, d2, worst = 0.0, 0.0, (0.0, "")
        for k, v in ref.items():
            dd = (cur[k].double() - v.double())
            mx = max(mx, float(dd.abs().max())); d2 += float((dd ** 2).sum())
            rel = float(dd.norm() / v.double().norm().clamp_min(1e-30))
            if rel > worst[0]: worst = (rel, k)
        n = min(len(kl), len(rkl))
        sk = max(abs(a - b) / abs(b) for a, b in zip(kl[:n], rkl[:n])) if n else float("nan")
        print(f"{d}: max |d log mult| {mx:.3e}, rel L2 {d2 ** 0.5 / float(rn):.3e} (worst tensor {worst[0]:.3e} {worst[1]}); "
              f"steps {len(kl)}, max rel step-KL diff {sk:.3e} (last {kl[n - 1]:.6f} vs {rkl[n - 1]:.6f}); "
              f"{sp.get('pos_s', 0):.1f} pos/s ({sp.get('pos_s', 0) / rsp.get('pos_s', 1):.2f}x), "
              f"{sp.get('rows_s', 0):.1f} rows/s, peak {sp.get('peak_gib', 0):.2f} GiB")


if __name__ == "__main__":
    main()
