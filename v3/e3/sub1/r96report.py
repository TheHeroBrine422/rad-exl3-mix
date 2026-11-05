#!/usr/bin/env python3
"""r96report.py [rowslab2.jsonl] -- #96 stage 1 numbers and verdict (README "#96").

Per family and overall: parameter-weighted median (the rule's statistic) and parameter-weighted mean of the row-slab
ceiling at S = 128 / 256 / 1024, for the three slab weights of rowslab2.py: in (E3c, input side only), out (Σdy² x
input norm per row: the rule's number) and rot (the same with EXL3's 128-row output Hadamard mixing).
Rule: stage 2 if the overall parameter-weighted median of out at S = 128 is >= 0.10 bit."""
import json, os, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/rowslab2.jsonl")
R = [json.loads(l) for l in open(path)]
FAM = ["ssm_qkv", "ssm_z", "ssm_out", "attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"]


def wmed(x, w):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w, float)[o]
    c = np.cumsum(w); return float(x[np.searchsorted(c, c[-1] / 2)])


def stats(rows, key):
    rows = [r for r in rows if key in r]
    if not rows: return None
    x = [r[key] for r in rows]; w = [r["params"] for r in rows]
    return wmed(x, w), float(np.average(x, weights=w))


cols = [(v, S) for S in (128, 256, 1024) for v in ("in", "out", "rot")]
print(f"#96 stage 1: {len(R)} tensors, {len({r['layer'] for r in R})} layers   (cells: weighted median / weighted mean, bits)")
print(f"{'family':12s} {'n':>3s} {'Mparam':>7s} " + " ".join(f"{v + str(S):>13s}" for v, S in cols))
groups = [(f, [r for r in R if r["w"] == f]) for f in FAM] + [("big (gu+dn)", [r for r in R if r["w"] in ("ffn_gate_up", "ffn_down")]), ("all", R)]
for name, rows in groups:
    if not rows: continue
    cells = []
    for v, S in cols:
        s = stats(rows, f"{v}_S{S}"); cells.append(f"{s[0]:6.3f}/{s[1]:6.3f}" if s else f"{'-':>13s}")
    print(f"{name:12s} {len(rows):3d} {sum(r['params'] for r in rows) / 1e6:7.0f} " + " ".join(cells))
med = stats(R, "out_S128")[0]; rmed = stats(R, "rot_S128")[0]
print(f"\nRule (stage 2 if the overall weighted median of out_S128 >= 0.10 bit): {med:.3f} bit -> "
      f"{'PASS: stage 2' if med >= 0.10 else 'FAIL: no stage 2'}   (rotation-aware rot_S128: {rmed:.3f})")
top = sorted(R, key=lambda r: -r["out_S128"])[:10]
print("largest out_S128: " + ", ".join(f"L{r['layer']} {r['w']} {r['out_S128']:.2f} (in {r['in_S128']:.2f})" for r in top))
