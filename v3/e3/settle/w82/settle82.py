#!/usr/bin/env python3
"""settle82.py rperm LAYER TENSOR K OUT.jsonl [ROWS] -- #117 1(b) (../w82.md): settle64.py (exllamav3's last-to-first
LDLQ block order, SETTLE_ORDER=rev) with a mode for a residual-stream 128-block permutation. Every other settle64 /
settle.py MODE passes through unchanged.

TENSOR also ffn_gate / ffn_up (gate_proj / up_proj on their own: settle.P2's ffn_gate_up alternates them by seed).
Mode rperm: proxy v2 rows and metric in the natural order (m = 1, adopted damping: 0.005 input group, 0.025 rest);
each arm permutes whole 128-blocks of Wt's columns and of the metric, quantizes, un-permutes, scores (w64's `order`
mode with the family's damping). Arms = stored orders (exllamav3 quantizes the LAST stored block first):
  natural; glob = W82_ORDER's stored_order (agree82.py consensus, heaviest last); own = this tensor's metric block
  trace ascending (heaviest last; oracle); globdesc = glob reversed (heaviest first). W82_ARMS picks a subset."""
import json, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_h, os.path.join(_h, ".."), os.path.join(_h, "../w64"), os.path.join(_h, "../..")]
import settle64                                              # patches trellis.ldlq_trellis (SETTLE_ORDER) and the codebooks
import e3, settle, trellis
torch.set_grad_enabled(False)
e3.ROUTE["ffn_gate"] = (["mlp.gate_proj"], "post_attention.input")
e3.ROUTE["ffn_up"] = (["mlp.up_proj"], "post_attention.input")


def mode_rperm(p, cb, K, damp0, base):
    s_exl, sx = settle.search_scale(p.T, cb, K)
    Hq, dq = p.metric(damp0); nb = p.Kd // 128
    tr = np.array([float(Hq.diagonal()[b * 128:(b + 1) * 128].sum()) for b in range(nb)])
    g = json.load(open(os.environ["W82_ORDER"]))["stored_order"]
    od = {"natural": list(range(nb)), "glob": g, "own": [int(b) for b in np.argsort(tr)], "globdesc": g[::-1]}
    arms = os.environ.get("W82_ARMS", "natural,glob,own,globdesc").split(",")
    rel, rel0, sec = {}, {}, {}
    for arm in arms:
        t0 = time.time()
        cols = torch.cat([torch.arange(b * 128, (b + 1) * 128) for b in od[arm]])
        Q = trellis.ldlq_trellis(p.Wt[:, cols] * s_exl, Hq[cols][:, cols], cb, K, dq)
        Qn = torch.empty_like(Q); Qn[:, cols] = Q
        rel[arm], rel0[arm] = settle.rels(p, Qn / s_exl, damp0)[0]
        sec[arm] = round(time.time() - t0, 1)
    return {**base, "s_exl": s_exl, "damp": damp0, "rel": rel, "rel_norefit": rel0, "sec_arm": sec,
            "orders": {a: od[a] for a in arms}, "trace": tr.tolist(),
            "spread_p90p10": float(np.percentile(tr, 90) / np.percentile(tr, 10))}


def main():
    if sys.argv[1] != "rperm": return settle64.main()
    layer, tensor, K, out = int(sys.argv[2]), sys.argv[3], float(Fraction(sys.argv[4])), sys.argv[5]
    rows = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    seed, rseed = int(os.environ.get("E3_SEED", 0)), int(os.environ.get("E3_ROWSEED", 0))
    t00 = time.time()
    p = settle.P2(layer, tensor, rows, seed, rseed)
    cbn = os.environ.get("SETTLE_CB", "mul1x_i8")
    cb = np.ascontiguousarray(settle64.codebook(cbn), dtype=np.float32)
    damp0 = 0.005 if tensor in settle.INPUT else 0.025
    base = {"exp": "rperm", "order": settle64.ORDER, "proxy": 2, "layer": layer, "w": tensor, "bits": K, "cb": cbn,
            "seed": seed, "rowseed": rseed, "rows": rows, "N": p.N, "K": p.Kd, **p.extra}
    o = mode_rperm(p, cb, K, damp0, base)
    o["sec_job"] = round(time.time() - t00, 1)
    with open(out + ".part", "w") as fh: fh.write(json.dumps(o) + "\n")
    os.replace(out + ".part", out)
    print(json.dumps(o), flush=True)


if __name__ == "__main__":
    main()
