#!/usr/bin/env python3
"""reval.py MODE LAYER TENSOR K OUT.jsonl [ROWS] -- w46: w43's settle.py (proxy v2, exllamav3's own transforms, scale
search and per-output-channel refit) with the codebooks and one mode the re-validations need (reval.md). settle.py
itself is unchanged; every settle.py MODE (gain / damp / cd / selftest) runs as there.

Codebooks added to SETTLE_CB (besides trellis2.codebook's mul1, mul1_i8, int4grid, cubicC_i8, ...):
  iqrh, iqrh:A:B   #100's IQR-style computed code (trellis4.py, the same values r100 used)
  mul1x, mul1x_i8  exllamav3's exact fp16 mul1 / int8-rounded mul1 (w45's r94b/cbx.py)
  int4x            the fork's EXL3_MUL1_I4 (codebook.cuh, 08004077): fp16 mul1x value v, q = clamp(rint(v * 7.5 / 2.6
                   - 0.5), -8, 7) + 0.5, value fp16(q * 2.6 / 7.5); not restandardised (the g_scale search absorbs it)

MODE order (#106, actblk/README.md): LDLQ over the input 128-column blocks in a chosen order, at m = 1 (exllamav3's
own g_scale), adopted damping. The blocks are permuted in the transformed basis: the input Hadamard and su act per
column / per 128-block, so moving a whole 128-block of the transformed rows and of the scaled metric is exactly
quantizing the column-permuted tensor with its signs carried along (rows, signs, scale and tile sample identical
across arms: paired). Codes are permuted back and scored in the original order (rel after the refit, rel_norefit).
Block importance = trace of the block's diagonal 128x128 block of the metric LDLQ runs under
(P diag(su) (H + delta I) diag(su) P; a block's trace is invariant under its own Hadamard). Arms:
  natural  identity order (must reproduce settle.py gain's rel at m = 1)
  head     the order a head permutation can express: heads = HEAD_DIM-column groups, heads sharing a K/V head
           (attention: 24 q heads / 4 kv heads, head_dim 256; GDN out_proj: 48 v heads / 16 k heads, dim 128;
           grouped contiguously) form a group. Groups by descending total trace, heads inside a group by descending
           trace, a head's own blocks in natural order. Exact: o_proj columns + q/gate rows (+ k/v rows for the
           group order); out_proj columns + the v / z / a / b head rows, A_log, dt_bias, conv channels (+ q/k heads).
  free     all 128-blocks by descending trace, no head constraint (an upper bound; not expressible: it splits heads)
  rev      ascending trace inside the head constraint (a sign check: should be worse than natural)"""
import json, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_h, os.path.join(_h, ".."), os.path.join(_h, "../..")]
import settle, trellis, trellis2, trellis4
torch.set_grad_enabled(False)

_base = trellis2.codebook


def _f16(bits): return float(np.frombuffer(np.uint16(bits).tobytes(), np.float16)[0])


def mul1x():
    s = np.arange(65536, dtype=np.uint64)
    h = (s * 0x83DCD12D) & 0xFFFFFFFF
    bs = sum(((h >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
    return (np.float16(1024 + bs).astype(np.float64) * _f16(0x1eee) + _f16(0xc931)).astype(np.float16)


def codebook(name):
    if name == "iqrh" or name.startswith("iqrh:"): return trellis4.codebook(name)
    if name == "mul1x": return mul1x().astype(np.float64)
    if name == "mul1x_i8":
        s4 = np.float32(4.0) * np.float32(_f16(0x1eee)); inv = np.float32(1.0) / s4
        q = np.clip(np.rint(mul1x().astype(np.float32) * inv), -127, 127).astype(np.float32)
        return (q * s4).astype(np.float16).astype(np.float64)
    if name == "int4x":
        st, inv = np.float32(2.6 / 7.5), np.float32(7.5 / 2.6)
        q = np.clip(np.rint(mul1x().astype(np.float32) * inv - np.float32(0.5)), -8, 7) + np.float32(0.5)
        return (q * st).astype(np.float16).astype(np.float64)
    return _base(name)


trellis2.codebook = codebook                               # settle.main reads trellis2.codebook(SETTLE_CB)

HEADS = {"attn_output": (256, 6), "ssm_out": (128, 3)}     # (head dim, q / v heads per k-v group)


def orders(tr, tensor):
    """tr: trace per 128-block -> {arm: block order}"""
    nb = len(tr); hd, gs = HEADS[tensor]; bph = hd // 128
    heads = [list(range(h * bph, (h + 1) * bph)) for h in range(nb // bph)]
    ht = [sum(tr[b] for b in hb) for hb in heads]
    groups = [list(range(g * gs, (g + 1) * gs)) for g in range(len(heads) // gs)]
    def constrained(desc):
        sg = sorted(groups, key=lambda g: -sum(ht[h] for h in g) if desc else sum(ht[h] for h in g))
        return [b for g in sg for h in sorted(g, key=lambda h: -ht[h] if desc else ht[h]) for b in heads[h]]
    return {"natural": list(range(nb)), "head": constrained(True), "free": sorted(range(nb), key=lambda b: -tr[b]),
            "rev": constrained(False)}


def order_mode(layer, tensor, K, out, rows):
    assert tensor in HEADS, tensor
    seed, rseed = int(os.environ.get("E3_SEED", 0)), int(os.environ.get("E3_ROWSEED", 0))
    t00 = time.time()
    p = settle.P2(layer, tensor, rows, seed, rseed)
    cbn = os.environ.get("SETTLE_CB", "mul1_i8")
    cb = np.ascontiguousarray(codebook(cbn), dtype=np.float32)
    damp0 = 0.025                                          # both tensors are in the "rest" group
    s_exl, sx = settle.search_scale(p.T, cb, K)
    Hq, dq = p.metric(damp0)
    nb = p.Kd // 128
    tr = [float(Hq.diagonal()[b * 128:(b + 1) * 128].sum()) for b in range(nb)]
    trh = [float(p.H.diagonal()[b * 128:(b + 1) * 128].sum()) for b in range(nb)]    # plain H (census's measure)
    arms = os.environ.get("ORDER_ARMS", "natural,head,free,rev").split(",")
    od = orders(tr, tensor)
    rel, rel0, sec = {}, {}, {}
    for arm in arms:
        t0 = time.time()
        cols = torch.cat([torch.arange(b * 128, (b + 1) * 128) for b in od[arm]])
        Q = trellis.ldlq_trellis(p.Wt[:, cols] * s_exl, Hq[cols][:, cols], cb, K, dq)
        Qn = torch.empty_like(Q); Qn[:, cols] = Q
        rel[arm], rel0[arm] = settle.rels(p, Qn / s_exl, damp0)[0]
        sec[arm] = round(time.time() - t0, 1)
    sp = lambda t: float(np.percentile(t, 90) / max(np.percentile(t, 10), 1e-300))
    o = {"exp": "order", "proxy": 2, "layer": layer, "w": tensor, "bits": K, "cb": cbn, "s_exl": s_exl,
         "drift": settle.DRIFT.get(float(K), 1.0), "search": sx, "seed": seed, "rowseed": rseed, "rows": rows,
         "N": p.N, "K": p.Kd, "damp": damp0, "rel": rel, "rel_norefit": rel0, "sec_arm": sec,
         "orders": {a: od[a] for a in arms}, "trace": tr, "trace_h": trh, "spread_p90p10": sp(tr),
         "spread_h_p90p10": sp(trh), **p.extra, "sec_job": round(time.time() - t00, 1)}
    with open(out + ".part", "w") as fh: fh.write(json.dumps(o) + "\n")
    os.replace(out + ".part", out)
    print(json.dumps({k: v for k, v in o.items() if k not in ("search", "orders", "trace", "trace_h")}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "order":
        order_mode(int(sys.argv[2]), sys.argv[3], float(Fraction(sys.argv[4])), sys.argv[5],
                   int(sys.argv[6]) if len(sys.argv) > 6 else 128)
    elif sys.argv[1] == "cbcheck":                         # value sets of the codebooks (VM-quick)
        for n in sys.argv[2:]:
            c = codebook(n); u = np.unique(c)
            print(f"{n:16s} mean {c.mean():+.5f} sd {c.std():.5f} levels {len(u):6d} range {c.min():+.4f} .. {c.max():+.4f}")
    else:
        settle.main()
