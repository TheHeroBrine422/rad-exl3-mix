#!/usr/bin/env python3
"""settle64.py MODE LAYER TENSOR K OUT.jsonl [ROWS] -- w64 (../w64.md): w43's settle.py with exllamav3's LDLQ block
order, exact / clipped-i8 codebooks and a mode for bf16 (row, 128) scales. settle.py itself is unchanged; every
settle.py MODE (gain / damp / cd / selftest) runs as there, through the patched functions below.

SETTLE_ORDER  rev (default): LDLQ quantizes the 16-column input blocks last to first, as exllamav3's ldlq()
              (`for j in range(size_k, 0, -buf_size_k)` / `for bj in range(buf_size_k, 0, -16)`). The blocks are
              permuted to reverse order with each block's own column order kept (every 16x16 tile is the same tile),
              trellis.ldlq_trellis runs under the permuted metric, the codes are permuted back. Block LDLQ in a block
              order is GPTQ in that order, so this is ldlq() up to rounding. CD sweeps (mode cd) stay ascending, as
              tilecd.py on the card. fwd: settle.py's order (bit for bit).
SETTLE_CB     besides trellis2.codebook's names: mul1x / mul1x_i8 (cbx.py: the fork's exact values) and
              mul1x_i8c<A> (item 2): v = mul1x, s4 = 4 k_inv (fp32), value fp16(clamp(rint(v * A / s4), -127, 127)
              * s4 / A): the int8 step s4 / A, values beyond 127 steps clipped (A = 1 is mul1x_i8 exactly).
MODE rs       (item 4) bf16 scales per (stored row, 128-column block) of the transformed weight, cost 0.125 bpw, at
              equal bits: plain@K, plain@K-1/8, post@K-1/8 (plain codes, then the row's block scales solved jointly
              under the LDLQ metric, bf16), pre@K-1/8 (each (row, block) divided by its RMS (bf16) before the
              trellis; g_scale searched on the normalised rows' tiles; LDLQ with the feedback in the unscaled domain),
              prepost@K-1/8 (pre, then the post refit). m = 1, adopted damping; rel / rel_norefit as settle.py (after
              / before the per-output-channel refit). One JSON line with every arm.
Output as settle.py: OUT written whole at the end (OUT.part, then renamed)."""
import json, os, sys, time
from fractions import Fraction
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_h, os.path.join(_h, ".."), os.path.join(_h, "../..")]
import settle, trellis, trellis2
torch.set_grad_enabled(False)

ORDER = os.environ.get("SETTLE_ORDER", "rev")
_ldlq_fwd = trellis.ldlq_trellis
_cb_base = trellis2.codebook


def blockrev(Kd):
    """column permutation: 16-column blocks in reverse order, columns inside a block in their own order (involution)"""
    return torch.arange(Kd).reshape(Kd // 16, 16).flip(0).reshape(-1)


def ldlq_rev(W, H, cb, K, damp=0.01):
    p = blockrev(W.shape[1])
    return _ldlq_fwd(W[:, p].contiguous(), H[p][:, p].contiguous(), cb, K, damp)[:, p].contiguous()


if ORDER == "rev": trellis.ldlq_trellis = ldlq_rev            # settle.main and trellis2.ldlq_cd look it up at call time
elif ORDER != "fwd": raise SystemExit(f"SETTLE_ORDER {ORDER}")


def _f16(bits): return float(np.frombuffer(np.uint16(bits).tobytes(), np.float16)[0])


def codebook(name):
    if not name.startswith("mul1x"): return _cb_base(name)
    s = np.arange(65536, dtype=np.uint64)
    h = (s * 0x83DCD12D) & 0xFFFFFFFF
    bs = sum(((h >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
    v = (np.float16(1024 + bs).astype(np.float64) * _f16(0x1eee) + _f16(0xc931)).astype(np.float16)
    if name == "mul1x": return v.astype(np.float64)
    a = 1.0 if name == "mul1x_i8" else float(name[len("mul1x_i8c"):]) if name.startswith("mul1x_i8c") else None
    if a is None: raise SystemExit(f"unknown codebook {name}")
    s4 = np.float32(4.0) * np.float32(_f16(0x1eee))
    if a == 1.0:                                           # cbx.py's mul1x_i8, literally
        q = np.clip(np.rint(v.astype(np.float32) * (np.float32(1.0) / s4)), -127, 127).astype(np.float32)
        return (q * s4).astype(np.float16).astype(np.float64)
    st = np.float32(s4 / np.float32(a))
    q = np.clip(np.rint(v.astype(np.float32) * (np.float32(1.0) / st)), -127, 127).astype(np.float32)
    return (q * st).astype(np.float16).astype(np.float64)


trellis2.codebook = codebook


# ---- item 4: (row, 128) scales -------------------------------------------------------------------------------------
def ldlq_scaled(W, H, cb, K, damp, S):
    """trellis.ldlq_trellis with a scale per (row, column): the trellis codes (target / S) and Q = S * code; the
    error feedback runs in W's (unscaled) domain. Block order as ORDER. S [R, Kd]"""
    if ORDER == "rev":
        p = blockrev(W.shape[1])
        return ldlq_scaled_fwd(W[:, p].contiguous(), H[p][:, p].contiguous(), cb, K, damp, S[:, p].contiguous())[:, p].contiguous()
    return ldlq_scaled_fwd(W, H, cb, K, damp, S)


def ldlq_scaled_fwd(W, H, cb, K, damp, S):
    W = W.clone(); R, Kd = W.shape
    H = H.clone(); dead = H.diagonal() == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H.diagonal().add_(damp * H.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for a in range(0, Kd, 16):
        blk = W[:, a:a + 16]; sc = S[:, a:a + 16]
        tiles = (blk / sc).reshape(R // 16, 16, 16).reshape(R // 16, 256).numpy()
        q = torch.from_numpy(trellis.vit(np.ascontiguousarray(tiles), cb, K).astype(np.float64)).reshape(R, 16) * sc
        Q[:, a:a + 16] = q
        if a + 16 < Kd:
            E = (blk - q) @ torch.linalg.inv(U[a:a + 16, a:a + 16])
            W[:, a + 16:] -= E @ U[a:a + 16, a + 16:]
    return Q


def post_fit(T, Q, M, chunk=16):
    """per row, the 128-block multipliers s minimising (t - sum_b s_b q_b) M (.)^T jointly (G s = c), rounded to bf16
    together with the block's existing scale B (pass B = ones for plain codes). Returns (Q', s)"""
    R, Kd = T.shape; nb = Kd // 128; out = []; ss = []; Mv = M.reshape(nb, 128, Kd)
    for r0 in range(0, R, chunk):
        Qb = Q[r0:r0 + chunk].reshape(-1, nb, 128)
        P = torch.einsum("rbi,bij->rbj", Qb, Mv)                                  # P_b = q_b M_(b,:)  [r, nb, Kd]
        G = torch.einsum("rbcj,rcj->rbc", P.reshape(-1, nb, nb, 128), Qb)        # G_bc = q_b M_bc q_c^T
        c = torch.einsum("rbj,rj->rb", P, T[r0:r0 + chunk])                      # c_b = q_b M_(b,:) t^T
        G = G + 1e-12 * G.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(nb, dtype=G.dtype)
        s = torch.linalg.solve(G, c[..., None])[..., 0]
        out.append((Qb * s[..., None]).reshape(-1, Kd)); ss.append(s)
    return torch.cat(out), torch.cat(ss)


def bf16(x): return x.to(torch.bfloat16).to(x.dtype)


def mode_rs(p, cb, K, damp0, base, rows):
    K8 = K - 0.125
    Hq, dq = p.metric(damp0); Wt = p.Wt; R, Kd = Wt.shape; nb = Kd // 128
    arms, sec, extra = {}, {}, {}

    def score(name, Q):                                     # Q in Wt's units (already divided by the input scale)
        arms[name] = settle.rels(p, Q, damp0)[0]
    t0 = time.time()
    s_K, _ = settle.search_scale(p.T, cb, K); s_8, sx8 = settle.search_scale(p.T, cb, K8)
    Q = trellis.ldlq_trellis(Wt * s_K, Hq, cb, K, dq) / s_K; score("plain@K", Q); del Q
    Q8 = trellis.ldlq_trellis(Wt * s_8, Hq, cb, K8, dq) / s_8; score("plain@K8", Q8)
    sec["plain"] = round(time.time() - t0, 1); t0 = time.time()
    # post: the plain K-1/8 codes, block scales jointly per row under the LDLQ metric, bf16
    Qp, sp = post_fit(Wt, Q8, Hq); Qp = (Q8.reshape(R, nb, 128) * bf16(sp)[..., None]).reshape(R, Kd)
    score("post@K8", Qp); extra["post_scale_sd"] = float(sp.std()); extra["post_scale_mean"] = float(sp.mean()); del Qp, Q8
    sec["post"] = round(time.time() - t0, 1); t0 = time.time()
    # pre: (row, block) RMS, bf16; g_scale on the normalised rows' tiles; LDLQ with per-(row, block) scales
    B = bf16(Wt.reshape(R, nb, 128).pow(2).mean(-1).sqrt().clamp(min=1e-12))           # [R, nb]
    Bc = B.repeat_interleave(128, dim=1)
    s_pre, sxp = settle.search_scale(settle.tiles_of(Wt / Bc).numpy(), cb, K8)
    Qr = ldlq_scaled(Wt * s_pre, Hq, cb, K8, dq, Bc) / s_pre
    score("pre@K8", Qr)
    extra["pre_rms_sd"] = float((B / B.mean(1, keepdim=True)).std()); extra["s_pre"] = s_pre; extra["s_K8"] = s_8
    sec["pre"] = round(time.time() - t0, 1); t0 = time.time()
    # prepost: pre's codes (Qr = code x B), then joint multipliers on the stored scale B, the product rounded to bf16
    _, sm = post_fit(Wt, Qr, Hq)
    C = Qr.reshape(R, nb, 128) / B[..., None]
    Qpp = (C * bf16(B * sm)[..., None]).reshape(R, Kd)
    score("prepost@K8", Qpp); extra["prepost_mult_sd"] = float(sm.std())
    sec["prepost"] = round(time.time() - t0, 1)
    return {**base, "damp": damp0, "K8": K8, "rel": {k: v[0] for k, v in arms.items()},
            "rel_norefit": {k: v[1] for k, v in arms.items()}, "sec_arm": sec, **extra}


# ---- item 5: #106 / #103 block orders on the order-fixed proxy (w46's reval.py order mode, transcribed) -------------
HEADS = {"attn_output": (256, 6), "ssm_out": (128, 3), "ffn_down": (128, 1)}   # ffn_down: no head constraint (#103)


def orders(tr, tensor):
    """reval.py's orders (stored order of the 128-blocks): natural; head = groups by descending trace, heads inside a
    group by descending trace; free = all blocks by descending trace; rev = head's constraint, ascending. For
    ffn_down (head = one block, group = one head) head == free == #103's blk128."""
    nb = len(tr); hd, gs = HEADS[tensor]; bph = hd // 128
    heads = [list(range(h * bph, (h + 1) * bph)) for h in range(nb // bph)]
    ht = [sum(tr[b] for b in hb) for hb in heads]
    groups = [list(range(g * gs, (g + 1) * gs)) for g in range(len(heads) // gs)]
    def constrained(desc):
        sg = sorted(groups, key=lambda g: -sum(ht[h] for h in g) if desc else sum(ht[h] for h in g))
        return [b for g in sg for h in sorted(g, key=lambda h: -ht[h] if desc else ht[h]) for b in heads[h]]
    return {"natural": list(range(nb)), "head": constrained(True), "free": sorted(range(nb), key=lambda b: -tr[b]),
            "rev": constrained(False)}


def mode_order(p, cb, K, base):
    """reval.py's order_mode with trellis.ldlq_trellis in ORDER (rev: the last stored block is quantized first, so
    the stored arm `rev` = heaviest first in exllamav3's quantization order). m = 1, damping 0.025 (rest group)"""
    damp0 = 0.025
    s_exl, sx = settle.search_scale(p.T, cb, K)
    Hq, dq = p.metric(damp0); nb = p.Kd // 128
    tr = [float(Hq.diagonal()[b * 128:(b + 1) * 128].sum()) for b in range(nb)]
    arms = os.environ.get("ORDER_ARMS", "natural,head,free,rev").split(",")
    if base["w"] == "ffn_down": arms = [a for a in arms if a != "free"]    # == head
    od = orders(tr, base["w"]); rel, rel0, sec = {}, {}, {}
    for arm in arms:
        t0 = time.time()
        cols = torch.cat([torch.arange(b * 128, (b + 1) * 128) for b in od[arm]])
        Q = trellis.ldlq_trellis(p.Wt[:, cols] * s_exl, Hq[cols][:, cols], cb, K, dq)
        Qn = torch.empty_like(Q); Qn[:, cols] = Q
        rel[arm], rel0[arm] = settle.rels(p, Qn / s_exl, damp0)[0]
        sec[arm] = round(time.time() - t0, 1)
    sp = lambda t: float(np.percentile(t, 90) / max(np.percentile(t, 10), 1e-300))
    return {**base, "s_exl": s_exl, "damp": damp0, "rel": rel, "rel_norefit": rel0, "sec_arm": sec,
            "orders": {a: od[a] for a in arms}, "trace": tr, "spread_p90p10": sp(tr)}


def main():
    mode = sys.argv[1]
    if mode not in ("rs", "order"): return settle.main()
    layer, tensor, K, out = int(sys.argv[2]), sys.argv[3], float(Fraction(sys.argv[4])), sys.argv[5]
    rows = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    seed, rseed = int(os.environ.get("E3_SEED", 0)), int(os.environ.get("E3_ROWSEED", 0))
    t00 = time.time()
    p = settle.P2(layer, tensor, rows, seed, rseed)
    cbn = os.environ.get("SETTLE_CB", "mul1x_i8")
    cb = np.ascontiguousarray(codebook(cbn), dtype=np.float32)
    fam_in = tensor in settle.INPUT; damp0 = 0.005 if fam_in else 0.025
    base = {"exp": "rs", "order": ORDER, "proxy": 2, "layer": layer, "w": tensor, "bits": K, "cb": cbn, "seed": seed,
            "rowseed": rseed, "rows": rows, "N": p.N, "K": p.Kd, **p.extra}
    if mode == "order": base["exp"] = "order"; o = mode_order(p, cb, K, base)
    else: o = mode_rs(p, cb, K, damp0, base, rows)
    o["sec_job"] = round(time.time() - t00, 1)
    with open(out + ".part", "w") as fh: fh.write(json.dumps(o) + "\n")
    os.replace(out + ".part", out)
    print(json.dumps(o), flush=True)


if __name__ == "__main__":
    main()
