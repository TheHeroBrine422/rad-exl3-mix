# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tile coordinate descent after exllamav3's LDLQ (#77 cd3; adopted for attn_qg). Merged from rad-exl3-mix
v3/e3/tilecd/tilecd.py (w50, 2026-11-06), unchanged without H_out, plus the two-sided objective for H_out (w76,
2026-11-07): cd() with M = None is the research module's arithmetic op for op (tests/check_tilecd.py).

After ldlq() (or ldlq_2hess()) has produced the codes, CD sweeps revisit every 16-row block a of the regularized
weight (k, n) in order and re-encode its tiles against the target that is optimal given every other tile's current
error. Objective f(E) = tr(E^T H E M), E = W - Q, H = the input metric LDLQ factored (the scaled Hessian with
scaled_hessian, else finalize_capture_H's rotated one; damped), M = the output metric:
- no H_out: M = I. Target T_a = Q_a + H_aa^-1 G_a, G = H E; a tile (16 rows x 16 columns) takes its new code only if
  it lowers f: with D = Q_a - Q_a' per column, delta = 2 <D, G_a> + D^T H_aa D, summed over the tile's 16 columns
  (columns are independent, so the tiles of one block are decided separately and exactly).
- H_out (diagonal per output channel, ldlq_2hess): M = prepare_H_out's matrix, the diagonal in the quantizer's basis
  (output signs / channel scales sv, the 128-blockwise Hadamard, normalized to mean 1, + sigma_reg_out): block
  diagonal with dense 128 x 128 blocks, so the 8 tiles of a 128-column group are coupled. G = H E M; tile (a, c)'s
  target T = Q_ac + H_aa^-1 G_ac M_cc^-1 (the tile's own optimum); its change is accepted if
  delta = 2 <D_c, G_ac> + tr(D_c^T H_aa D_c M_cc) < 0, deciding tile position t = 0..7 of every group in turn and
  updating the rows' G after each step (the cross terms 2 tr(D_t^T H_aa D_s M_st) of tiles accepted earlier in the
  group), so every accepted change lowers f exactly. With M = I every step reduces to the M-free arithmetic bit for bit
  (multiplications by exact ones and additions of exact zeros; tests/check_tilecd.py "gate").
The trellis search is exllamav3's own (quantize_tiles_multigpu: qtf2 / exl3cb / frac as installed), in the
tensor-core layout ldlq() uses; the stored indices of accepted tiles are replaced with the new ones.

rxq.engine.setup(tile_cd=(sweeps, families)) calls install(); quant.quant_args() adds args() per linear. Stats per
linear: OUTDIR/tilecd.jsonl (rel = f(E) / f(W) in the quantization domain after LDLQ and after each sweep, accepted
tile shares, seconds; "hout": true for the two-sided objective)."""
import json, os, time
import torch

# key suffix -> family: qwen35.FAMILY's names, which config.quant_settings tests (w115: gate and up were one family
# "ffn_gate_up" here, a name quant_settings never matches, so CD could not be switched on for them; the name stays an
# alias for both)
FAMILY = (("self_attn.q_proj", "attn_qg"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
          ("self_attn.o_proj", "attn_output"), ("linear_attn.in_proj_qkv", "ssm_qkv"), ("linear_attn.in_proj_z", "ssm_z"),
          ("linear_attn.out_proj", "ssm_out"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
          ("mlp.down_proj", "ffn_down"), ("lm_head", "head"))
_CFG = {"sweeps": 0, "families": set()}
_CAP = {}                                  # the current linear's metric inputs, captured from scaled_ldl / finalize / prepare_H_out
GROUP = 128                                # exllamav3's output Hadamard block (had_n): M is block diagonal in it


def family(key):
    return next((f for s, f in FAMILY if key.endswith(s)), None)


def args(key, K, outdir):
    """quant_args entries for one linear: {} unless its family is selected"""
    if not _CFG["sweeps"] or family(key) not in _CFG["families"]: return {}
    return {"tile_cd": _CFG["sweeps"], "tile_cd_key": key, "tile_cd_K": K, "tile_cd_log": os.path.join(outdir, "tilecd.jsonl")}


def install(Q, sweeps, families):
    """wrap exllamav3's quantize module Q: finalize_capture_H, scaled_ldl and prepare_H_out record the metrics, ldlq and
    ldlq_2hess run CD after themselves when quant_args["tile_cd"] is set"""
    fams = {f.strip() for f in families.split(",") if f.strip()}
    if "ffn_gate_up" in fams: fams = (fams - {"ffn_gate_up"}) | {"ffn_gate", "ffn_up"}
    bad = fams - {f for _, f in FAMILY}
    if bad: raise SystemExit(f"tile CD families: unknown {sorted(bad)} (known: {sorted({f for _, f in FAMILY})})")
    _CFG.update(sweeps=int(sweeps), families=fams)
    fin, sldl, ldlq, ldlq2, pho = Q.finalize_capture_H, Q.scaled_ldl, Q.ldlq, Q.ldlq_2hess, Q.prepare_H_out

    def finalize_capture_H(H_data, quant_args, verbose):
        r = fin(H_data, quant_args, verbose)
        _CAP.clear()
        if quant_args.get("tile_cd"): _CAP["H_rot"] = r[1]
        return r

    def scaled_ldl(H_orig, su, quant_args):
        if quant_args.get("tile_cd"): _CAP.update(H_orig=H_orig, su=su)
        return sldl(H_orig, su, quant_args)

    def prepare_H_out(H_out, sv, quant_args, verbose, device):
        L_out, Ho = pho(H_out, sv, quant_args, verbose, device)
        if quant_args.get("tile_cd"): _CAP["H_out"] = Ho
        return L_out, Ho

    def ldlq_cd(weight, L, quant_args, pb=None):
        weight_q, encoded = ldlq(weight, L, quant_args, pb)
        if not quant_args.get("tile_cd"): return weight_q, encoded
        H = metric(Q, quant_args, L.device)
        weight_q, encoded = cd(Q, weight, weight_q, encoded, H, quant_args)
        _CAP.clear(); del H
        return weight_q, encoded

    def ldlq_2hess_cd(weight, L_in, L_out, quant_args, pb=None):
        weight_q, encoded = ldlq2(weight, L_in, L_out, quant_args, pb)
        if not quant_args.get("tile_cd"): return weight_q, encoded
        H = metric(Q, quant_args, L_in.device)
        weight_q, encoded = cd(Q, weight, weight_q, encoded, H, quant_args, M=_CAP["H_out"].to(L_in.device))
        _CAP.clear(); del H
        return weight_q, encoded

    Q.finalize_capture_H, Q.scaled_ldl, Q.ldlq, Q.ldlq_2hess, Q.prepare_H_out = \
        finalize_capture_H, scaled_ldl, ldlq_cd, ldlq_2hess_cd, prepare_H_out


def metric(Q, quant_args, dev):
    """the Hessian LDLQ factored, dense, on dev: P diag(su) H diag(su) P (scaled_ldl's) or the rotated H"""
    if quant_args.get("scaled_hessian"):
        s = _CAP["su"].view(-1).float().to(dev)
        H = _CAP["H_orig"].to(dev) * s.unsqueeze(0)
        H *= s.unsqueeze(1)
        Q.blockwise_preapply_had_r_(H, Q.had_k)
        Q.blockwise_preapply_had_l_(H, Q.had_k)
        return H
    return _CAP["H_rot"].to(dev).float()


def _trace(H, X, chunk=4096):
    """tr(X^T H X), X (k, n), in column chunks"""
    return sum(float((X[:, c:c + chunk] * (H @ X[:, c:c + chunk])).sum()) for c in range(0, X.shape[1], chunk))


def out_blocks(M, dt):
    """M (n, n) block diagonal in GROUP -> its blocks (n / GROUP, GROUP, GROUP); refuses mass outside the blocks (a
    dense H_out: tile CD supports the diagonal estimator only). Blocks already (fork patch 0008's prepare_H_out_blocks,
    inputs.hout.ldlq = blocks) pass through"""
    if M.dim() == 3:
        assert M.shape[1] == M.shape[2] == GROUP, tuple(M.shape)
        return M.to(dt).contiguous()
    n = M.shape[0]; ng = n // GROUP
    assert n % GROUP == 0, n
    Mb = torch.diagonal(M.view(ng, GROUP, ng, GROUP), dim1=0, dim2=2).permute(2, 0, 1).to(dt).contiguous()
    tot, inb = float(torch.linalg.vector_norm(M)) ** 2, float(torch.linalg.vector_norm(Mb)) ** 2   # no n x n temporary
    if tot - inb > 1e-10 * max(inb, 1e-30):
        raise SystemExit("tile CD with H_out: the output metric is not block diagonal in 128 (a dense H_out?): not supported")
    return Mb


def _gmul(X, Mb):
    """X (r, n) @ blockdiag(Mb): per 128-column group"""
    r, n = X.shape; ng = Mb.shape[0]
    return torch.bmm(X.view(r, ng, GROUP).transpose(0, 1), Mb).transpose(0, 1).reshape(r, n)


def _trace2(H, X, Mb, chunk=4096):
    """tr(X^T H X M), X (k, n), in column chunks (multiples of GROUP)"""
    tot = 0.0
    for c in range(0, X.shape[1], chunk):
        Xc = X[:, c:c + chunk]; Mc = Mb[c // GROUP:(c + Xc.shape[1]) // GROUP]
        tot += float((_gmul(Xc, Mc) * (H @ Xc)).sum())
    return tot


def cd(Q, weight, weight_q, encoded, H, quant_args, M=None):
    if M is not None: return cd_hout(Q, weight, weight_q, encoded, H, quant_args, M)
    sweeps = int(quant_args["tile_cd"]); dev = H.device
    dt = getattr(torch, os.environ.get("TILECD_DTYPE", "float32"))   # TEST ONLY (not in the code stamp)
    t0 = time.time()
    H = H.to(dt); W = weight.to(dev, dt); Wq = weight_q.to(dev, dt, copy=True); enc = encoded.to(dev).clone()
    k, n = W.shape; tn = n // 16; m = k // 16
    perm, permi = Q.tensor_core_perm(dev), Q.tensor_core_perm_i(dev)
    Hb = torch.diagonal(H.view(m, 16, m, 16), dim1=0, dim2=2).permute(2, 0, 1).contiguous()   # (m, 16, 16) diagonal blocks
    Hbi = torch.linalg.inv(Hb)
    E = W - Wq
    den = max(_trace(H, W), 1e-30)
    rel = {"ldlq": _trace(H, E) / den}; acc, sec = {}, {"setup": round(time.time() - t0, 2)}
    for sw in range(1, sweeps + 1):
        t1 = time.time(); na = 0
        G = H @ E
        for b in range(m):
            a = 16 * b
            Qa, Ga = Wq[a:a + 16], G[a:a + 16]
            T = Qa + Hbi[b] @ Ga
            tiles = T.reshape(16, tn, 16).permute(1, 0, 2).reshape(tn, 256)[:, perm].float().contiguous()
            qw, qi = Q.quantize_tiles_multigpu(tiles, quant_args)
            qw = qw[:, permi].reshape(tn, 16, 16).permute(1, 0, 2).reshape(16, n).to(dt)
            D = Qa - qw                                               # E_new - E_old on these rows
            dcol = 2 * (D * Ga).sum(0) + (D * (Hb[b] @ D)).sum(0)     # per column
            keep = dcol.view(tn, 16).sum(1) < 0                       # per tile
            nk = int(keep.sum())
            if nk:
                kc = keep.repeat_interleave(16)
                D *= kc
                Wq[a:a + 16] = torch.where(kc, qw, Qa)
                enc[b][keep] = qi.to(enc.dtype)[keep]
                E[a:a + 16] += D
                G.addmm_(H[:, a:a + 16], D)
                na += nk
        torch.cuda.synchronize(dev)
        rel[f"cd{sw}"] = _trace(H, E) / den; acc[f"cd{sw}"] = na / (m * tn); sec[f"cd{sw}"] = round(time.time() - t1, 2)
    del G, E, W
    rec = {"key": quant_args.get("tile_cd_key"), "K": quant_args.get("tile_cd_K"), "sweeps": sweeps, "rel": rel,
           "ratio": rel[f"cd{sweeps}"] / rel["ldlq"] if sweeps else 1.0, "accepted": acc, "sec": sec,
           "scaled_hessian": bool(quant_args.get("scaled_hessian")), "shape": [k, n], "dtype": str(dt)}
    lp = quant_args.get("tile_cd_log")
    if lp:
        with open(lp, "a") as fh: fh.write(json.dumps(rec) + "\n")
    print(f"     tile CD {sweeps}: rel {rel['ldlq']:.4e} -> {rel[f'cd{sweeps}']:.4e} (x{rec['ratio']:.4f}), accepted "
          + " / ".join(f"{v:.3f}" for v in acc.values()) + f", {sum(v for kk, v in sec.items()):.1f}s", flush=True)
    return Wq.to(weight_q.device, weight_q.dtype), enc.to(encoded.device)


def cd_hout(Q, weight, weight_q, encoded, H, quant_args, M):
    """cd() for f(E) = tr(E^T H E M), M block diagonal in GROUP (module docstring). Same layout, search and bookkeeping
    as cd(); with M = I its arithmetic is cd()'s bit for bit"""
    sweeps = int(quant_args["tile_cd"]); dev = H.device
    dt = getattr(torch, os.environ.get("TILECD_DTYPE", "float32"))   # TEST ONLY (not in the code stamp)
    t0 = time.time()
    H = H.to(dt); W = weight.to(dev, dt); Wq = weight_q.to(dev, dt, copy=True); enc = encoded.to(dev).clone()
    k, n = W.shape; tn = n // 16; m = k // 16; ng = n // GROUP; tg = GROUP // 16
    Mb = out_blocks(M.to(dev), dt)                                                              # (ng, 128, 128)
    Mt = torch.diagonal(Mb.view(ng, tg, 16, tg, 16), dim1=1, dim2=3).permute(0, 3, 1, 2).reshape(tn, 16, 16).contiguous()
    Mti = torch.linalg.inv(Mt)                                                                  # (tn, 16, 16) tiles' M_cc^-1
    perm, permi = Q.tensor_core_perm(dev), Q.tensor_core_perm_i(dev)
    Hb = torch.diagonal(H.view(m, 16, m, 16), dim1=0, dim2=2).permute(2, 0, 1).contiguous()   # (m, 16, 16) diagonal blocks
    Hbi = torch.linalg.inv(Hb)
    tile = lambda X: X.view(16, tn, 16).transpose(0, 1)                                         # (16, n) -> (tn, 16, 16) view
    untile = lambda X: X.transpose(0, 1).reshape(16, n)
    pos = torch.arange(tn, device=dev) % tg                                                     # a tile's position in its group
    E = W - Wq
    den = max(_trace2(H, W, Mb), 1e-30)
    rel = {"ldlq": _trace2(H, E, Mb) / den}; acc, sec = {}, {"setup": round(time.time() - t0, 2)}
    for sw in range(1, sweeps + 1):
        t1 = time.time(); na = 0
        G = _gmul(H @ E, Mb)
        for b in range(m):
            a = 16 * b
            Qa, Ga = Wq[a:a + 16], G[a:a + 16]
            T = Qa + Hbi[b] @ untile(torch.bmm(tile(Ga), Mti))                 # Q + H_aa^-1 G_ac M_cc^-1 per tile
            tiles = T.reshape(16, tn, 16).permute(1, 0, 2).reshape(tn, 256)[:, perm].float().contiguous()
            qw, qi = Q.quantize_tiles_multigpu(tiles, quant_args)
            qw = qw[:, permi].reshape(tn, 16, 16).permute(1, 0, 2).reshape(16, n).to(dt)
            D = Qa - qw                                                      # E_new - E_old on these rows
            HD = Hb[b] @ D
            X = untile(torch.bmm(tile(HD), Mt))                              # H_aa D_c M_cc per tile
            Gw = Ga.clone(); keep = torch.zeros(tn, dtype=torch.bool, device=dev)
            for t in range(tg):                                              # tile position t of every 128-column group
                dcol = 2 * (D * Gw).sum(0) + (D * X).sum(0)                  # per column
                kt = (dcol.view(tn, 16).sum(1) < 0) & (pos == t)             # per tile
                if not bool(kt.any()): continue
                keep |= kt
                Dt = D * kt.repeat_interleave(16)
                Gw += _gmul(Hb[b] @ Dt, Mb)                                  # the rows' G after this step (cross terms)
            nk = int(keep.sum())
            if nk:
                kc = keep.repeat_interleave(16)
                D *= kc
                Wq[a:a + 16] = torch.where(kc, qw, Qa)
                enc[b][keep] = qi.to(enc.dtype)[keep]
                E[a:a + 16] += D
                G.addmm_(H[:, a:a + 16], _gmul(D, Mb))
                na += nk
        torch.cuda.synchronize(dev)
        rel[f"cd{sw}"] = _trace2(H, E, Mb) / den; acc[f"cd{sw}"] = na / (m * tn); sec[f"cd{sw}"] = round(time.time() - t1, 2)
    del G, E, W
    rec = {"key": quant_args.get("tile_cd_key"), "K": quant_args.get("tile_cd_K"), "sweeps": sweeps, "rel": rel,
           "ratio": rel[f"cd{sweeps}"] / rel["ldlq"] if sweeps else 1.0, "accepted": acc, "sec": sec,
           "scaled_hessian": bool(quant_args.get("scaled_hessian")), "shape": [k, n], "dtype": str(dt), "hout": True}
    lp = quant_args.get("tile_cd_log")
    if lp:
        with open(lp, "a") as fh: fh.write(json.dumps(rec) + "\n")
    print(f"     tile CD {sweeps} (H_out): rel {rel['ldlq']:.4e} -> {rel[f'cd{sweeps}']:.4e} (x{rec['ratio']:.4f}), accepted "
          + " / ".join(f"{v:.3f}" for v in acc.values()) + f", {sum(v for kk, v in sec.items()):.1f}s", flush=True)
    return Wq.to(weight_q.device, weight_q.dtype), enc.to(encoded.device)
