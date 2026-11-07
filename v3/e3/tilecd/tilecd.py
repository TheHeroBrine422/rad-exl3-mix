"""tilecd.py -- tile coordinate descent after exllamav3's LDLQ (#77 cd3 on the GPU; w50, README.md here).

After ldlq() has produced the codes, CD sweeps revisit every 16-row block a of the regularized weight (k, n) in order
and re-encode its tiles against the target that is optimal given every other block's current error:
    T_a = Q_a + H_aa^-1 G_a,   G = H E,  E = W - Q   (H = the metric LDLQ factored: the scaled Hessian with
                                                       --scaled-hessian, else finalize_capture_H's rotated one; damped)
and a tile (16 rows x 16 columns) takes its new code only if it lowers tr(E^T H E): with D = Q_a - Q_a' per column,
delta = 2 <D, G_a> + D^T H_aa D, summed over the tile's 16 columns (columns are independent, so the tiles of one
block are decided separately and exactly). G is updated after each block and recomputed at the start of each sweep.
The trellis search is exllamav3's own (quantize_tiles_multigpu: qtf2 / exl3cb / frac as installed), in the
tensor-core layout ldlq() uses; the stored indices of accepted tiles are replaced with the new ones. The math is
trellis2.ldlq_cd's (settle.py's CPU reference), transposed to exllamav3's (k, n) layout.

ourq.py --tile-cd N [--tile-cd-families attn_qg,...] calls install() once and args() per linear (lines tagged
"# +cd"): N sweeps for the linears of the named families, nothing for the rest. Stats per linear: OUTDIR/tilecd.jsonl
(rel = tr(E^T H E) / tr(W^T H W) in the quantization domain after LDLQ and after each sweep, accepted tile shares,
seconds). Not implemented with an output-side Hessian (--hout: ldlq_2hess): that combination is refused."""
import json, os, time
import torch

# ourq.py key suffix -> family (the challenge's tensor names)
FAMILY = (("self_attn.q_proj", "attn_qg"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
          ("self_attn.o_proj", "attn_output"), ("linear_attn.in_proj_qkv", "ssm_qkv"), ("linear_attn.in_proj_z", "ssm_z"),
          ("linear_attn.out_proj", "ssm_out"), ("mlp.gate_proj", "ffn_gate_up"), ("mlp.up_proj", "ffn_gate_up"),
          ("mlp.down_proj", "ffn_down"), ("lm_head", "head"))
_CFG = {"sweeps": 0, "families": set()}
_CAP = {}                                  # the current linear's metric inputs, captured from scaled_ldl / finalize


def family(key):
    return next((f for s, f in FAMILY if key.endswith(s)), None)


def args(key, K, outdir):
    """quant_args entries for one linear: {} unless its family is selected"""
    if not _CFG["sweeps"] or family(key) not in _CFG["families"]: return {}
    return {"tile_cd": _CFG["sweeps"], "tile_cd_key": key, "tile_cd_K": K, "tile_cd_log": os.path.join(outdir, "tilecd.jsonl")}


def install(Q, sweeps, families):
    """wrap exllamav3's quantize module Q: finalize_capture_H and scaled_ldl record the metric, ldlq runs CD after
    itself when quant_args["tile_cd"] is set, ldlq_2hess refuses it"""
    fams = {f.strip() for f in families.split(",") if f.strip()}
    bad = fams - {f for _, f in FAMILY}
    if bad: raise SystemExit(f"--tile-cd-families: unknown {sorted(bad)} (known: {sorted({f for _, f in FAMILY})})")
    _CFG.update(sweeps=int(sweeps), families=fams)
    fin, sldl, ldlq, ldlq2 = Q.finalize_capture_H, Q.scaled_ldl, Q.ldlq, Q.ldlq_2hess

    def finalize_capture_H(H_data, quant_args, verbose):
        r = fin(H_data, quant_args, verbose)
        _CAP.clear()
        if quant_args.get("tile_cd"): _CAP["H_rot"] = r[1]
        return r

    def scaled_ldl(H_orig, su, quant_args):
        if quant_args.get("tile_cd"): _CAP.update(H_orig=H_orig, su=su)
        return sldl(H_orig, su, quant_args)

    def ldlq_cd(weight, L, quant_args, pb=None):
        weight_q, encoded = ldlq(weight, L, quant_args, pb)
        if not quant_args.get("tile_cd"): return weight_q, encoded
        H = metric(Q, quant_args, L.device)
        weight_q, encoded = cd(Q, weight, weight_q, encoded, H, quant_args)
        _CAP.clear(); del H
        return weight_q, encoded

    def ldlq_2hess(weight, L_in, L_out, quant_args, pb=None):
        if quant_args.get("tile_cd"):
            raise SystemExit(f"{quant_args.get('tile_cd_key')}: tile CD with an output-side Hessian (--hout) is not implemented")
        return ldlq2(weight, L_in, L_out, quant_args, pb)

    Q.finalize_capture_H, Q.scaled_ldl, Q.ldlq, Q.ldlq_2hess = finalize_capture_H, scaled_ldl, ldlq_cd, ldlq_2hess


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


def cd(Q, weight, weight_q, encoded, H, quant_args):
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
