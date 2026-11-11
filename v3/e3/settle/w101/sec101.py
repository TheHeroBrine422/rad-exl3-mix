#!/usr/bin/env python3
"""sec101.py -- w101 #131 (../w101.md): per-section K inside fused q_proj (q | gate) / in_proj_qkv (q | k | v), w87's
two-sided block proxy on chosen 128-row blocks.

  sec101.py prep LAYER TENSOR OUTDIR        the P2 transforms of the whole tensor (seed 0, rseed 0: P2's signs, su and
                                            g_scale tile sample) for this experiment's sampled blocks + ref_b of every
                                            block; OUTDIR/prep/LAYER-TENSOR.pt (restartable)
  sec101.py job LAYER TENSOR K OUTDIR       g_scale search at K (i8), then two-sided block LDLQ of every sampled block;
                                            OUTDIR/rows/LAYER-TENSOR-K.json (skipped if present)
  sec101.py gate LAYER TENSOR OUTDIR        G1: prep on P2's own block (rseed 0) == settle.P2 bit for bit (Wt, T, su,
                                            r, sv); G2: the batched LDLQ == settle87.ldlq2 on one block (max |dQ| 0)
TENSOR: attn_qg | ssm_qkv. Proxy = settle87 (ldlq2's order, hout_metric's rotated H_out with sigma_out 0.01 over all N
rows, block_ldl, per-output-channel refit) with the published H (E3_HESS) for quantizer and scoring, sigma 0.005
(input group, K >= 1), mul1x_i8, ourq's drift in the scale search (settle77). Env: E3_CKPT, E3_HESS, W87_HOUT."""
import json, os, sys, time
import numpy as np
import torch

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "..", "w87"), os.path.join(_h, "..", "w77"), os.path.join(_h, "..", "w64"),
                os.path.join(_h, ".."), os.path.join(_h, "..", "..")]
import settle87                                            # imports settle77 (codebooks, drift) -> settle64 -> settle
import settle, settle77, trellis, trellis2, e3
from safetensors import safe_open
torch.set_grad_enabled(False)

SIG, SIG_OUT, CB = 0.005, 0.01, "mul1x_i8"
NAME = {"attn_qg": "self_attn.q_proj", "ssm_qkv": "linear_attn.in_proj_qkv"}


def sections(tensor, N):
    """{section: [natural 128-row block indices]}; asserts the boundaries are on 128-row blocks"""
    nb = N // 128
    if tensor == "ssm_qkv":
        assert N == 10240 and 2048 % 128 == 0       # [q 2048 | k 2048 | v 6144] (modeling_qwen3_5: split key, key, value)
        return {"q": list(range(0, 16)), "k": list(range(16, 32)), "v": list(range(32, 80))}
    assert tensor == "attn_qg" and N == 12288 and 256 % 128 == 0   # per head [q 256 | gate 256] (view(.., 2 hd), chunk 2)
    return {"q": [b for b in range(nb) if b % 4 < 2], "gate": [b for b in range(nb) if b % 4 >= 2]}


def sample(layer, tensor, N):
    """8 distinct blocks per section (rng 131 + layer): half A = the first 4 drawn, half B = the last 4"""
    rng = np.random.default_rng(131 + layer); out = {}
    for s, bl in sections(tensor, N).items():
        d = rng.choice(len(bl), 8, replace=False); out[s] = {"A": [bl[i] for i in d[:4]], "B": [bl[i] for i in d[4:]]}
    return out


def prep(layer, tensor, outdir, blocks=None):
    """settle.P2.__init__ (seed 0, rseed 0) on given blocks: same signs, su, tile sample T; + ref_b of every block"""
    f = f"{outdir}/prep/{layer}-{tensor}.pt"
    if blocks is None and os.path.exists(f): return torch.load(f, weights_only=False)
    seed = rseed = 0
    names, hn = e3.ROUTE[tensor]
    W0 = e3.load_w(layer, names); N, Kd = W0.shape
    g = torch.Generator().manual_seed(1000 + layer + 7919 * seed)
    sv = (torch.randint(0, 2, (N,), generator=g) * 2 - 1).double()
    su_sign = (torch.randint(0, 2, (Kd,), generator=g) * 2 - 1).double()
    r = W0.pow(2).mean(1).sqrt().clamp(min=1e-12)
    X = W0 / r[:, None]
    su = X.pow(2).mean(0).sqrt(); su[su < 1e-30] = 0.1; su = su * su_sign
    Xf = settle.had_cols(settle.had_rows((X * sv[:, None]).float()) / su.float())
    tn, tk = N // 16, Kd // 16
    w4 = Xf.reshape(tn, 16, tk, 16)
    dl = max(tk, tn); ii = torch.arange(dl).repeat_interleave(3); ww = torch.arange(3).repeat(dl)
    kk = ii % tk; nn = (ii + ww) % tn
    ms = w4.square().mean(dim=(1, 3)).flatten()
    nx = min(max(8, dl * 3 // 16), (ms.numel() + 1) // 2)
    ex = torch.cat((torch.topk(ms, nx).indices, torch.topk(ms, nx, largest=False).indices))
    T = w4[torch.cat((nn, ex // tk)), :, torch.cat((kk, ex % tk)), :].reshape(-1, 256).numpy()
    rng = np.random.default_rng(17 + layer + 1000 * seed + 100 * rseed)
    if len(T) > 2304: T = T[np.sort(rng.choice(len(T), 2304, replace=False))]
    del Xf, w4
    smp = None
    if blocks is None:
        smp = sample(layer, tensor, N); blocks = sorted(b for s in smp.values() for h in s.values() for b in h)
    idx = (torch.tensor(blocks)[:, None] * 128 + torch.arange(128)).reshape(-1)
    key = f"{e3.P}{layer}.{NAME[tensor]}"
    with safe_open(os.path.join(os.environ["W87_HOUT"], key + ".safetensors"), "pt") as fh: h = fh.get_tensor("hout_diag").double()
    assert h.numel() == N and bool(torch.isfinite(h).all()) and float(h.min()) > 0
    o = {"layer": layer, "tensor": tensor, "key": key, "N": N, "Kd": Kd, "blocks": blocks, "sample": smp, "T": T,
         "su": su, "Wo": W0[idx].clone(), "r": r[idx][:, None].clone(), "sv": sv[idx][:, None].clone(),
         "Wt": settle.had_cols(settle.had_rows(X[idx] * sv[idx][:, None]) / su), "h": h, "hr2_mean": float((h * r ** 2).mean()),
         "r_all": r.clone()}
    if smp is not None:                                     # ref_b = sum_i h_i w_i H w_i^T for every natural block
        H = e3.load_h(layer, hn); rr = torch.zeros(N, dtype=torch.float64)
        for c in range(0, N, 1024): rr[c:c + 1024] = ((W0[c:c + 1024] @ H) * W0[c:c + 1024]).sum(1)
        o["ref_all"] = (h * rr).reshape(-1, 128).sum(1)
        os.makedirs(f"{outdir}/prep", exist_ok=True)
        torch.save(o, f + ".part"); os.replace(f + ".part", f)
    return o


def lin_of(o, H):
    """settle.P2.metric + settle87.in_metric: block-LDL of P diag(su) (H + sigma mean diag I) diag(su) P"""
    Hd = H.clone(); dead = Hd.diagonal() == 0; assert int(dead.sum()) == 0
    Hd.diagonal().add_(SIG * H.diagonal().mean())
    Hd *= o["su"][None, :]; Hd *= o["su"][:, None]
    Hd = settle.had_cols(Hd); Hd = settle.had_cols(Hd.T.contiguous()).T.contiguous()
    return settle87.block_ldl(Hd)


def lout_of(o, j):
    """settle87.hout_metric for stored block j (row slice j*128 of o's sampled rows)"""
    b = o["blocks"][j]; sl = slice(128 * b, 128 * b + 128)
    r = o["r_all"][sl]; d = o["h"][sl] * r ** 2
    hd = settle.had128(); Ho = (hd * d[None, :]) @ hd / o["hr2_mean"]
    Ho.diagonal().add_(SIG_OUT)
    return settle87.block_ldl(Ho)


def ldlq2_batch(X, Lin, Lout, cb, K):
    """settle87.ldlq2 on nb independent 128-row blocks at once: X [nb, 128, Kd], Lout [nb, 128, 128]"""
    nb, R, Kd = X.shape; F = torch.zeros_like(X); Q = torch.zeros_like(X)
    for a in range(Kd - 16, -1, -16):
        for c in range(R - 16, -1, -16):
            t = (X[:, c:c + 16, a:a + 16] + F[:, c:c + 16, a:a + 16]).reshape(nb, 256).numpy()
            q = torch.from_numpy(np.asarray(trellis.vit(t, cb, K), dtype=np.float64)).reshape(nb, 16, 16)
            Q[:, c:c + 16, a:a + 16] = q
            if c > 0: F[:, :c, a:a + 16] += Lout[:, c:c + 16, :c].transpose(1, 2) @ (X[:, c:c + 16, a:a + 16] - q)
        if a > 0:
            G = Lout.transpose(1, 2) @ (X[:, :, a:a + 16] - Q[:, :, a:a + 16])
            F[:, :, :a] += G @ Lin[a:a + 16, :a]
    return Q


def back(o, Q):
    """settle.P2.back for the stored rows: had_rows(had_cols(Q) * su) * sv * r"""
    return settle.had_rows(settle.had_cols(Q) * o["su"]) * o["sv"] * o["r"]


def errors(o, Qt, H):
    """per sampled block: two-sided error after the per-output-channel refit under H + delta I (settle87.scores)"""
    q = back(o, Qt); dl = SIG * H.diagonal().mean(); hq = q @ H; W = o["Wo"]
    c = ((hq * W).sum(1) + dl * (q * W).sum(1)) / ((hq * q).sum(1) + dl * (q * q).sum(1)).clamp(min=1e-300)
    D = W - q * c[:, None]; e = ((D @ H) * D).sum(1)
    idx = (torch.tensor(o["blocks"])[:, None] * 128 + torch.arange(128)).reshape(-1)
    e2 = (o["h"][idx] * e).reshape(-1, 128).sum(1); e1 = e.reshape(-1, 128).sum(1)
    D0 = W - q; e0 = (o["h"][idx] * ((D0 @ H) * D0).sum(1)).reshape(-1, 128).sum(1)
    return e2, e1, e0


def job(layer, tensor, K, outdir):
    f = f"{outdir}/rows/{layer}-{tensor}-K{K:g}.json"
    if os.path.exists(f): return
    t0 = time.time(); o = prep(layer, tensor, outdir)
    cb = np.ascontiguousarray(trellis2.codebook(CB), dtype=np.float32)
    s, sx = settle.search_scale(o["T"], cb, K); t_scale = time.time() - t0
    H = e3.load_h(layer, e3.ROUTE[tensor][1]); Lin = lin_of(o, H)
    nb = len(o["blocks"])
    Lout = torch.stack([lout_of(o, j) for j in range(nb)])
    t1 = time.time()
    Q = ldlq2_batch(o["Wt"].reshape(nb, 128, -1) * s, Lin, Lout, cb, K) / s; del Lin
    e2, e1, e0 = errors(o, Q.reshape(nb * 128, -1), H)
    rec = {"exp": "w101-131", "layer": layer, "tensor": tensor, "K": K, "cb": CB, "s_exl": s, "search": sx,
           "drift": settle77.drift(K), "blocks": o["blocks"], "e2": e2.tolist(), "e1": e1.tolist(), "e2_norefit": e0.tolist(),
           "sec_scale": round(t_scale, 1), "sec_ldlq": round(time.time() - t1, 1), "sec": round(time.time() - t0, 1)}
    os.makedirs(f"{outdir}/rows", exist_ok=True)
    with open(f + ".part", "w") as fh: json.dump(rec, fh)
    os.replace(f + ".part", f)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("e2", "e1", "e2_norefit", "blocks", "search")}), flush=True)


def gate(layer, tensor, outdir):
    """G1 prep vs settle.P2 (rows 128, seed 0, rseed 0); G2 batched LDLQ vs settle87.ldlq2 on that block at K2.5"""
    os.environ.setdefault("SETTLE_CACHE", f"{outdir}/gatecache")
    p = settle.P2(layer, tensor, 128, 0, 0); b = p.extra["blocks"]
    o = prep(layer, tensor, outdir, blocks=b)
    g1 = {k: float((torch.as_tensor(getattr(p, a)).double() - torch.as_tensor(o[k]).double()).abs().max())
          for k, a in (("Wt", "Wt"), ("T", "T"), ("su", "su"), ("r", "r"), ("sv", "sv"), ("Wo", "Wo"))}
    K = 2.5
    cb = np.ascontiguousarray(trellis2.codebook(CB), dtype=np.float32)
    Lin = lin_of(o, p.H); Lo = lout_of(o, 0)
    Lref, _, _ = settle87.hout_metric(layer, tensor, p, SIG_OUT)
    Hq, _ = p.metric(SIG); Lin_ref = settle87.block_ldl(Hq)
    s = 1.0 / 0.93                                           # any fixed scale: the same for both paths
    t0 = time.time(); Q1 = settle87.ldlq2(p.Wt * s, Lin_ref, Lref, settle87.vit_fun(cb, K)); t1 = time.time()
    Q2 = ldlq2_batch((o["Wt"] * s)[None], Lin, Lo[None], cb, K)[0]; t2 = time.time()
    out = {"gate": f"{layer}-{tensor}", "block": b, "G1_maxabs": g1, "Lout_maxabs": float((Lo - Lref).abs().max()),
           "Lin_maxabs": float((Lin - Lin_ref).abs().max()), "G2_max_dQ": float((Q1 - Q2).abs().max()),
           "tiles_differ": int(((Q1 - Q2).abs().reshape(8, 16, -1, 16).amax(dim=(1, 3)) > 0).sum()),
           "sec_ref": round(t1 - t0, 1), "sec_batch": round(t2 - t1, 1)}
    e_ref = float(errors(o, Q1 / s, p.H)[0].sum()); e_bat = float(errors(o, Q2 / s, p.H)[0].sum())
    out.update(e2_ref=e_ref, e2_batch=e_bat, e2_ratio=e_bat / e_ref)
    # G2: identical codes, or (fp reassociation in the batched matmuls) <= 0.5% of the tiles flipped and e2 within 1e-3
    out["PASS"] = all(v == 0 for v in g1.values()) and (out["G2_max_dQ"] == 0.0 or (
        out["tiles_differ"] <= 0.005 * Q1.numel() / 256 and abs(out["e2_ratio"] - 1) <= 1e-3))
    print(json.dumps(out), flush=True)
    os.makedirs(f"{outdir}/gates", exist_ok=True)
    json.dump(out, open(f"{outdir}/gates/{layer}-{tensor}.json", "w"), indent=1)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "prep": prep(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif m == "job": job(int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5])
    elif m == "gate": gate(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else: raise SystemExit(__doc__)
