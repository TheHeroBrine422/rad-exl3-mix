#!/usr/bin/env python3
"""a1test.py -- #81, faraway idea A1 on the CPU trellis proxy: code down_proj as the residual after predicting each
neuron's down column from its decoded up and gate rows (docs/ideas/2026-11-02-faraway/faraway.md, FA-A1).

  a1test.py uq LAYER K R0 R1 OUTDIR     stage 1: quantize gate_up rows [R0, R1) (gate = rows 0..I-1, up = I..2I-1) as
                                        trellis2 does (128-block Hadamard x signs, LDLQ, calib H_in, codebook
                                        cubic0.02_i8 = the adopted i8c, scale factor E3_FACTORS), decode to the original
                                        basis -> OUTDIR/L<layer>-K<K>/gu-<R0>.npy (float32)
  a1test.py down LAYER K ARM OUTDIR OUT.jsonl
                                        stage 2: down_proj, 256 sampled rows (trellis2's row sample), arm
                                          D   down itself (control)
                                          ug  R = D - U_hat^T diag(a) - G_hat^T diag(b)   (a, b per neuron, least squares
                                              over all 5120 outputs, from the decoded predictors at the same K)
                                          u   R = D - U_hat^T diag(a)   (up only: the cheaper kernel variant, A2)
                                          ugtrue  as ug with the unquantized up / gate (oracle; checks the faraway numbers)
                                          ugperm / uperm  as ug / u with the predictor rows of a random other neuron
                                              (control, #81-02: a real prediction must beat it; an artifact of coding a
                                              residual at all would not)
                                        quantized like trellis2 (LDLQ, i8c, H = down's calib input Hessian). Reported
                                        rel_D = tr(dR H dR^T) / tr(D H D^T) on the sampled rows (error relative to D's
                                        energy for every arm, so arms compare directly), plus the predictable share.
The decoder rebuilds D_hat = R_hat + U_hat^T a + G_hat^T b exactly, so the coded error is R's only (closed loop).
Side information a, b: 2 x fp16 per neuron = 0.006 bpw on down; not counted.
"""
import glob, json, math, os, sys, time
import numpy as np
import torch

for _d in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")):
    sys.path.insert(0, _d)
import e3, trellis, trellis2
torch.set_grad_enabled(False)
CB = os.environ.get("A1_CB", "cubic0.02_i8")


def had_inv(n, seed):
    """inverse of e3.had_blocks(n, seed) (orthogonal: h symmetric, signs +-1)"""
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < 128: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h /= math.sqrt(128)
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).double()
    def f(y):
        sh = y.shape
        return (y.reshape(*sh[:-1], n // 128, 128) @ h).reshape(sh) * s
    return f


def quant_rows(W, H, K, seed):
    """trellis2's procedure on rows W [R, Kd] (original basis) -> (decoded rows, original basis; chosen factor)"""
    R, Kd = W.shape
    rk = e3.had_blocks(Kd, seed); ri = had_inv(Kd, seed)
    Wr = rk(W); Hr = rk(rk(H).T).T
    rms = Wr.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    damp = float(os.environ.get("E3_DAMP", 0.01))
    F = tuple(float(v) for v in os.environ.get("E3_FACTORS", "1.0").split(","))
    cb = np.ascontiguousarray(trellis2.codebook(CB), dtype=np.float32)
    pad = (-R) % 16
    Ws = torch.cat([torch.cat([Wr / (rms * f), torch.zeros(pad, Kd, dtype=Wr.dtype)], 0) for f in F], 0)
    Qs = trellis.ldlq_trellis(Ws, Hr, cb, K, damp)
    Rp = R + pad
    rf = [e3.rel(Wr, Qs[i * Rp:i * Rp + R] * rms * f, Hr) for i, f in enumerate(F)]
    i = int(np.argmin(rf)); f = F[i]
    return ri(Qs[i * Rp:i * Rp + R] * rms * f), f, Hr, Wr


def uq(layer, K, r0, r1, outdir):
    d = f"{outdir}/L{layer}-K{K:g}"; os.makedirs(d, exist_ok=True)
    out = f"{d}/gu-{r0:05d}.npy"
    if os.path.exists(out): return
    W = e3.load_w(layer, ["mlp.gate_proj", "mlp.up_proj"])[r0:r1]
    H = e3.load_h(layer, "post_attention.input")
    t0 = time.time()
    Q, f, _, _ = quant_rows(W, H, K, 1000 + layer)
    np.save(out + ".part.npy", Q.float().numpy()); os.replace(out + ".part.npy", out)
    print(json.dumps({"stage": "uq", "layer": layer, "K": K, "rows": [r0, r1], "factor": f,
                      "relerr_plain": float(((W - Q) ** 2).sum() / (W ** 2).sum()), "sec": round(time.time() - t0, 1)}), flush=True)


def down(layer, K, arm, outdir, out_path):
    D = e3.load_w(layer, ["mlp.down_proj"])                        # [N=5120, I=17408]
    N, I = D.shape
    H = e3.load_h(layer, "mlp.down_proj.input")
    g = torch.Generator().manual_seed(7 + layer + 100003 * int(os.environ.get("E3_ROWSEED", 0)))
    idx = torch.randperm(N, generator=g)[:int(os.environ.get("A1_ROWS", 256))].sort()[0]
    extra = {}
    if arm == "D":
        T = D[idx]
    else:
        if arm.endswith("true"):
            GU = e3.load_w(layer, ["mlp.gate_proj", "mlp.up_proj"])
        else:
            d = f"{outdir}/L{layer}-K{K:g}"
            parts = sorted(glob.glob(f"{d}/gu-*.npy"))
            GU = torch.from_numpy(np.concatenate([np.load(p) for p in parts], 0)).double()
        assert GU.shape == (2 * I, N), GU.shape
        G, U = GU[:I], GU[I:]                                       # decoded gate / up rows [I, N]
        if arm.endswith("perm"):
            pm = torch.randperm(I, generator=torch.Generator().manual_seed(12345)); G, U = G[pm], U[pm]
        P = [U] if arm in ("u", "uperm") else [U, G]
        # per neuron j: D[:, j] ~ sum_p c_pj P_p[j, :]  (least squares over the N outputs)
        X = torch.stack(P, 2)                                       # [I, N, np]
        A = torch.einsum("jnp,jnq->jpq", X, X) + 1e-12 * torch.eye(len(P), dtype=X.dtype)
        b = torch.einsum("jnp,nj->jp", X, D)
        c = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)      # [I, np]
        pred = torch.einsum("jnp,jp->nj", X, c)                     # [N, I]
        del X; R = D - pred; del pred                               # (memory: these jobs are the long pole)
        e = lambda M: float(torch.einsum("ij,jk,ik->", M, H, M))     # ratios only: H's scale cancels
        extra = {"pred_share": 1 - e(R[idx]) / e(D[idx]), "pred_share_all_rows": 1 - e(R) / e(D)}
        T = R[idx]
        del GU, G, U, R
    t0 = time.time()
    Q, f, Hr, Tr = quant_rows(T, H, K, 1000 + layer)
    Dr = D[idx]
    dT = T - Q
    rel_D = float(torch.einsum("ij,jk,ik->", dT, H, dT) / torch.einsum("ij,jk,ik->", Dr, H, Dr))
    o = {"layer": layer, "w": "ffn_down", "arm": arm, "K": K, "cb": CB, "rel_D": rel_D, "factor": f,
         "rows": len(idx), "sec": round(time.time() - t0, 1), **extra}
    print(json.dumps(o), flush=True)
    with open(out_path, "a") as fh: fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "uq":
        uq(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif sys.argv[1] == "down":
        down(int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        raise SystemExit(__doc__)
