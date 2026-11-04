#!/usr/bin/env python3
"""a1q.py -- #81-04: A1 (down_proj coded as the residual after a per-neuron prediction from the decoded up / gate rows)
on the 0.8B, end to end in KL, with load-time expansion (allowed for measuring; shipping needs the online kernel).

  a1q.py res CKPT BASEWQ OUT ARM     ARM u | ug. From BASEWQ's decoded gate / up (ourq --save-wq, [out, in] f16), per
                                     neuron j least squares over the hidden outputs: D[:, j] ~ a_j U[j, :] (+ b_j G[j, :])
                                     (a1test.py's fit). Writes OUT/ckpt: the checkpoint with every trunk down_proj replaced
                                     by R = D - pred (bf16; the other files symlinked), and OUT/coef.pt.
  a1q.py expand OUT RWQ BASEWQ       RWQ: ourq's --save-wq of R (down_proj only). D_hat = R_hat + pred from BASEWQ's
                                     decoded predictors (exact, closed loop) -> OUT/wqall: BASEWQ with each down replaced.
"""
import glob, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
torch.set_grad_enabled(False)
P = "model.language_model.layers."

def wq(d, key): return load_file(f"{d}/{key}.safetensors")["weight"].double()

def preds(base, layers, coef, arm):
    for l in layers:
        k = f"{P}{l}.mlp"
        U = wq(base, f"{k}.up_proj"); G = wq(base, f"{k}.gate_proj")        # [I, N]
        c = coef[l]
        pred = U.T * c[:, 0]                                               # [N, I]
        if arm == "ug": pred = pred + G.T * c[:, 1]
        yield l, pred

if sys.argv[1] == "res":
    ck, base, out, arm = sys.argv[2:6]
    os.makedirs(f"{out}/ckpt", exist_ok=True)
    shard = sorted(glob.glob(f"{ck}/*.safetensors"))
    assert len(shard) == 1, shard
    for f in os.listdir(ck):
        if not f.endswith(".safetensors") and not os.path.exists(f"{out}/ckpt/{f}"): os.symlink(f"{ck}/{f}", f"{out}/ckpt/{f}")
    T = load_file(shard[0]); meta = safe_open(shard[0], "pt").metadata()
    layers = sorted({int(k.split(".")[3]) for k in T if k.startswith(P) and k.endswith("mlp.down_proj.weight")})
    coef, shares = {}, []
    for l in layers:
        k = f"{P}{l}.mlp"; D = T[f"{k}.down_proj.weight"].double()            # [N, I]
        U = wq(base, f"{k}.up_proj"); G = wq(base, f"{k}.gate_proj")
        X = torch.stack([U, G] if arm == "ug" else [U], 2)                   # [I, N, np]
        A = torch.einsum("jnp,jnq->jpq", X, X) + 1e-12 * torch.eye(X.shape[2], dtype=X.dtype)
        b = torch.einsum("jnp,nj->jp", X, D)
        c = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)               # [I, np]
        R = D - torch.einsum("jnp,jp->nj", X, c)
        coef[l] = torch.cat([c, torch.zeros(c.shape[0], 2 - c.shape[1], dtype=c.dtype)], 1)
        shares.append(1 - float((R * R).sum() / (D * D).sum()))
        T[f"{k}.down_proj.weight"] = R.to(T[f"{k}.down_proj.weight"].dtype).contiguous()
    save_file(T, f"{out}/ckpt/{os.path.basename(shard[0])}", metadata=meta)
    torch.save(coef, f"{out}/coef.pt")
    json.dump({"arm": arm, "layers": layers, "pred_share_unweighted": shares}, open(f"{out}/res.json", "w"))
    print(f"RES DONE {arm}: {len(layers)} layers, predicted share (unweighted) mean {sum(shares) / len(shares):.3f}")
elif sys.argv[1] == "expand":
    out, rwq, base = sys.argv[2:5]
    coef = torch.load(f"{out}/coef.pt"); arm = json.load(open(f"{out}/res.json"))["arm"]
    dst = f"{out}/wqall"; os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{base}/*.safetensors"):
        if ".mlp.down_proj." not in f and not os.path.exists(f"{dst}/{os.path.basename(f)}"): os.symlink(os.path.realpath(f), f"{dst}/{os.path.basename(f)}")
    n = 0
    for l, pred in preds(base, sorted(coef), coef, arm):
        key = f"{P}{l}.mlp.down_proj"
        Dh = wq(rwq, key) + pred
        save_file({"weight": Dh.half().contiguous()}, f"{dst}/{key}.safetensors"); n += 1
    assert len(glob.glob(f"{dst}/*.safetensors")) == len(glob.glob(f"{base}/*.safetensors")), "incomplete wqall"
    open(f"{dst}/.done", "w").close()
    print(f"EXPAND DONE {n} down_proj")
