#!/usr/bin/env python3
"""downperm.py CKPT HIN PERMS.npz OUT.json -- B6's other side: an MLP neuron permutation also permutes down_proj's
INPUT channels, which changes how the 128-point input Hadamard mixes them and so the input-side LDLQ cost.
Per layer and permutation (PERMS.npz: predict.py --perms-out, L<l>/<name> int arrays) the expected input-side cost with white
rounding noise, as ourq's base settings build it (scaled Hessian): Hd = H + 0.025 mean(diag H) I in the original
basis, S = diag(su) with su_i = RMS over outputs of row i of W_down / sv (regularize's input scales, global factors
cancel) times random signs, A = P S Hd S P (blockwise 128 Hadamard), block-16 LDL A = L D L^T, cost =
tr(L^-1 (P S H S P) L^-T) (undamped H). Reported relative to the natural order, two sign draws each.
CPU, float64 (DP_DTYPE=float32 for the 27B's 17408^2: ~X min a layer on 36 cores, see README); ~4 min on 4 threads
for the 0.8B (24 x 3584^2). DP_DRAWS (2) sign draws."""
import base64, io, json, os, struct, sys, time
import numpy as np, torch
from safetensors import safe_open
if "had" not in globals():                         # run.sh's stdin bundle defines blk's functions before this file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from blk import had, ldl_unit, HB

ckpt, hin, pf, out = sys.argv[1:5]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
DT = getattr(torch, os.environ.get("DP_DTYPE", "float64")); DRAWS = int(os.environ.get("DP_DRAWS", 2))
PZ = np.load(io.BytesIO(base64.b64decode(PERMS_B64)) if pf == "-" else pf)
idx = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
def ck(name):
    with safe_open(f"{ckpt}/{idx[name]}", "pt") as f: return f.get_tensor(name)
def hess(g):
    p = f"{hin}/{g}.safetensors"
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    k = int(meta["k"])
    with safe_open(p, "pt") as f: up = f.get_tensor("hessian_upper").to(DT)
    H = torch.zeros(k, k, dtype=DT); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up
    return H + H.triu(1).T
def rot(X):                                        # blockwise 128 Hadamard on both sides
    P = had(dtype=X.dtype); k = X.shape[0]; B = k // HB
    X = X.view(k, B, HB) @ P; X = X.view(k, k)
    return (P @ X.view(B, HB, k)).view(k, k)
def cost(H, su, sign, p):
    s = (su * sign)[p] if p is not None else su * sign
    Hp = H[p][:, p] if p is not None else H
    G = rot(Hp * s[:, None] * s[None, :])
    Hd = Hp.clone(); Hd.diagonal().add_(0.025 * Hp.diagonal().mean())
    A = rot(Hd * s[:, None] * s[None, :])
    L, _ = ldl_unit(A[None]); L = L[0]
    Li = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], dtype=L.dtype), upper=False, unitriangular=True)
    return float(((Li @ G) * Li).sum()), float(G.diagonal().sum())
res = {}; t0 = time.time()
layers = sorted({int(k.split("/")[0][1:]) for k in PZ.files})
for l in layers:
    pre = f"model.language_model.layers.{l}.mlp.down_proj"
    W = ck(pre + ".weight").to(DT)                       # (hidden, inter): rows = outputs
    sv = W.pow(2).mean(1).sqrt(); sv = sv / sv.mean()
    su = (W / sv[:, None]).pow(2).mean(0).sqrt()           # per input channel (Hadamard over outputs keeps it)
    H = hess(pre + ".input")
    r = {}
    for draw in range(DRAWS):
        sign = torch.from_numpy(np.random.default_rng(1000 * l + draw).choice([-1.0, 1.0], H.shape[0])).to(DT)
        c0, one = cost(H, su, sign, None)
        for nm in [n.split("/")[1] for n in PZ.files if n.startswith(f"L{l}/")]:
            c, _ = cost(H, su, sign, torch.from_numpy(PZ[f"L{l}/{nm}"]))
            r.setdefault(nm, []).append(c / c0)
        r.setdefault("natural_vs_onesided_noLDL", []).append(c0 / one)
    res[l] = {k: float(np.mean(v)) for k, v in r.items()} | {f"{k}_draws": v for k, v in r.items()}
    print(f"layer {l}: " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in r.items()) + f"  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
txt = json.dumps(res, indent=1)
if out == "-": print(txt)
else: open(out, "w").write(txt)
