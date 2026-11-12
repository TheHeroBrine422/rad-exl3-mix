#!/usr/bin/env python3
"""score119.py DUMPDIR CKPT OUTDIR SPEC [SPEC ...] -- exact head-only KL of head options on a head-input dump (w119, RULES.md),
headsim94's arithmetic on one card, every gen row kept, plus per-128-row-block sums (score96's d / delta) in a stored order.

Reference: bf16 dump rows x bf16 lm_head, f32 accumulation; p = softmax. An option's logits z replace the reference's.
SPEC = NAME=KIND:ARGS (comma-separated key=value), KIND:
  exl3   wq=DIR/wq/lm_head.safetensors  perm=P.npy (the stored order: stored row i = vocab row P[i]; 'id' = none)
         [act=f16|a8] [qhead=DIR/q-head.safetensors (a8: the donor's suh)] [blocks=1]
  bf16   perm=P.npy [blocks=1]                     the bf16 rows cast to f16, f16 inputs (a bf16 class)
  int8   npz=FILE (codes int8 [V, 5120], scale f32 [V, 40], vocab order) [act=bf16|a8] [fwht=0|1]
Per option OUTDIR/<NAME>.npz: kl [N] f32, hit [N] u8 (top-1 = reference argmax), and with blocks=1 also d, dl [N, 1940]
f32 in the stored order (KL_r = sum_b d + log1p(sum_b dl) for any block -> donor plan of donors that share the order).
OUTDIR/ref.npz: rows (dump row of each gen row), doc, lse, top (vocab id). Existing outputs are skipped (restartable).
a8 = the A8 path's input: x~ = had128(x * suh) per 128 inputs (orthonormal Sylvester), int8 round-half-even with scale
absmax / 127 per (row, 128), mapped back exactly (exl3); for int8 the same on the unrotated (or fwht128) input."""
import json, os, sys, time
import numpy as np, torch
from safetensors import safe_open

DUMP, CK, OUT = sys.argv[1:4]; SPECS = sys.argv[4:]
RB, NB, BLK = 128, 1940, int(os.environ.get("SC_BLK", 256))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # cpu: tests only
torch.backends.cuda.matmul.allow_tf32 = False
os.makedirs(OUT, exist_ok=True)
H, bf, N = open(f"{DUMP}/336.meta").read().split(); H, N = int(H), int(N)
gen = np.fromfile(f"{DUMP}/gen.u8", np.uint8).astype(bool); rows = np.nonzero(gen)[0]
if os.environ.get("SC_MAXROWS"): rows = rows[np.unique(np.linspace(0, len(rows) - 1, int(os.environ["SC_MAXROWS"])).astype(int))]
X = torch.from_numpy(np.fromfile(f"{DUMP}/336.x", np.int16).reshape(N, H)[rows]).view(torch.bfloat16)
n = len(rows)
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: Wref = f.get_tensor("lm_head.weight").to(dev)
V = Wref.shape[0]; assert V == NB * RB


def mm(x, w):                                         # f32 products of 16-bit operands
    if dev.type == "cuda": return torch.mm(x, w.T, out_dtype=torch.float32)
    xf = x.float(); return torch.cat([xf @ w[i:i + 16384].float().T for i in range(0, w.shape[0], 16384)], 1)


def had128():
    h = torch.tensor([[1.0]])
    while h.shape[0] < 128: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / 128 ** 0.5).to(dev)


HAD = had128()


def a8(x, pre=None, rot=True):
    """x [m, 5120] f32 -> the input the int8 path effectively multiplies (f32)"""
    y = x * pre if pre is not None else x
    y = y.view(-1, 40, 128)
    if rot: y = y @ HAD
    s = y.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    y = torch.round(y / s).clamp(-127, 127) * s
    if rot: y = y @ HAD.T
    y = y.reshape(-1, 5120)
    return y / pre if pre is not None else y


def parse(s):
    name, rest = s.split("=", 1); kind, _, args = rest.partition(":")
    kv = dict(a.split("=", 1) for a in args.split(",") if a)
    return name, kind, kv


# reference pass (cached): lse, argmax, and p kept per chunk on the fly
ref_f = f"{OUT}/ref.npz"
if not os.path.exists(ref_f):
    lse = np.zeros(n, np.float32); top = np.zeros(n, np.int64)
    with torch.no_grad():
        for b0 in range(0, n, BLK):
            x = X[b0:b0 + BLK].to(dev); zr = mm(x, Wref)
            lse[b0:b0 + len(x)] = torch.logsumexp(zr, -1).cpu().numpy(); top[b0:b0 + len(x)] = zr.argmax(-1).cpu().numpy()
    doc = np.fromfile(f"{DUMP}/doc.i32", np.int32)[rows]
    np.savez(ref_f, rows=rows, doc=doc, lse=lse, top=top)
for spec in SPECS:
    name, kind, kv = parse(spec); of = f"{OUT}/{name}.npz"
    if os.path.exists(of): print(f"{name}: done", file=sys.stderr); continue
    t0 = time.time(); blocks = kv.get("blocks") == "1"; act = kv.get("act", "f16" if kind != "int8" else "bf16")
    perm = None
    if kv.get("perm", "id") != "id": perm = torch.from_numpy(np.load(kv["perm"])).to(dev)
    pre = None; rot = True
    if kind == "exl3":
        with safe_open(kv["wq"], "pt") as f: W = f.get_tensor("weight").to(dev)          # f16, stored order
        if act == "a8":
            with safe_open(kv["qhead"], "pt") as f: pre = f.get_tensor("lm_head.suh").float().to(dev)
            W = W.float()
    elif kind == "bf16":
        W = (Wref[perm] if perm is not None else Wref).to(torch.float16)
    elif kind == "int8":
        z = np.load(kv["npz"]); W = (torch.from_numpy(z["codes"]).to(dev).float() * torch.from_numpy(z["scale"]).to(dev).float().repeat_interleave(128, 1))
        rot = kv.get("fwht") == "1"
        # fwht=1: codes x scale approximate W in the rotated basis (W @ had per 128 inputs): the input is rotated instead
    else: sys.exit(f"unknown kind {kind}")
    kl = np.zeros(n, np.float32); hit = np.zeros(n, np.uint8)
    if blocks: D = np.zeros((n, NB), np.float32); DL = np.zeros((n, NB), np.float32)
    with torch.no_grad():
        for b0 in range(0, n, BLK):
            x = X[b0:b0 + BLK].to(dev); m = len(x)
            zr = mm(x, Wref)
            lp = zr - torch.logsumexp(zr, -1, keepdim=True); p = lp.exp(); tp = zr.argmax(-1)
            if kind == "int8":
                xi = x.float()
                if rot: xi = (xi.view(-1, 40, 128) @ HAD).reshape(-1, 5120)
                if act == "a8": xi = a8(xi, None, False)
                zq = xi @ W.T                                                             # f32 (W8A16: f32 accumulate)
            elif act == "a8":                                                             # f16 input -> A8, f32 products
                zq = a8(x.to(torch.float16).float(), pre, True) @ W.T
            else:
                zq = mm(x.to(torch.float16), W)
            if perm is not None:                                                          # stored order: compare to ref columns P
                p = p[:, perm]; zr = zr[:, perm]
            dz = zq - zr
            d = -(p * dz); dl = p * torch.expm1(dz)
            ds, dls = d.view(m, NB, RB).sum(-1), dl.view(m, NB, RB).sum(-1)
            kl[b0:b0 + m] = (ds.sum(-1, dtype=torch.float64) + torch.log1p(dls.sum(-1, dtype=torch.float64))).cpu().numpy()
            am = zq.argmax(-1); am = perm[am] if perm is not None else am
            hit[b0:b0 + m] = (am == tp).cpu().numpy()
            if blocks: D[b0:b0 + m] = ds.cpu().numpy(); DL[b0:b0 + m] = dls.cpu().numpy()
    del W; torch.cuda.empty_cache()
    out = {"kl": kl, "hit": hit}
    if blocks: out.update(d=D, dl=DL)
    np.savez(of + ".part.npz", **out); os.replace(of + ".part.npz", of)
    print(f"{name} ({kind} {act}{' blocks' if blocks else ''}): mean KL {kl.mean():.4e} top-1 {hit.mean():.4f} "
          f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
