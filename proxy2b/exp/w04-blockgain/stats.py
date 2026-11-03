#!/usr/bin/env python3
"""stats.py CKPT HDIAG OUT.npz [--hin DIR] [--wq TAG=DIR ...] [--keys a,b] [--state hdiag-state.pt] -- per trunk linear (the keys of HDIAG,
collect.py YAQA_DIAG / ourq --hout layout, "hout_diag"):
  h      the diagonal output factor (HDIAG/<key>.safetensors)
  sv2    regularize()'s output scales squared (row RMS of the bf16 weight / mean, zero rows 0.1), what prepare_H_out
         multiplies into the factor (before the scale refit; independent of K and seed)
with --hin (challenge-layout input Hessians, ourq's group names; divided by their row count):
  ref    w_c^T H w_c per output channel; trH  tr(H) (h_c sums the Fisher diagonal over a row's inputs, so
         diag(h) (x) H / tr(H) is the Kronecker estimate on the Fisher's scale: sum_c h_c e_c / tr(H) compares tensors)
  e/TAG  (w_c - q_c)^T H (w_c - q_c) for every --wq TAG=DIR (ourq --save-wq DIR/<key>.safetensors, (out, in) f16).
         Recomputed here from the bf16 weight: ourq's own rowerr.npz is not usable (see README, "rowerr.npz").
OUT '-' writes the npz to stdout. CPU only; threads from OMP_NUM_THREADS."""
import argparse, glob, io, json, os, struct, sys, time
import numpy as np, torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("hdiag"); ap.add_argument("out")
ap.add_argument("--hin"); ap.add_argument("--wq", action="append", default=[]); ap.add_argument("--keys")
ap.add_argument("--state", help="collect.py's hdiag-state.pt (YAQA_DIAG): also h_even / h_odd, the means over the even- and "
                "odd-numbered sequences (independent halves: predict.py's split-half cross-fit)")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
idx = json.load(open(f"{a.ckpt}/model.safetensors.index.json"))["weight_map"]
_open = {}
def ck(name):
    f = idx[name]
    if f not in _open: _open[f] = safe_open(f"{a.ckpt}/{f}", "pt", device="cpu")
    return _open[f].get_tensor(name)
def vec(p, name):
    with safe_open(p, "pt") as f: return f.get_tensor(name)

def group(key):                                     # ourq.py layer_plan: input-Hessian name of a linear
    p, rest = key.rsplit(".", 2)[0], ".".join(key.rsplit(".", 2)[1:])
    if rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z"):
        return f"{p}.input"
    return {"self_attn.o_proj": f"{p}.self_attn.o_proj.input", "linear_attn.out_proj": f"{p}.linear_attn.out_proj.input",
            "mlp.gate_proj": f"{p}.post_attention.input", "mlp.up_proj": f"{p}.post_attention.input",
            "mlp.down_proj": f"{p}.mlp.down_proj.input"}[rest]
_H = {}
def hess(g):
    if g not in _H:
        _H.clear(); p = f"{a.hin}/{g}.safetensors"
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
        k, rows = int(meta["k"]), int(meta["rows"])
        up = vec(p, "hessian_upper").float()
        H = torch.zeros(k, k); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up
        _H[g] = (H + H.triu(1).T) / rows
    return _H[g]
def rowquad(H, X):                                  # per row x^T H x of X (out, in), in row chunks
    return torch.cat([((X[i:i + 2048] @ H) * X[i:i + 2048]).sum(1) for i in range(0, X.shape[0], 2048)]).double().numpy()

keys = sorted(os.path.basename(f)[:-len(".safetensors")] for f in glob.glob(f"{a.hdiag}/*.safetensors"))
if a.keys: keys = [k for k in keys if any(s in k for s in a.keys.split(","))]
wq = dict(t.split("=", 1) for t in a.wq)
res = {}; t0 = time.time()
if a.state:
    st = torch.load(a.state, map_location="cpu"); n = st["state"]["n"]; ne, no = (n + 1) // 2, n // 2
    HALF = {"model.language_model.layers." + k: (st["acc_even"][k].double() / ne, (v.double() - st["acc_even"][k].double()) / no)
            for k, v in st["acc"].items()}
for key in sorted(keys, key=lambda k: (group(k), k)):          # group order: one Hessian load per group
    w = ck(key + ".weight").float()
    r = w.pow(2).mean(1).sqrt().double(); m = r.mean()
    s = r / m; s[s.abs() < 1e-30] = 0.1
    res[f"{key}/h"] = vec(f"{a.hdiag}/{key}.safetensors", "hout_diag").double().numpy()
    res[f"{key}/sv2"] = (s * s).numpy()
    if a.state: res[f"{key}/h_even"], res[f"{key}/h_odd"] = (x.numpy() for x in HALF[key])
    if a.hin:
        H = hess(group(key)); res[f"{key}/ref"] = rowquad(H, w); res[f"{key}/trH"] = np.array(float(H.diagonal().sum()))
        for tag, d in wq.items():
            q = vec(f"{d}/{key}.safetensors", "weight").float()
            res[f"{key}/e/{tag}"] = rowquad(H, w - q)
    print(f"{key} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
tc = json.load(open(f"{a.ckpt}/config.json")); tc = tc.get("text_config", tc)
res["_cfg"] = np.array(json.dumps({k: tc.get(k) for k in ("head_dim", "linear_key_head_dim", "linear_num_key_heads",
                                                          "linear_value_head_dim", "linear_num_value_heads")}))
if a.out == "-":
    b = io.BytesIO(); np.savez(b, **res); sys.stdout.buffer.write(b.getvalue())
else:
    np.savez(a.out, **res)
