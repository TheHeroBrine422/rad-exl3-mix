#!/usr/bin/env python3
"""selftest2.py -- CPU checks of the GPU-arm plumbing on a synthetic 2-layer checkpoint (output: selftest2.txt):
 1. mkperm build: the permuted MLP computes the same function; the permuted down_proj Hessian = the Hessian of the
    permuted inputs; the permuted h vectors follow the rows; untouched files are symlinks;
 2. mkperm unperm: a "quant" equal to the permuted weights comes back as the original weights, bit for bit, and the
    non-MLP tensors are the donor's;
 3. splice: sorted / random take exactly half the 128-blocks from the high donor (sorted: the top predicted costs),
    uniform links the uniform donor."""
import json, os, subprocess, sys, tempfile
import numpy as np, torch
from safetensors.torch import save_file, load_file
H_ = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H_)
import blk
T = tempfile.mkdtemp(); ok = True; L = "model.language_model.layers"
def check(n, c, m):
    global ok; ok &= bool(c); print(f"{'PASS' if c else 'FAIL'} {n}: {m}")
g = torch.Generator().manual_seed(0); hid, inter, NL = 128, 512, 2
ck, hin, hd = f"{T}/ck", f"{T}/hin", f"{T}/hd"
for d in (ck, hin, hd): os.makedirs(d)
W = {}
for l in range(NL):
    W[f"{L}.{l}.mlp.gate_proj.weight"] = torch.randn(inter, hid, generator=g).bfloat16()
    W[f"{L}.{l}.mlp.up_proj.weight"] = torch.randn(inter, hid, generator=g).bfloat16()
    W[f"{L}.{l}.mlp.down_proj.weight"] = torch.randn(hid, inter, generator=g).bfloat16()
    W[f"{L}.{l}.linear_attn.out_proj.weight"] = torch.randn(hid, hid, generator=g).bfloat16()
save_file(W, f"{ck}/m.safetensors"); json.dump({"weight_map": {k: "m.safetensors" for k in W}}, open(f"{ck}/model.safetensors.index.json", "w"))
json.dump({"text_config": {}}, open(f"{ck}/config.json", "w"))
X = {}
for l in range(NL):
    x = torch.randn(1000, inter, generator=g) * torch.linspace(0.1, 3, inter); X[l] = x
    H = x.T @ x; iu = torch.triu_indices(inter, inter)
    save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{hin}/{L}.{l}.mlp.down_proj.input.safetensors", metadata={"k": str(inter), "rows": "1000"})
    for x_ in ("gate", "up"): save_file({"hout_diag": torch.rand(inter, generator=g)}, f"{hd}/{L}.{l}.mlp.{x_}_proj.safetensors")
    save_file({"hout_diag": torch.rand(hid, generator=g)}, f"{hd}/{L}.{l}.mlp.down_proj.safetensors")
PZ = {f"L{l}/balanced": np.random.default_rng(l).permutation(inter) for l in range(NL)}
np.savez(f"{T}/perms.npz", **PZ)
run = lambda *a: subprocess.run([sys.executable, f"{H_}/mkperm.py", *a], check=True, capture_output=True, text=True).stdout.strip()
print("INFO", run("build", f"{T}/perms.npz", "balanced", ck, hin, hd, f"{T}/p"))
P = load_file(f"{T}/p/model/model-perm.safetensors")
mlp = lambda Wd, l, x: (torch.nn.functional.silu(x @ Wd[f"{L}.{l}.mlp.gate_proj.weight"].float().T) * (x @ Wd[f"{L}.{l}.mlp.up_proj.weight"].float().T)) @ Wd[f"{L}.{l}.mlp.down_proj.weight"].float().T
x = torch.randn(7, hid, generator=g)
rel = max(float((mlp(W, l, x) - mlp(P, l, x)).norm() / mlp(W, l, x).norm()) for l in range(NL))
check("function", rel < 1e-5, f"permuted MLP = original, rel L2 {rel:.1e} (fp32 summation order)")
up = load_file(f"{T}/p/w/challenge/hessians/calib/{L}.1.mlp.down_proj.input.safetensors")["hessian_upper"]
p = torch.from_numpy(PZ["L1/balanced"]); xp = X[1][:, p]; Hp = xp.T @ xp; iu = torch.triu_indices(inter, inter)
check("hessian", torch.allclose(up, Hp[iu[0], iu[1]], rtol=1e-4, atol=1e-2), "permuted H = H of the permuted inputs")
hv = load_file(f"{T}/p/hout/{L}.1.mlp.up_proj.safetensors")["hout_diag"]
check("h vectors", torch.equal(hv, load_file(f"{hd}/{L}.1.mlp.up_proj.safetensors")["hout_diag"][p]) and
      os.path.islink(f"{T}/p/hout/{L}.1.mlp.down_proj.safetensors"), "gate/up h permuted, down's linked")
# 2. unperm
pw, dw = f"{T}/pwq", f"{T}/dwq"; os.makedirs(pw); os.makedirs(dw)
for k, v in P.items(): save_file({"weight": v.half()}, f"{pw}/{k[:-7]}.safetensors")
for k, v in W.items(): save_file({"weight": (v.half() + 1) if "out_proj" in k else torch.zeros_like(v).half()}, f"{dw}/{k[:-7]}.safetensors")
print("INFO", run("unperm", f"{T}/perms.npz", "balanced", pw, dw, f"{T}/uwq"))
U = {f[:-12]: load_file(f"{T}/uwq/{f}")["weight"] for f in os.listdir(f"{T}/uwq")}
check("unperm", all(torch.equal(U[k[:-7]], W[k].half() + (1 if "out_proj" in k else 0)) for k in W), "MLP back in natural order, others = donor")
# 3. splice (in_proj-like tensor of 4 blocks)
k = f"{L}.0.mlp.gate_proj"; kf = f"{T}/keys.txt"; open(kf, "w").write(k + "\n")
hi, lo, un = f"{T}/hi", f"{T}/lo", f"{T}/un"
for d_, v in ((hi, 1.0), (lo, 2.0), (un, 3.0)):
    os.makedirs(d_); save_file({"weight": torch.full((inter, hid), v).half()}, f"{d_}/{k}.safetensors")
sp = lambda *a: subprocess.run([sys.executable, f"{H_}/splice.py", *a], check=True, capture_output=True, text=True).stdout.strip()
for mode in ("sorted", "random"):
    sp(mode, ck, hd, kf, dw, f"{T}/s-{mode}", hi, lo)
    s = load_file(f"{T}/s-{mode}/{k}.safetensors")["weight"][:, 0].view(-1, 128)
    hib = (s == 1).all(1).numpy()
    good = hib.sum() == 2 and ((s == 1) | (s == 2)).all()
    if mode == "sorted":
        c, _ = blk.block_costs(load_file(f"{hd}/{k}.safetensors")["hout_diag"].double().numpy() * blk.out_scales2(W[k + ".weight"]))
        good &= set(np.where(hib)[0]) == set(np.argsort(-c)[:2])
    check(f"splice {mode}", good and os.path.islink(f"{T}/s-{mode}/{L}.0.mlp.up_proj.safetensors"), f"high-K blocks {np.where(hib)[0].tolist()}")
sp("uniform", ck, hd, kf, dw, f"{T}/s-u", un)
check("splice uniform", os.path.realpath(f"{T}/s-u/{k}.safetensors") == os.path.realpath(f"{un}/{k}.safetensors"), "linked to the uniform donor")
print("SELFTEST2", "PASS" if ok else "FAIL")
