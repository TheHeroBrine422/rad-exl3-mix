#!/usr/bin/env python3
"""test117.py CKPT HESS TMP -- perm117.py CLI checks (../w84.md item 1): (1) `hess` on 3 calib Hessians (a residual input,
a writer input, lm_head.input) then `hess` again with the inverse order gives the original packed triangles bit for bit,
the writer-input file is a link, metadata kept; (2) `ckpt` on one shard of CKPT (layer 0's): every tensor equals
permute_tensor() of the original (or the original), dtype / metadata kept, and the inverse order restores it."""
import json, os, shutil, subprocess, sys
import torch
from safetensors import safe_open
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import perm117
CK, HS, T = sys.argv[1:4]
shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
so, idx = perm117.load_order(f"{HERE}/order82.json")
inv_so = torch.argsort(torch.tensor(so)).tolist()
json.dump({"stored_order": inv_so}, open(f"{T}/inv.json", "w"))
_, iidx = perm117.load_order(f"{T}/inv.json"); assert torch.equal(idx[iidx], torch.arange(5120))
run = lambda *a: subprocess.run([sys.executable, f"{HERE}/perm117.py", *a], check=True, capture_output=True, text=True).stdout.strip()
# (1) Hessians
os.makedirs(f"{T}/h0")
names = ["model.language_model.layers.0.input", "model.language_model.layers.0.linear_attn.out_proj.input", "lm_head.input"]
for n in names: os.symlink(f"{HS}/{n}.safetensors", f"{T}/h0/{n}.safetensors")
print(run("hess", f"{HERE}/order82.json", f"{T}/h0", f"{T}/h1")); print(run("hess", f"{T}/inv.json", f"{T}/h1", f"{T}/h2"))
ok = True
for n in names:
    a = perm117._hdr(f"{T}/h0/{n}.safetensors")["__metadata__"]; b = perm117._hdr(f"{T}/h2/{n}.safetensors")["__metadata__"]
    with safe_open(f"{T}/h0/{n}.safetensors", "pt") as f: u0 = f.get_tensor("hessian_upper")
    with safe_open(f"{T}/h1/{n}.safetensors", "pt") as f: u1 = f.get_tensor("hessian_upper")
    with safe_open(f"{T}/h2/{n}.safetensors", "pt") as f: u2 = f.get_tensor("hessian_upper")
    link = os.path.islink(f"{T}/h1/{n}.safetensors")
    perm = perm117.hess_permuted(n)
    r = torch.equal(u0, u2) and a == b and (link != perm) and (perm == (not torch.equal(u0, u1)))
    # one entry by hand: H'[p, q] = H[idx[p], idx[q]]
    if perm:
        k = 5120; iu = torch.triu_indices(k, k); H = torch.zeros(k, k, dtype=u0.dtype); H[iu[0], iu[1]] = u0; H = H + H.triu(1).T
        p, q = 100, 4000; r &= bool(u1[p * k - p * (p - 1) // 2 + (q - p)] == H[idx[p], idx[q]])
    ok &= r; print(f"hess {n}: dtype {u0.dtype}, permuted {perm}, link {link}, roundtrip+meta+entry {r}")
# (2) one checkpoint shard
wm = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
sh = wm["model.language_model.layers.0.mlp.up_proj.weight"]
os.makedirs(f"{T}/c0"); os.symlink(f"{CK}/{sh}", f"{T}/c0/{sh}")
print(run("ckpt", f"{HERE}/order82.json", f"{T}/c0", f"{T}/c1")); print(run("ckpt", f"{T}/inv.json", f"{T}/c1", f"{T}/c2"))
n = np = 0
with safe_open(f"{T}/c0/{sh}", "pt") as f0, safe_open(f"{T}/c1/{sh}", "pt") as f1, safe_open(f"{T}/c2/{sh}", "pt") as f2:
    assert set(f0.keys()) == set(f1.keys()) == set(f2.keys())
    for k in f0.keys():
        t0, t1, t2 = f0.get_tensor(k), f1.get_tensor(k), f2.get_tensor(k); p = perm117.permute_tensor(k, t0, idx)
        r = torch.equal(t1, t0 if p is None else p) and torch.equal(t2, t0) and t1.dtype == t0.dtype
        ok &= r; n += 1; np += p is not None
        if not r: print("MISMATCH", k)
m0, m1 = perm117._hdr(f"{T}/c0/{sh}").get("__metadata__"), perm117._hdr(f"{T}/c1/{sh}").get("__metadata__")
ok &= m0 == m1
print(f"ckpt {sh}: {n} tensors ({np} permuted), roundtrip and permute_tensor equal, metadata kept {m0 == m1}")
shutil.rmtree(T)
print(f"TEST117 {'PASS' if ok else 'FAIL'}")
