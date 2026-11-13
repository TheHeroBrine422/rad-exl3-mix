#!/usr/bin/env python3
"""abgptq.py CKPT HESS ABQ LIBQUANT OUT [--jobs N] -- arms G and R of RULES-178: every GDN layer's in_proj_a / in_proj_b
through radiance libquant's gptqsim in the IQ8R recipe's form (codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym
rule=search), a | b fused (they read the same input; GPTQ rows are independent), G with calib= (the published
layer-input Hessian written as RADGRAM2), R without (round to nearest). Writes OUT/ab-G.safetensors and OUT/ab-R.safetensors
({blk.N.ssm_a.weight, blk.N.ssm_b.weight}: bf16 [48, 5120], the decoded values in the weight's own domain) and
OUT/proxy.jsonl (tr(E H E^T) / tr(W H W^T) per weight, H the undamped calib Hessian). Restartable per layer (OUT/l/)."""
import argparse, json, os, struct, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument("ckpt"); p.add_argument("hess"); p.add_argument("abq"); p.add_argument("libquant"); p.add_argument("out")
p.add_argument("--jobs", type=int, default=3); p.add_argument("--layers", default="0-63")
a = p.parse_args()
L = "model.language_model.layers"
GRID = ["codes=i8", "group=128", "scale=bf16", "transform=fwht128", "clamp=sym", "rule=search"]
idx = json.load(open(f"{a.ckpt}/model.safetensors.index.json"))["weight_map"]
lo, hi = map(int, a.layers.split("-"))
layers = [l for l in range(lo, hi + 1) if l % 4 != 3]


def ck(name):
    with safe_open(f"{a.ckpt}/{idx[name]}", "pt") as f: return f.get_tensor(name).float()


def hessian(l):
    pth = f"{a.hess}/{L}.{l}.input.safetensors"
    with open(pth, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    k, rows = int(meta["k"]), int(meta["rows"])
    with safe_open(pth, "pt") as f: up = f.get_tensor("hessian_upper").float()
    H = torch.zeros(k, k); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = up
    return H + H.triu(1).T, rows


def layer(l):
    d = f"{a.out}/l/{l:02d}"; done = f"{d}/done.json"
    if os.path.exists(done): return json.load(open(done))
    os.makedirs(d, exist_ok=True)
    H, rows = hessian(l)
    k = H.shape[0]
    gram = f"{d}/gram.r0.blk.{l}.ssm_a.bin"
    with open(gram + ".part", "wb") as fh:   # RADGRAM2: magic, k u32, rank u32, rows u64, domain u32, pad (arch/common/rad_arch.h)
        fh.write(b"RADGRAM2" + struct.pack("<IIQII", k, 0, rows, 0, 0)); fh.write(H.numpy().astype(np.float32).tobytes())
    os.replace(gram + ".part", gram)
    rec = {}
    # a and b are one call: GPTQ's rows are independent (one U, scales per row and group), so the fused [96, 5120]
    # weight gives each row what its own weight would, at one factorization a layer
    Ws = [ck(f"{L}.{l}.linear_attn.in_proj_{ab}.weight") for ab in ("a", "b")]
    W = torch.cat(Ws); n_a = Ws[0].shape[0]
    src = f"{d}/ab.f32"; W.numpy().astype(np.float32).tofile(src)
    for arm, extra in (("G", [f"calib={d}"]), ("R", [])):
        o = f"{d}/ab-{arm}.bf16"
        if not os.path.exists(o):
            r = subprocess.run([a.abq, a.libquant, "gptqsim", f"blk.{l}.ssm_a.weight", str(W.shape[0]), str(W.shape[1]),
                                src, o + ".part", *GRID, *extra], capture_output=True, text=True)
            if r.returncode: sys.exit(f"L{l} {arm}: abq rc {r.returncode}: {r.stderr}")
            if r.stderr.strip(): print(f"L{l} {arm}: {r.stderr.strip()}", flush=True)
            os.replace(o + ".part", o)
        Q = torch.from_numpy(np.fromfile(o, dtype=np.int16).copy()).view(torch.bfloat16).float().view_as(W)
        for ab, sl in (("a", slice(0, n_a)), ("b", slice(n_a, None))):
            Wd, E = W[sl].double(), (W[sl] - Q[sl]).double()
            rec[f"{ab}-{arm}"] = float(torch.einsum("nk,kj,nj->", E, H.double(), E) / torch.einsum("nk,kj,nj->", Wd, H.double(), Wd))
    os.remove(src)
    os.remove(gram)
    json.dump(rec, open(done + ".part", "w")); os.replace(done + ".part", done)
    print(f"L{l:02d} " + " ".join(f"{k_} {v:.3e}" for k_, v in rec.items()), flush=True)
    return rec


os.makedirs(a.out, exist_ok=True)
with ThreadPoolExecutor(a.jobs) as ex: recs = dict(zip(layers, ex.map(layer, layers)))
with open(f"{a.out}/proxy.jsonl", "w") as fh:
    for l in layers: fh.write(json.dumps({"layer": l, **recs[l]}) + "\n")
if (lo, hi) == (0, 63):
    for arm in ("G", "R"):
        t = {}
        for l in layers:
            raw = np.fromfile(f"{a.out}/l/{l:02d}/ab-{arm}.bf16", dtype=np.int16).copy()
            q = torch.from_numpy(raw).view(torch.bfloat16).view(96, -1)
            t[f"blk.{l}.ssm_a.weight"], t[f"blk.{l}.ssm_b.weight"] = q[:48].contiguous(), q[48:].contiguous()
        save_file(t, f"{a.out}/ab-{arm}.safetensors")
    print("wrote", [f"{a.out}/ab-{x}.safetensors" for x in ("G", "R")])
