#!/usr/bin/env python3
"""unshrink196.py OUT.safetensors --level LEVEL --model BF16 --hessians HDIR [--force-one] [--selftest KEY] [--only TYPE,..]
-- #196 U (RULES196): the closed-form un-shrink of every output row, on the listed linears.

For a linear with bf16 weight W (torch layout [n, k]: row r = output channel r) and the served decoded weight Q
(the virtual level's codes with its current row scales, i.e. the row the served path computes), the per-row factor is

    c_r = sqrt( w_r H w_r^T / q_r H q_r^T ),   clamped to [0.9, 1.2],

H = the calibration input Gram of the linear (the published calib hessians: hessian_upper = the packed row-major upper
triangle of sum x x^T; H is symmetric), w_r = W[r, :], q_r = Q[r, :]. Scaling output row r by c_r restores the bf16 row's
output energy under the calibration inputs (the quantized row shrinks it). The factor multiplies the row's output-side
scale: qad196.py sets it as the start value of the output-side log multiplier (c param = log c_r), so it is stored in the
checkpoint's scales with the trained part.

Served decoded row (QLinear.forward, rewritten): y = ((x * su) A) wq B * (sv * s), with A / B the blockwise 128-Hadamards
on the input / output index. So Q[j, i] = sv_j * s * su_i * M[j, i] with M = B wq^T A (computed here as two blockwise
Hadamard products). --selftest KEY checks that Q x^T equals QLinear.forward on a random x (the served form's arithmetic).
--force-one writes c = 1 for every linear (the identity check: the trainer's ckpt scales must then be all ones).
The output: safetensors, key = the student's qlin key (e.g. model.language_model.layers.3.self_attn.o_proj), value [n] f32;
metadata JSON with per-linear stats (raw ratios before the clamp, clamp fractions)."""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad"))
sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import qmodel  # noqa: E402
from rxq import deq  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

TYPES = {"self_attn.o_proj": "attn_output", "linear_attn.out_proj": "ssm_out", "mlp.down_proj": "ffn_down",
         "self_attn.v_proj": "attn_v", "linear_attn.in_proj_qkv": "ssm_qkv"}
LO, HI = 0.9, 1.2


def blockwise(X, dim, H):
    """X with its `dim` index split into 128-blocks, each block multiplied on the right by H (the 128-Hadamard)"""
    X = X.movedim(dim, -1); sh = X.shape
    X = (X.reshape(*sh[:-1], sh[-1] // 128, 128) @ H).reshape(sh)
    return X.movedim(-1, dim)


def served_rows(ql):
    """Q (f64 [n, k]) = the served decoded weight with the current row scales: Q[j, i] = sv_j s su_i M[j, i]"""
    H = ql.H.double()
    wq = ql.wq.double() if ql.wq.dtype != torch.bfloat16 else ql.wq.float().double()
    M = blockwise(wq.T, 1, H)                          # wq^T A   (n x k), A on the input index
    M = blockwise(M, 0, H)                             # B (wq^T A)  (B on the output index)
    s = float(ql.s) if not torch.is_tensor(ql.s) else float(ql.s.reshape(-1)[0])
    su = ql.suh.double() if ql.r is None else (ql.suh.double() * ql.r.detach().double().exp())
    sv = ql.svh.double() if ql.c is None else (ql.svh.double() * ql.c.detach().double().exp())
    return (sv * s)[:, None] * M * su[None, :]


def gram(path):
    with safe_open(path, "pt") as f:
        k = int(f.metadata()["k"]); up = f.get_tensor("hessian_upper").double()
    H = torch.zeros(k, k, dtype=torch.float64)
    iu = torch.triu_indices(k, k)
    H[iu[0], iu[1]] = up
    return H + H.T - torch.diag(H.diagonal())


def hfile(hdir, key):
    layer = key.split(".self_attn.")[0] if ".self_attn." in key else key.split(".linear_attn.")[0]
    if key.endswith((".self_attn.v_proj", ".linear_attn.in_proj_qkv")): return f"{hdir}/{layer}.input.safetensors"
    return f"{hdir}/{key}.input.safetensors"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--level", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--hessians", required=True); ap.add_argument("--force-one", action="store_true")
    ap.add_argument("--selftest", default=None); ap.add_argument("--only", default=",".join(TYPES.values()))
    ap.add_argument("--dev", default="cpu")
    a = ap.parse_args()
    want = set(a.only.split(","))
    lev = deq.Level(a.level, "cpu"); cb = lev.codebook
    table = torch.from_numpy(deq.lut(cb)).to(a.dev)     # the decode indexes the table on the linear's device
    idx = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
    keys = sorted(k for k in lev.keys() if any(k.endswith("." + t) for t in TYPES) and TYPES[[t for t in TYPES if k.endswith("." + t)][0]] in want)
    if a.selftest: keys = [a.selftest]
    out, stats = {}, {}
    for k in keys:
        t0 = time.time()
        if a.force_one:                                  # identity check: no decode, c = 1 on the level's n rows
            n = int(lev.raw(k)["svh"].shape[0]); c = torch.ones(n, dtype=torch.float64)
            out[k] = c.float().contiguous(); stats[k] = {"n": n, "forced": 1.0}
            print(k, json.dumps(stats[k]), flush=True)
            continue
        ql = qmodel.QLinear(k, lev.raw(k), cb, table, a.dev, train_r=False, train_c=False)
        n, kk = ql.n, ql.k
        if False:
            pass
        else:
            with safe_open(f"{a.model}/{idx[k + '.weight']}", "pt") as f: W = f.get_tensor(k + ".weight").double()
            assert tuple(W.shape) == (n, kk), (k, tuple(W.shape), (n, kk))
            Hg = gram(hfile(a.hessians, k)); assert Hg.shape == (kk, kk), (k, Hg.shape)
            if a.dev != "cpu": W = W.to(a.dev); Hg = Hg.to(a.dev)      # GPU: the f64 products (CPU f64 is ~20x slower here)
            Q = served_rows(ql).to(W.device)
            if a.selftest:
                x = torch.randn(4, kk, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
                yq = (ql.forward(x.float().to(a.dev)).double().cpu())
                ym = x @ Q.T
                rel = float((yq - ym).norm() / ym.norm())
                print(json.dumps({"selftest": k, "rel_err_vs_forward": rel, "n": n, "k": kk}), flush=True)
                return
            wHw = ((W @ Hg) * W).sum(1); qHq = ((Q @ Hg) * Q).sum(1)
            ratio = torch.where(qHq > 0, wHw / qHq.clamp_min(1e-300), torch.ones_like(wHw))
            c = ratio.clamp_min(0).sqrt().clamp(LO, HI).cpu()
            r = ratio.clamp_min(0).sqrt().cpu()
            stats[k] = {"n": n, "k": kk, "raw_mean": float(r.mean()), "raw_min": float(r.min()), "raw_max": float(r.max()),
                        "clamp_lo": float((r < LO).double().mean()), "clamp_hi": float((r > HI).double().mean()),
                        "sec": round(time.time() - t0, 1)}
            del W, Hg, Q; torch.cuda.empty_cache() if a.dev != "cpu" else None
        out[k] = c.float().cpu().contiguous()
        print(k, json.dumps(stats.get(k, {})), flush=True)
    if a.selftest: return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(out, a.out + ".part", metadata={"stats": json.dumps(stats), "lo": str(LO), "hi": str(HI), "types": ",".join(sorted(want)),
                                             "level": a.level, "hessians": a.hessians})
    os.replace(a.out + ".part", a.out)
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=1)
    print(f"wrote {a.out}: {len(out)} linears", flush=True)


if __name__ == "__main__":
    main()
