#!/usr/bin/env python3
"""r1test.py DUMPDIR MAP Q.npy IDX[,IDX...] OUT.jsonl -- does an R1 residual-stream rotation (SpinQuant) lower the
int8/int4 activation-quantization error of the residual readers? (design/v3.md §7, the CPU test; aqproxy.py's dumps.)

Residual readers: attn q/k/v (attn_in), GDN in_proj qkv/z (gdn_in), gate_up -- their GEMM input is
x = RMSNorm(h) * (1 + w) (Qwen3.5 norm gain). R1 folds one orthogonal Q (5120², shared by every layer) into the
residual stream: the readers see u = (x / (1 + w)) Q, the gain and Q go into the weights, W' = W diag(1 + w) Q.
Per linear and config, the kernel's 128-block Hadamard + quantizer (aqproxy.rotate / quant) on:
  base   x with the current quant's suh (signs and learned input scales): today
  signs  x with sign(suh) only (separates the learned scales' share)
  r1     u with random signs (a requant would bring its own suh)
rel_out = sum ||W e_x||² / sum ||W x||², e_x the error mapped back to x space (for r1: e_x = (e_u Qᵀ) * (1 + w)).
One JSON line per (linear, variant, config). Q: make with  python3 r1test.py --makeq Q.npy  (seeded QR)."""
import json, math, os, sys
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aqproxy as A
import mkmask

CONFIGS = [(8, 128, 1.0), (8, 0, 1.0), (4, 128, 1.0), (4, 128, 0.85), (4, 64, 0.85)]
P = "model.language_model.layers."


def gain(layer, fam):
    n = f"{P}{layer}." + ("post_attention_layernorm.weight" if fam == "gate_up" else "input_layernorm.weight")
    idx = json.load(open(f"{A.CK}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{A.CK}/{idx[n]}", "pt") as f: return 1.0 + f.get_tensor(n).float().numpy()


def main():
    if sys.argv[1] == "--makeq":
        g = torch.Generator().manual_seed(1234)
        q, r = torch.linalg.qr(torch.randn(5120, 5120, generator=g, dtype=torch.float64))
        q = q * torch.sign(torch.diagonal(r))              # Haar-distributed
        np.save(sys.argv[2], q.float().numpy()); print("Q saved", q.shape); return
    D, mp, qf, idxs, out = sys.argv[1], sys.argv[2], sys.argv[3], [int(i) for i in sys.argv[4].split(",")], sys.argv[5]
    rows = sorted(tuple(map(int, l.split())) for l in open(mp) if l.strip())
    lab = mkmask.label(rows); Q = np.load(qf)
    rs = np.where(np.random.default_rng(99).random(5120) < 0.5, -1.0, 1.0).astype(np.float32)
    for i in idxs:
        fam, layer = lab[i]; _, n, k, mats = rows[i]
        if fam not in ("attn_in", "gdn_in", "gate_up"): continue
        K, bf, m2 = map(int, open(f"{D}/{i}.meta").read().split())
        raw = np.fromfile(f"{D}/{i}.x", dtype=np.uint16).reshape(-1, K)
        x = ((raw.astype(np.uint32) << 16).view(np.float32) if bf else raw.view(np.float16).astype(np.float32))
        suhs = [np.fromfile(f"{D}/{i}.suh", dtype=np.float16).astype(np.float32)]
        if m2 == 2: suhs.append(np.fromfile(f"{D}/{i}.suh1", dtype=np.float16).astype(np.float32))
        p = f"{P}{layer}."
        cands = ([p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight"] if fam == "gdn_in" else
                 [p + f"self_attn.{t}_proj.weight" for t in ("q", "k", "v")] if fam == "attn_in" else [])
        Ws = A.weight(A.hf_names(fam, layer, n, k, A.shapes_for(cands)))
        assert len(Ws) == len(suhs)
        gm = gain(layer, fam)
        xn = np.divide(x, gm, out=np.zeros_like(x), where=gm != 0)    # gain 0: a dead channel, x is 0 there too
        u = (xn @ Q).astype(np.float32)
        ref = sum(float(((x @ W.T).astype(np.float64) ** 2).sum()) for W in Ws)
        # outlier stats: share of input energy in the top channel, before / after R1
        xe = (x.astype(np.float64) ** 2).sum(0); ue = (u.astype(np.float64) ** 2).sum(0)
        stats = {"top1_share_x": float(xe.max() / xe.sum()), "top1_share_u": float(ue.max() / ue.sum())}
        for var in ("base", "signs", "r1"):
            for bits, g, c in CONFIGS:
                eo = 0.0
                for s, W in zip(suhs, Ws):
                    if var == "r1":
                        r = A.rotate(u, rs); e = A.unrotate(r - A.quant(r, bits, g, c), rs)
                        ex = (e @ Q.T) * gm
                    else:
                        ss = s if var == "base" else np.sign(s).astype(np.float32)
                        r = A.rotate(x, ss); ex = A.unrotate(r - A.quant(r, bits, g, c), ss)
                    eo += float(((ex @ W.T).astype(np.float64) ** 2).sum())
                o = {"i": i, "fam": fam, "layer": layer, "N": n, "var": var, "bits": bits, "group": g, "clip": c,
                     "rel_out": eo / ref, **stats}
                with open(out, "a") as f: f.write(json.dumps(o) + "\n")
        print(i, fam, layer, "done", flush=True)


if __name__ == "__main__":
    main()
