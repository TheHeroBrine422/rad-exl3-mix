#!/usr/bin/env python3
"""houtstruct.py SNAPDIR HCALIB OUT.jsonl -- structure and noise of the YAQA Sketch B factors, per linear (CPU).

SNAPDIR holds snapshots snap1024 .. snap8192 (proxy2b/yaqaB.py), each the MEAN over its first n sequences. Per linear, with H = H(8192) normalised
to unit trace:
- spectrum of H_out: participation ratio PR = tr(H)^2 / ||H||_F^2 and entropy rank exp(-sum p log p) (both as a
  fraction of the dimension d), energy of the top 1% / 10% of eigenvalues, condition number of H + 0.01 mean(diag) I
  (the paper's damping);
- diagonal: share of ||H||_F^2 on the diagonal, coefficient of variation of the diagonal (row-importance spread);
- noise: the halves A = H(4096) and B = 2 H(8192) - H(4096) are independent (disjoint sequences); Frobenius cosine
  cos(A, B), cosine of their diagonals, and the same for H_in. If cos(A, B) is well below 1, H_out is noise-limited;
- convergence: cos(H(n), H(8192)) for n = 1024, 2048, 4096;
- H_in vs the plain calib input Hessian: cos(H_in(8192), Hc) (Sketch B's H_in is gradient-weighted).
One JSON line per linear, then a summary by type is printed. Read-only on the snapshots."""
import glob, json, os, struct, sys
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
SD, HC, OUT = sys.argv[1:4]
SN = [1024, 2048, 4096, 8192]
L = "model.language_model.layers"
GRP = {"linear_attn.in_proj_qkv": "input", "linear_attn.in_proj_z": "input", "self_attn.q_proj": "input",
       "self_attn.k_proj": "input", "self_attn.v_proj": "input", "linear_attn.out_proj": "linear_attn.out_proj.input",
       "self_attn.o_proj": "self_attn.o_proj.input", "mlp.gate_proj": "post_attention.input",
       "mlp.up_proj": "post_attention.input", "mlp.down_proj": "mlp.down_proj.input"}


def upper(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; k = int(json.loads(fh.read(n))["__metadata__"]["k"])
    with safe_open(p, "pt") as f: u = f.get_tensor("hessian_upper").double()
    H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k); H[iu[0], iu[1]] = u
    return H + H.triu(1).T


def hout(n, key):
    with safe_open(f"{SD}/snap{n}/hout/{key}.safetensors", "pt") as f: H = f.get_tensor("hout").double()
    return (H + H.T) / 2


cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
unit = lambda H: H / torch.diagonal(H).sum()


def spec(H):
    d = H.shape[0]; H = unit(H); ev = torch.linalg.eigvalsh(H).clamp(min=0).flip(0)
    p = ev / ev.sum(); fro2 = float((H * H).sum()); dg = torch.diagonal(H)
    ent = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
    damp = ev + 0.01 * float(dg.mean())
    return {"d": d, "pr": 1.0 / fro2 / d, "erank": ent / d,
            "top1pct": float(ev[:max(1, d // 100)].sum() / ev.sum()), "top10pct": float(ev[:max(1, d // 10)].sum() / ev.sum()),
            "cond_damped": float(damp[0] / damp[-1]), "diag_share": float((dg * dg).sum()) / fro2,
            "diag_cv": float(dg.std() / dg.mean())}


keys = sorted(os.path.basename(f)[:-12] for f in glob.glob(f"{SD}/snap8192/hout/*.safetensors"))
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["key"])
with open(OUT, "a") as fo:
    for key in keys:
        if key in done: continue
        lay, typ = int(key.split(".")[3]), key.split(f"{L}.")[1].split(".", 1)[1]
        Ho = {n: hout(n, key) for n in SN}
        A, B = Ho[4096], 2 * Ho[8192] - Ho[4096]          # snapshots are means: second half = 2 H(8192) - H(4096)
        Hi8, Hi4 = upper(f"{SD}/snap8192/hin/{key}.safetensors"), upper(f"{SD}/snap4096/hin/{key}.safetensors")
        Hc = upper(f"{HC}/{L}.{lay}.{GRP[typ]}.safetensors")
        r = {"key": key, "layer": lay, "type": typ, **spec(Ho[8192]),
             "halves_cos": cos(A, B), "halves_diag_cos": cos(torch.diagonal(A), torch.diagonal(B)),
             "conv_cos": {n: cos(Ho[n], Ho[8192]) for n in SN[:3]},
             "hin_halves_cos": cos(Hi4, 2 * Hi8 - Hi4), "hin_vs_calib_cos": cos(Hi8, Hc),
             "hin_vs_calib_diag_cos": cos(torch.diagonal(Hi8), torch.diagonal(Hc))}
        fo.write(json.dumps(r) + "\n"); fo.flush()
        print(f"{key.split('layers.')[1]:34s} d {r['d']:5d} pr {r['pr']:.3f} erank {r['erank']:.3f} diag {r['diag_share']:.3f} "
              f"halves {r['halves_cos']:.3f} hin~calib {r['hin_vs_calib_cos']:.3f}", flush=True)

rows = [json.loads(l) for l in open(OUT)]
print("\nby type (median):")
cols = ["d", "pr", "erank", "top10pct", "diag_share", "diag_cv", "halves_cos", "halves_diag_cos", "hin_halves_cos", "hin_vs_calib_cos"]
print("type".ljust(26) + "".join(c[:12].rjust(13) for c in cols))
for t in sorted({r["type"] for r in rows}):
    g = [r for r in rows if r["type"] == t]
    print(f"{t} ({len(g)})".ljust(26) + "".join(f"{np.median([r[c] for r in g]):13.4g}" for c in cols))
