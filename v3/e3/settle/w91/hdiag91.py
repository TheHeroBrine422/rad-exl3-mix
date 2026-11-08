#!/usr/bin/env python3
"""hdiag91.py HESS CENSUS.jsonl OUT.json -- lead step 0 (notes/w91.md, 11-08 1:10 PM CST): is channel 3994 the
dominant residual channel? (a) residual readers: the diagonal of each residual-input Hessian (layers.N.input = GDN
qkv / z or attention q / k / v; layers.N.post_attention.input = gate / up; lm_head.input), per channel share of the
trace: channel 3994's share, every channel > 5%, the top 5. (b) writers: census91.py's output-energy shares
o_c = w_c^T H_in w_c / sum (bf16 checkpoint, calib H_in), row 3994 and the top rows. Prints a per-layer table."""
import json, struct, sys
import numpy as np
from safetensors import safe_open

hess, cen, outp = sys.argv[1:4]


def diag(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    k = int(meta["k"])
    with safe_open(path, "np") as f: up = f.get_tensor("hessian_upper")
    i = np.arange(k, dtype=np.int64); idx = i * k - i * (i - 1) // 2      # row-major packed upper: (i, i)
    return up[idx].astype(np.float64)


C = {json.loads(x)["key"]: json.loads(x) for x in open(cen) if x.strip()}
out = {"readers": {}, "writers": {}}
names = [f"model.language_model.layers.{l}.{s}" for l in range(64) for s in ("input", "post_attention.input")] + ["lm_head.input"]
for nm in names:
    d = diag(f"{hess}/{nm}.safetensors"); s = d / d.sum(); top = np.argsort(-s)[:5]
    out["readers"][nm] = {"share3994": float(s[3994]), "over5pct": [[int(c), float(s[c])] for c in np.nonzero(s > 0.05)[0]],
                          "top5": [[int(c), float(s[c])] for c in top], "mean_over_median": float(d.mean() / np.median(d)),
                          "trace_minus_max_mean_over_mean": float((d.sum() - d.max()) / (len(d) - 1) / d.mean())}
for k, r in sorted(C.items(), key=lambda kv: (kv[1]["layer"], kv[0])):
    out["writers"][k] = {"o3994": r["o3994"], "e3994": r["e3994"], "top_o": r["top_o"][:5]}
json.dump(out, open(outp, "w"), indent=1)
print("layer | reader H diag share of ch 3994: input / post_attn (other ch > 5%) | writer output-energy share of row 3994 "
      "(other rows > 5%)")
for l in range(64):
    a, b = out["readers"][f"model.language_model.layers.{l}.input"], out["readers"][f"model.language_model.layers.{l}.post_attention.input"]
    oth = sorted({c for c, _ in a["over5pct"] + b["over5pct"]} - {3994})
    ws = [(k.split(".")[-1], w) for k, w in out["writers"].items() if f".layers.{l}." in k]
    wo = sorted({c for _, w in ws for c, v in w["top_o"] if v > 0.05} - {3994})
    print(f"L{l:2d} | {a['share3994']:.3f} / {b['share3994']:.3f} {oth if oth else ''} (mean/median diag {a['mean_over_median']:.1f} / "
          f"{b['mean_over_median']:.1f}) | " + ", ".join(f"{n} {w['o3994']:.3f}" for n, w in ws) + (f" {wo}" if wo else ""))
h = out["readers"]["lm_head.input"]
print(f"lm_head.input: share3994 {h['share3994']:.3f}, top5 {h['top5']}")
