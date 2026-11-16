#!/usr/bin/env python3
"""bf16cmp.py CKPT_DIR TENSOR OURS.bf16 [--chunk N] -- #197 (w185): compare a dequantised bf16 dump with the true bf16
tensor of the source checkpoint (read straight from its safetensors shard, chunked by rows; numpy only).
Prints one JSON line: relative Frobenius error, mean per-row cosine, scale <d,t>/<t,t>, max |d-t|, nonfinite count.
Run with `python3 -I` (no cwd imports)."""
import json, os, struct, sys
import numpy as np

args = sys.argv[1:]
chunk = 4096
if "--chunk" in args:
    i = args.index("--chunk"); chunk = int(args[i + 1]); del args[i:i + 2]
ckpt, name, ours = args[0], args[1], args[2]
idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
shard = os.path.join(ckpt, idx[name])
with open(shard, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
base = 8 + n
m = hdr[name]
assert m["dtype"] == "BF16", m["dtype"]
R, C = m["shape"]
o0 = m["data_offsets"][0]
dt = np.dtype("<u2")
tf = open(shard, "rb")
tf.seek(base + o0)
of = open(ours, "rb")
sum_dt = sum_tt = sum_dd = sum_err2 = 0.0
cos_acc = 0.0; rel_row_acc = 0.0; maxabs = 0.0; nonfin = 0
r = 0
while r < R:
    k = min(chunk, R - r)
    t = np.frombuffer(tf.read(k * C * 2), dtype=dt).reshape(k, C).astype(np.uint32) << 16
    t = t.view(np.float32).astype(np.float64)
    d = np.frombuffer(of.read(k * C * 2), dtype=dt).reshape(k, C).astype(np.uint32) << 16
    d = d.view(np.float32).astype(np.float64)
    nonfin += int((~np.isfinite(d)).sum())
    e = d - t
    sum_dt += float((d * t).sum()); sum_tt += float((t * t).sum()); sum_dd += float((d * d).sum())
    sum_err2 += float((e * e).sum()); maxabs = max(maxabs, float(np.abs(e).max()))
    tn = np.sqrt((t * t).sum(1)); dn = np.sqrt((d * d).sum(1)); en = np.sqrt((e * e).sum(1))
    cos_acc += float(((d * t).sum(1) / np.maximum(dn * tn, 1e-30)).sum())
    rel_row_acc += float((en / np.maximum(tn, 1e-30)).sum())
    r += k
out = {"tensor": name, "shape": [R, C], "rel_fro": (sum_err2 ** 0.5) / (sum_tt ** 0.5), "cos_row_mean": cos_acc / R,
       "rel_row_mean": rel_row_acc / R, "scale_dt_tt": sum_dt / sum_tt, "rms_true": (sum_tt / (R * C)) ** 0.5,
       "rms_ours": (sum_dd / (R * C)) ** 0.5, "max_abs_err": maxabs, "nonfinite": nonfin}
print(json.dumps(out))
