#!/usr/bin/env python3
"""swjapply.py LEVEL SCALES DST -- the container form of a fixed-code scale refit: DST = LEVEL (an ourq.py export) with
suh *= r and svh *= c of every tensor that has SCALES/scales/<key>.safetensors (swjfit27.py's output: gate / up, r = 1),
rounded to f16 as stored. Codes, rotations and every other tensor are LEVEL's, byte for byte. The ourq stamp metadata
of each shard is kept and "swj27" added (source, scales, sha of the scales); stamp.json, the index and config.json are
copied, so `exl3i src=DST codebook=...` imports DST exactly as it imports LEVEL (its stamp check included).
Restartable: shards are written .part and renamed, finished ones skipped; DST/.done last. Peak RAM: one shard."""
import hashlib, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, sc, dst = sys.argv[1:4]
if os.path.exists(f"{dst}/.done"): print(f"{dst} done"); sys.exit(0)
os.makedirs(dst, exist_ok=True)
sd = f"{sc}/scales"
keys = sorted(f[:-len(".safetensors")] for f in os.listdir(sd) if f.endswith(".safetensors"))
h = hashlib.sha256()
for k in keys: h.update(k.encode()); h.update(open(f"{sd}/{k}.safetensors", "rb").read())
meta = json.dumps({"src": os.path.abspath(src), "scales": os.path.abspath(sc), "scales_sha256": h.hexdigest()[:16]})
for f in ("config.json", "model.safetensors.index.json", "stamp.json"):
    if os.path.exists(f"{src}/{f}"): shutil.copy(f"{src}/{f}", f"{dst}/{f}")
used, edited = set(), 0
for f in sorted(os.listdir(src)):
    if not (f.startswith("q-") and f.endswith(".safetensors")): continue
    with safe_open(f"{src}/{f}", "pt", device="cpu") as fh:
        t = {k: fh.get_tensor(k) for k in fh.keys()}; md = dict(fh.metadata() or {})
    for k in [k for k in t if k[:-4] in keys and k[-4:] in (".suh", ".svh")]:
        with safe_open(f"{sd}/{k[:-4]}.safetensors", "pt") as s: mult = s.get_tensor("r" if k.endswith(".suh") else "c")
        assert mult.shape == t[k].shape and t[k].dtype == torch.float16, (k, tuple(mult.shape), tuple(t[k].shape), t[k].dtype)
        if not bool((mult == 1).all()): t[k] = (t[k].float() * mult.float()).half(); edited += 1
        used.add(k[:-4])
    if os.path.exists(f"{dst}/{f}"): continue        # done earlier (the scan above only counts)
    md["swj27"] = meta
    save_file(t, f"{dst}/{f}.part", metadata=md); os.replace(f"{dst}/{f}.part", f"{dst}/{f}")
miss = sorted(set(keys) - used)
if miss: sys.exit(f"{len(miss)} scales without a tensor in {src} (first {miss[0]}): {dst} left without .done")
open(f"{dst}/.done", "w").close()
print(f"SWJAPPLY DONE {len(used)} tensors ({edited} scale vectors changed) -> {dst}")
