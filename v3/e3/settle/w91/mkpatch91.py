#!/usr/bin/env python3
"""mkpatch91.py LEVEL CKPT KEYS ROWS OUT -- ../w91.md item 1: the fixed-code oracle's two checkpoint dirs.
KEYS: comma-separated checkpoint keys of the writers to store dense; ROWS: comma-separated output rows to restore.
  OUT/ck-C: the bf16 checkpoint with every KEYS tensor replaced by bf16(What) (the level's own codes, decoded with
            rxq/deq.py in f32, rounded once to bf16)  -> the dense control
  OUT/ck-A: the same, with rows ROWS of every KEYS tensor = the checkpoint's bf16 rows -> the arm
Each dir: symlinks to every file of CKPT except the index, a patch shard w91-patch.safetensors, and an index whose
weight_map points KEYS at the shard (rad-convert reads only the shard the index names for a tensor). Also writes
OUT/keys.txt (the --changed list) and OUT/patch.json (rows, keys, per-tensor stats). Idempotent (skips if done)."""
import json, os, sys

import torch
from safetensors.torch import save_file

from rxq import deq
from rxq.quant import Ckpt

level, ckpt, keys, rows, out = sys.argv[1:6]
keys = [k for k in keys.split(",") if k]; rows = sorted(int(r) for r in rows.split(",") if r)
if os.path.exists(f"{out}/patch.json"): print("MKPATCH91 already done"); sys.exit(0)
os.makedirs(out, exist_ok=True)
lev, ck = deq.Level(level, "cpu"), Ckpt(ckpt)
C, A, stats = {}, {}, {}
for key in keys:
    W = ck.get(key + ".weight")
    assert W.dtype == torch.bfloat16, (key, W.dtype)
    Wh = lev.get(key).to(torch.bfloat16).contiguous()
    assert Wh.shape == W.shape, (key, Wh.shape, W.shape)
    Wa = Wh.clone(); Wa[rows] = W[rows]
    C[key + ".weight"], A[key + ".weight"] = Wh, Wa
    d = (Wa.view(torch.int16) != Wh.view(torch.int16)).any(1).nonzero().flatten().tolist()
    assert set(d) <= set(rows), (key, d[:10])
    e = (W.float() - Wh.float())
    stats[key] = {"shape": list(W.shape), "rows_changed": d, "rel_mse": float((e ** 2).sum() / (W.float() ** 2).sum()),
                  "rel_mse_rows": [float((e[r] ** 2).sum() / (W[r].float() ** 2).sum()) for r in rows],
                  "rel_mse_after": float(((W.float() - Wa.float()) ** 2).sum() / (W.float() ** 2).sum())}
    print(key, stats[key]["rel_mse"], "->", stats[key]["rel_mse_after"], flush=True)
ix = json.load(open(f"{ckpt}/model.safetensors.index.json"))
for tag, T in (("C", C), ("A", A)):
    d = f"{out}/ck-{tag}"; os.makedirs(d, exist_ok=True)
    for f in os.listdir(ckpt):
        if f == "model.safetensors.index.json": continue
        if not os.path.lexists(f"{d}/{f}"): os.symlink(os.path.join(ckpt, f), f"{d}/{f}")
    save_file(T, f"{d}/w91-patch.safetensors.part", metadata={"format": "pt"})
    os.replace(f"{d}/w91-patch.safetensors.part", f"{d}/w91-patch.safetensors")
    j = json.loads(json.dumps(ix))
    for k in T: assert k in j["weight_map"], k; j["weight_map"][k] = "w91-patch.safetensors"
    json.dump(j, open(f"{d}/model.safetensors.index.json", "w"), indent=1)
open(f"{out}/keys.txt", "w").write("".join(k + ".weight\n" for k in keys))
json.dump({"level": level, "ckpt": ckpt, "rows": rows, "keys": keys, "stats": stats}, open(f"{out}/patch.json.part", "w"), indent=1)
os.replace(f"{out}/patch.json.part", f"{out}/patch.json")
print("MKPATCH91 DONE")
