#!/usr/bin/env python3
"""mksplice111.py UNIT CKPT PERMJSON OUT NKEYS -- w111 #146-02 (../w111.md; = w101 mksplice101.py with the key count NKEYS and its own patch name): one arm's dense-splice checkpoint dir.
UNIT = an rxq quant output dir (UNIT/wq/<key>.safetensors "weight", fp16 [out, in]); its arm is N or P by the dir name
(N-... / P-...). P rows are un-permuted with PERMJSON (stored row j = natural row perm[j]), then every Wq is rounded once
to bf16. OUT: symlinks to every file of CKPT except the index, w111-patch.safetensors (the dense tensors), an index whose
weight_map points them at the shard, keys.txt (rad-convert --changed), patch.json (keys, per-tensor rel MSE vs bf16 W).
Idempotent (skips if OUT/patch.json exists)."""
import glob, json, os, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w101"))
from perm101 import inverse  # noqa: E402

unit, ckpt, pj, out, nkeys = sys.argv[1:6]
if os.path.exists(f"{out}/patch.json"): print("MKSPLICE111 already done"); sys.exit(0)
arm = os.path.basename(unit.rstrip("/")).split("-")[0]; assert arm in ("N", "P"), unit
PERM = json.load(open(pj)) if arm == "P" else {}
ix = json.load(open(f"{ckpt}/model.safetensors.index.json")); wmap = ix["weight_map"]
T, stats = {}, {}
for wf in sorted(glob.glob(f"{unit}/wq/*.safetensors")):
    key = os.path.basename(wf)[:-len(".safetensors")]
    with safe_open(wf, "pt") as f: wq = f.get_tensor("weight")
    with safe_open(f"{ckpt}/{wmap[key + '.weight']}", "pt") as f: W = f.get_tensor(key + ".weight")
    assert wq.shape == W.shape and W.dtype == torch.bfloat16, (key, wq.shape, W.shape, W.dtype)
    if arm == "P": wq = wq[inverse(torch.tensor(PERM[key]))]
    wb = wq.to(torch.bfloat16).contiguous()
    stats[key] = {"shape": list(W.shape), "rel_mse": float(((W.float() - wb.float()) ** 2).sum() / (W.float() ** 2).sum())}
    T[key + ".weight"] = wb
    print(arm, key, f"rel_mse {stats[key]['rel_mse']:.4e}", flush=True)
assert len(T) == int(nkeys), len(T)
os.makedirs(out, exist_ok=True)
for f in os.listdir(ckpt):
    if f == "model.safetensors.index.json": continue
    if not os.path.lexists(f"{out}/{f}"): os.symlink(os.path.join(ckpt, f), f"{out}/{f}")
save_file(T, f"{out}/w111-patch.safetensors.part", metadata={"format": "pt", "w111": arm, "unit": unit})
os.replace(f"{out}/w111-patch.safetensors.part", f"{out}/w111-patch.safetensors")
j = json.loads(json.dumps(ix))
for k in T: assert k in j["weight_map"], k; j["weight_map"][k] = "w111-patch.safetensors"
json.dump(j, open(f"{out}/model.safetensors.index.json", "w"), indent=1)
open(f"{out}/keys.txt", "w").write("".join(k + "\n" for k in sorted(T)))
json.dump({"unit": unit, "arm": arm, "keys": sorted(k[:-7] for k in T), "stats": stats}, open(f"{out}/patch.json.part", "w"), indent=1)
os.replace(f"{out}/patch.json.part", f"{out}/patch.json")
print("MKSPLICE111 DONE", flush=True)
