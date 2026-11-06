#!/usr/bin/env python3
"""check_assemble.py OURQ_LEVEL OUT N_LAYERS -- the level assembly (rxq.job.assemble) on CPU: an existing ourq level is
split into per-tensor job files (t-<key>.safetensors with rxq stamps, a few per job dir), assembled back, and every
tensor of the result compared with the original (torch.equal), plus the index's key -> shard map; then a level with
one tensor's stamp changed must be refused. Last line: "ASSEMBLE PASS|FAIL"."""
import json, os, shutil, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rxq import job, qwen35  # noqa: E402

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
shutil.rmtree(out, ignore_errors=True)
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
keys = qwen35.trunk_keys(n)
jobs = [f"{out}/jobs/j{i}" for i in range(5)]
for d in jobs: os.makedirs(d)
for i, k in enumerate(keys):
    names = sorted(x for x in wm if x.startswith(k + ".") and x[len(k) + 1:].count(".") == 0)
    t = {}
    for x in names:
        with safe_open(f"{src}/{wm[x]}", "pt") as f: t[x] = f.get_tensor(x)
    save_file(t, job.tensor_file(jobs[i % 5], k), metadata={"rxq_settings": "s" * 16, "rxq_code": "c" * 16, "K": "3",
                                                            "codebook": "i8", "level": "K3", "key": k})
job.assemble(jobs, f"{out}/level", None, n)
wm2 = json.load(open(f"{out}/level/model.safetensors.index.json"))["weight_map"]
bad = [x for x in wm if wm2.get(x) != wm[x]]
neq = 0
for x in wm:
    with safe_open(f"{src}/{wm[x]}", "pt") as a, safe_open(f"{out}/level/{wm2[x]}", "pt") as b:
        neq += not torch.equal(a.get_tensor(x), b.get_tensor(x))
k0 = keys[3]
with safe_open(job.tensor_file(jobs[3], k0), "pt") as f: t = {x: f.get_tensor(x) for x in f.keys()}; m = dict(f.metadata())
m["rxq_code"] = "d" * 16; save_file(t, job.tensor_file(jobs[3], k0), metadata=m)
try: job.assemble(jobs, f"{out}/level2", None, n); refused = False
except SystemExit as e: refused = "mixed stamps" in str(e)
print(f"{len(wm)} tensors: {neq} differ, {len(bad)} in another shard; mixed stamps refused {refused}")
print(f"ASSEMBLE {'PASS' if not neq and not bad and refused else 'FAIL'}")
