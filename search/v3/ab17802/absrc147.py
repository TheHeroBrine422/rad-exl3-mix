#!/usr/bin/env python3
"""absrc147.py JOBDIR LEVEL_FINAL OUT -- (w147, #178-02) w142's trellis a | b (rxq's job output JOBDIR = /data/q27/w142/tq/K4:
t-<key>.safetensors per GDN layer, key ...linear_attn.in_proj_ab) as an exl3quant source: OUT/model.safetensors with the 48
layers' .trellis / .suh / .svh / .mul1 tensors as stored (bytes unchanged), config.json from a level's final/ (the importer
reads quant_method and num_hidden_layers), stamp.json naming the job's K and codebook (exl3quant checks codebook= against
it). Prints the K and the tensor count."""
import json, os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

job, lvl, out = sys.argv[1:4]
spec = json.load(open(f"{job}/job.json"))["spec"]
os.makedirs(out, exist_ok=True)
t = {}
for key in spec["keys"]:
    with safe_open(f"{job}/t-{key}.safetensors", "pt") as f:
        for k in f.keys(): t[k] = f.get_tensor(k).contiguous()
    assert t[f"{key}.trellis"].shape == (320, 8, int(16 * spec["K"])), (key, t[f"{key}.trellis"].shape)
save_file(t, f"{out}/model.safetensors.part", metadata={"format": "pt"})
os.replace(f"{out}/model.safetensors.part", f"{out}/model.safetensors")
shutil.copy(f"{lvl}/config.json", f"{out}/config.json")
json.dump([{"stamp": "w142-ab178-K%s" % spec["K"], "K": str(spec["K"]), "codebook": spec["codebook"],
            "args": {"K": spec["K"], "codebook": spec["codebook"]}, "source": job}], open(f"{out}/stamp.json", "w"), indent=1)
print(f"K {spec['K']} codebook {spec['codebook']}: {len(spec['keys'])} keys, {len(t)} tensors -> {out}")
