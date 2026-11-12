#!/usr/bin/env python3
"""c9e3.py CUR E3 INJOB OUT -- #145 amendment 1 (../w110.md): the e3 arm's C9 scales = INJOB's c (the level's in-job
LEVEL/c9, what T ships), with the rows where the CPU e_clip 0.3 run (E3) differs from the CPU e_clip 0.5 run (CUR)
replaced by E3's values. OUT/scales/<key>.safetensors {"c", "r"} (r = INJOB's), OUT/e3rows.jsonl, OUT/.done last.
Prints per family the rows changed and the c change; last line "E3 DONE {json}"."""
import json, os, sys
import torch
from safetensors.torch import load_file, save_file

cur, e3, inj, out = sys.argv[1:5]
if os.path.exists(f"{out}/.done"): print(f"{out} done"); sys.exit(0)
os.makedirs(f"{out}/scales", exist_ok=True)
fs = sorted(f for f in os.listdir(f"{inj}/scales") if f.endswith(".safetensors"))
st = {"writers": len(fs), "rows": 0, "rows_changed": 0, "writers_changed": 0, "max_rel": 0.0}
with open(f"{out}/e3rows.jsonl", "w") as log:
    for f in fs:
        a, b, j = load_file(f"{cur}/scales/{f}"), load_file(f"{e3}/scales/{f}"), load_file(f"{inj}/scales/{f}")
        m = a["c"] != b["c"]
        c = torch.where(m, b["c"], j["c"]).contiguous()
        save_file({"c": c, "r": j["r"].contiguous()}, f"{out}/scales/{f}.part"); os.replace(f"{out}/scales/{f}.part", f"{out}/scales/{f}")
        n = int(m.sum()); rel = float(((c.double() - j["c"].double()).abs() / j["c"].double().abs()).max())
        st["rows"] += int(m.numel()); st["rows_changed"] += n; st["writers_changed"] += int(n > 0); st["max_rel"] = max(st["max_rel"], rel)
        log.write(json.dumps({"key": f[:-12], "rows": int(m.numel()), "rows_changed": n, "max_rel": rel,
                              "c_mean_t": float(j["c"].double().mean()), "c_mean_e3": float(c.double().mean())}) + "\n")
open(f"{out}/.done", "w").close()
print("E3 DONE " + json.dumps(st))
