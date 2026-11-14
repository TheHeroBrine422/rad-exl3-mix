#!/usr/bin/env python3
"""mkconf.py OUT --root R --cards 0,1,... [--kinds quant[,swj]] [--steps jobs,assemble,post,recipe] [--levels a,b]
[--layers A-B] [--dup F] [--workers N] [--swj-batch N] -- an rxrun.run conf for #177-03 (w151) from the host's ladder conf
(/data/q27/multi/run.json: env, inputs, model, swj_meta, locks), with the staged tree's settings file. Levels as rxrun
takes them (names or K-codebook)."""
import argparse, copy, json

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--root", required=True); ap.add_argument("--cards", required=True)
ap.add_argument("--kinds", default="quant"); ap.add_argument("--steps", default="jobs,assemble,post,recipe")
ap.add_argument("--levels"); ap.add_argument("--layers"); ap.add_argument("--dup", type=float, default=0.015)
ap.add_argument("--workers", type=int, default=3); ap.add_argument("--swj-batch", type=int, default=2)
ap.add_argument("--base", default="/data/q27/multi/run.json")
a = ap.parse_args()
b = json.load(open(a.base))
c = {k: copy.deepcopy(b[k]) for k in ("python", "env", "inputs", "model", "swj_meta", "dup_seed_delta", "chunk", "mode", "ram_lock")}
c.update(root=a.root, settings="/data/q27/stage/code/pipeline/config/v3.toml", passes=["1", "2", "3"], steps=a.steps.split(","),
         dup_frac=a.dup, swj_batch=a.swj_batch)
if a.levels: c["levels"] = a.levels.split(",")
if a.layers: c["layers"] = a.layers
d0 = b["devices"][0]
c["devices"] = [dict(copy.deepcopy(d0), name=f"c{i}", device=int(i), lock=f"/data/q27/locks/card{i}.lock", workers=a.workers,
                     kinds=a.kinds.split(","), env={"ROCR_VISIBLE_DEVICES": str(i)} if "swj" in a.kinds else d0.get("env", {}))
                for i in a.cards.split(",")]
for d in c["devices"]:
    if not d["env"]: d.pop("env")
json.dump(c, open(a.out, "w"), indent=1)
print(a.out, c["root"], [d["name"] for d in c["devices"]], c["steps"], c.get("levels", "all"))
