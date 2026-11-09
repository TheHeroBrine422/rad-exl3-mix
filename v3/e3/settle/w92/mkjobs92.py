#!/usr/bin/env python3
"""mkjobs92.py LEVEL SEED CKPT HESS HOUT OUTDIR [--layers A-B] -- the rxq job specs of one ladder level (../w92.md, #122),
as the donors get them: rxq.ladder.level_jobs (the settings file's split, group = one Hessian group of a layer), each
job's params turned into the spec rad-quant-at-home's rxq_quant.stage() writes for `python -m rxq job`, with gpubox's
inputs in place of the datapack's staged links (the bf16 checkpoint dir, the published calib H_in dir, the H_out dir).
LEVEL = "2.5-i8" | "1-i4" | ... (K-codebook, as rxq.ladder jobs --levels). Writes OUTDIR/<LNN-gG>.json per job and
OUTDIR/jobs.jsonl (the rqah params, for the record: req / est_s, c9, hout). Run with the staged pipeline/ on PYTHONPATH."""
import argparse, json, os, sys

from rxq import config as C, ladder

ap = argparse.ArgumentParser()
ap.add_argument("level"); ap.add_argument("seed", type=int); ap.add_argument("ckpt"); ap.add_argument("hess"); ap.add_argument("hout")
ap.add_argument("out"); ap.add_argument("--layers"); ap.add_argument("--config")
a = ap.parse_args()
cfgpath = a.config or os.path.join(os.path.dirname(os.path.abspath(C.__file__)), "..", "config", "v3.toml")
cfgpath = os.path.normpath(cfgpath)
cfg = C.load(cfgpath)
lvs = [v for v in ladder.levels(cfg) if f"{ladder.fmt_k(v['K'])}-{v['codebook']}" == a.level]
if len(lvs) != 1: sys.exit(f"level {a.level}: {len(lvs)} matches in the ladder")
lv = lvs[0]
jobs = ladder.level_jobs(cfg, lv, "q27", "w92", None, seed_offset=a.seed, layers=a.layers)
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "jobs.jsonl"), "w") as fh:
    for jb in jobs:
        p = jb["params"]
        # = rqah jobs/rxq_quant.py stage(): the spec `python -m rxq job` reads
        spec = {"type": "rxq.quant", "version": 1, "level": p["level"], "K": int(p["K"]) if float(p["K"]).is_integer() else p["K"],
                "codebook": p["codebook"], "seed_offset": p["settings"]["seed_offset"], "keys": list(p["keys"]),
                "settings_stamp": p["settings"]["rxq"], "config": cfgpath,
                "inputs": {"ckpt": a.ckpt, "hessians": a.hess, "hin": None, "wpre": None, "hout": a.hout if p["hout"] else None}}
        name = f"L{p['layer']:02d}-g{p['group']}"
        json.dump(spec, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
        fh.write(json.dumps({"name": name, **p}) + "\n")
est = sum(j["params"]["req"]["est_s"] for j in jobs)
print(f"{lv['name']} seed {a.seed}: {len(jobs)} jobs -> {a.out} (settings {C.settings_stamp(cfg)}, est {est / 3600:.2f} R9700-h, "
      f"hout {jobs[0]['params']['hout']}, c9 writers {sum(len(j['params']['c9']) for j in jobs)})")
