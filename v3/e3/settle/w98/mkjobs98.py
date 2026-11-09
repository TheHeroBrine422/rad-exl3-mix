#!/usr/bin/env python3
"""mkjobs98.py K CODEBOOK SEED CKPT HESS HOUT OUTDIR --layers L1,L2 [--config FILE] -- the rxq job specs of one level's
layers (../w98.md, #138-01), as w92's mkjobs92.py makes them (rxq.ladder.level_jobs, group split, each job's params
turned into the spec rad-quant-at-home's rxq_quant.stage() writes), for any (K, codebook): the level comes from
rxq.ladder.levels when the settings file lists it (same name as w92's), else it is named the same way (K0.5 i8: the
one-codebook decision #126, not yet in the HEAD settings file). Writes OUTDIR/<LNN-gG>.json and OUTDIR/jobs.jsonl."""
import argparse, json, os

from rxq import config as C, ladder

ap = argparse.ArgumentParser()
ap.add_argument("K", type=float); ap.add_argument("codebook"); ap.add_argument("seed", type=int)
ap.add_argument("ckpt"); ap.add_argument("hess"); ap.add_argument("hout"); ap.add_argument("out")
ap.add_argument("--layers", required=True); ap.add_argument("--config")
a = ap.parse_args()
cfgpath = os.path.normpath(a.config or os.path.join(os.path.dirname(os.path.abspath(C.__file__)), "..", "config", "v3.toml"))
cfg = C.load(cfgpath)
K = int(a.K) if a.K.is_integer() else a.K
lvs = [v for v in ladder.levels(cfg) if float(v["K"]) == float(K) and v["codebook"] == a.codebook]
lv = lvs[0] if lvs else {"name": f"{cfg['ladder']['prefix']}-{C.settings_stamp(cfg)[:8]}-k{ladder.fmt_k(K)}-{a.codebook}",
                         "K": K, "codebook": a.codebook, "priority": 0}
jobs = ladder.level_jobs(cfg, lv, "q27", "w98", None, seed_offset=a.seed, layers=a.layers)
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "jobs.jsonl"), "w") as fh:
    for jb in jobs:
        p = jb["params"]
        spec = {"type": "rxq.quant", "version": 1, "level": p["level"], "K": int(p["K"]) if float(p["K"]).is_integer() else p["K"],
                "codebook": p["codebook"], "seed_offset": p["settings"]["seed_offset"], "keys": list(p["keys"]),
                "settings_stamp": p["settings"]["rxq"], "config": cfgpath,
                "inputs": {"ckpt": a.ckpt, "hessians": a.hess, "hin": None, "wpre": None, "hout": a.hout if p["hout"] else None}}
        name = f"L{p['layer']:02d}-g{p['group']}"
        json.dump(spec, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
        fh.write(json.dumps({"name": name, **p}) + "\n")
print(f"{lv['name']} seed {a.seed} layers {a.layers}: {len(jobs)} jobs -> {a.out} (settings {C.settings_stamp(cfg)}, "
      f"est {sum(j['params']['req']['est_s'] for j in jobs) / 60:.1f} R9700-min, hout {jobs[0]['params']['hout']})")
