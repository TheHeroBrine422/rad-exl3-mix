#!/usr/bin/env python3
"""mkconf115.py TEMPLATE_CONF SETTINGS ROOT SEED OUT_CONF -- #152-02 / #153-02 (../w115.md): an rxrun conf for one K2.5 i8
level (arm settings file, seed offset SEED) under ROOT, from w114's lv114-B.json with w115's paths, and the level's
specs pre-made with that seed offset (rxrun.run makes them with seed offset 0 and skips a specs dir that exists)."""
import json, os, sys

from rxq import ladder
from rxrun import run as RR

tpl, settings, root, seed, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
c = json.load(open(tpl))
c.update(root=root, settings=settings, levels=["2.5-i8"], swj_batch=1, dup_frac=0.0)
c["env"] = {k: (v.replace("/data/w114/", "/data/w115/") if isinstance(v, str) else v) for k, v in c["env"].items()}
assert all("/data/w114" not in str(v) for v in c["env"].values()), c["env"]
os.makedirs(root, exist_ok=True)
json.dump(c, open(out + ".part", "w"), indent=1); os.replace(out + ".part", out)
orig = ladder.level_jobs
ladder.level_jobs = lambda *a, **k: orig(*a, **dict(k, seed_offset=seed))
r = RR.Run(c)
assert len(r.levels) == 1, [v["name"] for v in r.levels]
d = r.specs(r.levels[0])
so = {json.load(open(f"{d}/{f}"))["seed_offset"] for f in os.listdir(d) if f.endswith(".json") and f != "jobs.jsonl"}
assert so == {seed}, so
print(f"{r.levels[0]['name']} settings {r.st} seed {seed}: specs {d} ({sum(1 for _ in open(d + '/jobs.jsonl'))} jobs)")
