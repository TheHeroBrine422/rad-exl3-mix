# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sumruns.py RUNS [TAG ...] -- #144: one line per dev run (wall, stages, run / load seconds, VRAM peak, s per source)"""
import json, os, sys

R = sys.argv[1]; tags = sys.argv[2:] or sorted(t for t in os.listdir(R) if os.path.exists(f"{R}/{t}/wall.json"))
print(f"{'tag':10s} {'S':>3s} {'wall':>7s} {'stages':>6s} {'run_s':>7s} {'load_s':>6s} {'peakGiB':>7s} {'wall/S':>7s}")
for t in tags:
    if not os.path.exists(f"{R}/{t}/wall.json"): continue
    w = json.load(open(f"{R}/{t}/wall.json")); st = [json.loads(l) for l in open(f"{R}/{t}/scr/timing.jsonl")]
    S = st[0]["sources"]
    wall = w["wall"] if isinstance(w["wall"], (int, float)) else float("nan")
    print(f"{t:10s} {S:3d} {wall:7.1f} {len(st):6d} {sum(x['run_s'] for x in st):7.1f} {sum(x['load_s'] for x in st):6.1f} "
          f"{max(x['peak_gib'] for x in st):7.2f} {wall / S:7.1f}  {w.get('extra', '')}")
