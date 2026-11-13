#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkfitin.py DRIVER_FIT.in.json MENU OUT.in.json [--add OBS.jsonl]... [--opt k=v]... -- a fit.py input from the live
search's latest fit input (its obs = search rows + prior obs) with another menu (embmenu.py's) and extra rows
(calobs.py's), for an offline refit (w135, #174). No x0 (the layout changes with new embedding levels)."""
import argparse, json, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mkfitin.py")
    ap.add_argument("inp"); ap.add_argument("menu"); ap.add_argument("out")
    ap.add_argument("--add", action="append", default=[]); ap.add_argument("--opt", action="append", default=[])
    a = ap.parse_args(argv)
    d = json.load(open(a.inp)); obs = d["obs"]
    for p in a.add: obs = obs + [json.loads(l) for l in open(p)]
    opts = dict(d.get("opts") or {})
    for kv in a.opt:
        k, v = kv.split("=", 1); opts[k] = json.loads(v)
    json.dump({"menu": a.menu, "obs": obs, "opts": opts, "ndraw": 0, "seed": 0}, open(a.out, "w"))
    print(f"{a.out}: {len(obs)} rows, opts {opts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
