# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""python -m rxq quant --keys K1,K2|--layers A-B -K K --codebook CB --ckpt DIR --hess DIR -o OUT [--hout DIR] [--hin DIR]
                       [--wpre DIR] [--seed-offset N] [--config FILE] [--device N] [--save-wq] [--no-prof]
   python -m rxq job SPEC.json -o OUT [--device N]
   python -m rxq assemble OUT JOBDIR... [--template config.json] [--layers N] [--head]
   python -m rxq stamp [--config FILE]
   python -m rxq plan -K K [--codebook CB] [--split tensor|group|layer] [--layers N] [--seed-offset N] [--config FILE]
VRAM_CAP_GB caps the process's CUDA allocator (shared cards)."""
import argparse, json, os, sys

import torch


def _cap(device):
    cap = os.environ.get("VRAM_CAP_GB")
    if cap and torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 2**30 / tot), device)


def main(argv=None):
    from . import config as C, job, qwen35
    ap = argparse.ArgumentParser(prog="rxq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quant")
    q.add_argument("--keys"); q.add_argument("--layers"); q.add_argument("-K", required=True, help="level, or a comma list (one process; -o must contain {K})")
    q.add_argument("--codebook", default=None); q.add_argument("--ckpt", required=True); q.add_argument("--hess", required=True)
    q.add_argument("-o", required=True); q.add_argument("--hout"); q.add_argument("--hin"); q.add_argument("--wpre")
    q.add_argument("--seed-offset", type=int, default=0); q.add_argument("--config"); q.add_argument("--device", type=int, default=0)
    q.add_argument("--save-wq", action="store_true"); q.add_argument("--no-prof", action="store_true")
    q.add_argument("--set", action="append", default=[], help="settings override PATH=JSON, e.g. inputs.hout.enable=true")
    j = sub.add_parser("job"); j.add_argument("spec"); j.add_argument("-o", required=True); j.add_argument("--device", type=int, default=0)
    s = sub.add_parser("assemble"); s.add_argument("out"); s.add_argument("jobs", nargs="+"); s.add_argument("--template")
    s.add_argument("--layers", type=int, default=64); s.add_argument("--head", action="store_true")
    t = sub.add_parser("stamp"); t.add_argument("--config")
    pl = sub.add_parser("plan"); pl.add_argument("-K", required=True, type=float); pl.add_argument("--codebook")
    pl.add_argument("--split", choices=("tensor", "group", "layer"), default="tensor"); pl.add_argument("--layers", type=int, default=64)
    pl.add_argument("--seed-offset", type=int, default=0); pl.add_argument("--config")
    a = ap.parse_args(argv)
    if a.cmd == "stamp":
        print(C.settings_stamp(C.load(a.config))); return
    if a.cmd == "plan":
        for sp in job.plan_level(C.load(a.config), a.K, a.codebook, a.split, a.layers, a.seed_offset): print(json.dumps(sp))
        return
    if a.cmd == "assemble":
        print(job.assemble(a.jobs, a.out, a.template, a.layers, a.head)); return
    _cap(a.device)
    if a.cmd == "job":
        spec = json.load(open(a.spec)); print(json.dumps(job.run_quant(spec, a.o, a.device))); return
    cfg = C.load(a.config)
    if a.set: cfg = C.with_overrides(cfg, **{p.split("=", 1)[0]: json.loads(p.split("=", 1)[1]) for p in a.set})
    if a.keys: keys = a.keys.split(",")
    else:
        lo, hi = map(int, a.layers.split("-")); keys = [k for l in range(lo, hi + 1) for _, ks in qwen35.layer_plan(l) for k in ks]
    Ks = [int(float(v)) if float(v).is_integer() else float(v) for v in a.K.split(",")]
    assert len(Ks) == 1 or "{K}" in a.o, "several levels: -o needs {K}"
    for K in Ks:
        spec = {"type": "rxq.quant", "version": 1, "level": f"K{K}", "K": K, "codebook": a.codebook or C.codebook_for(cfg, K),
                "seed_offset": a.seed_offset, "keys": keys, "settings_stamp": C.settings_stamp(cfg),
                "inputs": {"ckpt": a.ckpt, "hessians": a.hess, "hin": a.hin, "wpre": a.wpre, "hout": a.hout}}
        print(json.dumps(job.run_quant(spec, a.o.replace("{K}", str(K)), a.device, cfg=cfg, profile=not a.no_prof,
                                       save_wq=a.save_wq)), flush=True)


if __name__ == "__main__":
    main()
