#!/usr/bin/env python3
"""tilecd_gate_gpu.py --keys K1,K2 -K K --ckpt DIR --hess DIR -o OUT [--device N] -- tile CD on the real quantizer (GPU):
rxq quant jobs with tile CD on (no H_out); at every CD call (inside ldlq, on the tensor's real weight, codes and metric)
three CDs run from the same inputs and are compared:
  research   v3/e3/tilecd/tilecd.py's cd (RXQ_RXM tree)         -- rxq/tilecd.py is the merged copy: equal
  rxq        rxq.tilecd.cd(M = None)                           -- what the job keeps
  gate       rxq.tilecd.cd(M = I): the two-sided path with all output weights 1 -- equal to rxq bit for bit
Wq and indices torch.equal, rel records equal. One line per tensor, last line "TILECD GPU GATE PASS|FAIL n/m"."""
import argparse, importlib.util, json, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rxq import config as C, engine, job, tilecd as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--keys", required=True); ap.add_argument("-K", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--hess", required=True); ap.add_argument("-o", required=True); ap.add_argument("--device", type=int, default=0)
a = ap.parse_args()
sp = importlib.util.spec_from_file_location("tilecd_research", os.path.join(engine.rxm_root(), "v3", "e3", "tilecd", "tilecd.py"))
R = importlib.util.module_from_spec(sp); sp.loader.exec_module(R)
orig, out = T.cd, []


def strip(d): return {k: v for k, v in d.items() if k not in ("tile_cd_log",)}


def rec_of(fn, *x, **kw):
    p = os.path.join(a.o, "gate.tmp.jsonl")
    if os.path.exists(p): os.remove(p)
    r = fn(*x[:-1], dict(strip(x[-1]), tile_cd_log=p), **kw)
    rr = json.loads(open(p).read().splitlines()[-1]); os.remove(p)
    return r, {k: v for k, v in rr.items() if k not in ("sec", "hout")}


def wrapped(Q, weight, weight_q, encoded, H, qa, M=None):
    assert M is None, "this gate runs without H_out"
    (w0, e0), r0 = rec_of(R.cd, Q, weight, weight_q, encoded, H, qa)
    (w1, e1), r1 = rec_of(orig, Q, weight, weight_q, encoded, H, qa)
    (w2, e2), r2 = rec_of(orig, Q, weight, weight_q, encoded, H, qa, M=torch.eye(weight.shape[1], device=H.device))
    o = {"key": qa.get("tile_cd_key"), "research": torch.equal(w0, w1) and torch.equal(e0, e1) and r0 == r1,
         "gate": torch.equal(w1, w2) and torch.equal(e1, e2) and r1 == r2, "ratio": r1["ratio"]}
    out.append(o); print("GATE", json.dumps(o), flush=True)
    return orig(Q, weight, weight_q, encoded, H, qa)        # the job's own CD (writes its tilecd.jsonl line)


T.cd = wrapped
cfg = C.with_overrides(C.load(), **{"inputs.hout.enable": False, "quant.tile_cd.enable": True})
os.makedirs(a.o, exist_ok=True)
K = float(a.K); K = int(K) if K.is_integer() else K
spec = {"type": "rxq.quant", "version": 1, "level": f"K{K}", "K": K, "codebook": C.codebook_for(cfg, K), "seed_offset": 0,
        "keys": a.keys.split(","), "settings_stamp": C.settings_stamp(cfg),
        "inputs": {"ckpt": a.ckpt, "hessians": a.hess, "hin": None, "wpre": None, "hout": None}}
job.run_quant(spec, a.o, a.device, cfg=cfg, profile=False)
ok = sum(o["research"] and o["gate"] for o in out)
print(f"TILECD GPU GATE {'PASS' if out and ok == len(out) else 'FAIL'} {ok}/{len(out)}")
