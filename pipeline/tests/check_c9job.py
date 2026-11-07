#!/usr/bin/env python3
"""check_c9job.py JOB_OUT C9E27 MODEL HESS WORK -- the C9 scales an rxq quant job computed (c9-<key>.safetensors, its
log.jsonl "c9" lines) against v3/e3/settle/w75/c9e27.py run on the same codes (DEV from the environment: cuda = the
device the job used, for bit identity; cpu = the f64 GEMMs on another device, reported as max |dc| / c).
WORK/level: a level dir made of the job's t-files (index + stamp.json with their codebook), then c9e27.py on it with
C9E27_ONLY = the job's writers (alpha attn_o=-1, gdn_out=3: the settings' values). Prints each writer's in-job C9 time
(log.jsonl rxq.c9_s) next to c9e27's, then "C9JOB PASS|FAIL n/m" (PASS: scales torch.equal and c9.jsonl equal but
"target"). C9E27 needs deq.py next to it (gpubox: /data/c9w27/code/c9e27.py, used read-only)."""
import json, os, shutil, subprocess, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

JOB, C9E27, MODEL, HESS, W = sys.argv[1:6]
lv, ref = f"{W}/level", f"{W}/c9e27"
for d in (lv, ref): shutil.rmtree(d, ignore_errors=True)
os.makedirs(lv)
recs = {}
for l in open(f"{JOB}/log.jsonl"):
    if l.strip(): r = json.loads(l); recs[r["key"]] = r
keys = sorted(k for k in recs if "c9" in recs[k] and os.path.exists(f"{JOB}/c9-{k}.safetensors"))
assert keys, f"{JOB}: no C9 writers (post.c9 off, or no writer keys?)"
wm, cbs = {}, set()
for k in keys:
    with safe_open(f"{JOB}/t-{k}.safetensors", "pt") as f:
        t = {n: f.get_tensor(n) for n in f.keys()}; cbs.add(f.metadata()["codebook"])
    name = f"q-{k}.safetensors"; save_file(t, f"{lv}/{name}"); wm.update({n: name for n in t})
assert len(cbs) == 1, cbs
json.dump({"metadata": {}, "weight_map": wm}, open(f"{lv}/model.safetensors.index.json", "w"))
cb = cbs.pop()
json.dump([{"codebook": cb, "args": {"codebook": cb}}], open(f"{lv}/stamp.json", "w"))   # rxq's and ourq's form (deq.py reads args)
env = dict(os.environ, C9E27_ONLY=",".join(keys))
t0 = time.time()
rc = subprocess.call([sys.executable, C9E27, lv, ref, "--alpha", "attn_o=-1,gdn_out=3", "--model", MODEL, "--hess", HESS],
                     env=env, stdout=open(f"{ref}.log", "w"), stderr=subprocess.STDOUT)
dt = time.time() - t0
if rc: print(open(f"{ref}.log").read()[-2000:]); print("C9JOB FAIL 0/%d (c9e27 rc %d)" % (len(keys), rc)); sys.exit(1)
rj = {json.loads(l)["key"]: json.loads(l) for l in open(f"{ref}/c9.jsonl")}
ok = 0
for k in keys:
    with safe_open(f"{JOB}/c9-{k}.safetensors", "pt") as x, safe_open(f"{ref}/scales/{k}.safetensors", "pt") as y:
        a, b = x.get_tensor("c"), y.get_tensor("c")
        same = torch.equal(a, b) and torch.equal(x.get_tensor("r"), y.get_tensor("r"))
        rel = float(((a.double() - b.double()).abs() / b.double()).max())
    jr = {n: v for n, v in recs[k]["c9"].items() if n != "target"}
    good = same and jr == rj.get(k)
    ok += good
    print(f"{'SAME' if good else 'DIFF'} {k}: scales equal {same} (max |dc|/c {rel:.2e}), c9.jsonl equal {jr == rj.get(k)}; "
          f"in-job C9 {recs[k]['rxq'].get('c9_s')} s, tensor quant {recs[k]['rxq']['wall']} s")
print(f"c9e27.py on {os.environ.get('DEV', 'cuda')}: {dt:.1f} s for {len(keys)} writers (process incl. imports)")
print(f"C9JOB {'PASS' if ok == len(keys) else 'FAIL'} {ok}/{len(keys)}")
sys.exit(0 if ok == len(keys) else 1)
