#!/usr/bin/env python3
"""check_deq_c9.py WORK RXM LEVEL MODEL HESS [UNIT ...] -- rxq.deq and rxq.post's C9 against the research tools on a
real 27B level on the CPU (LEVEL: an ourq-layout level dir; the colo check uses hconv's K2.5 i8 layer 10 shard with a
generated index and stamp, made by the caller).
  lut    rxq.deq.lut(cb) == rad-quant-at-home checker/exl3dec.lut(cb) (= what exl3deq.py reads) for mul1 / i8 / i8c / i4
  deq    every tensor of LEVEL decoded by rxq.deq.Level, v3/yaqa/swj27/exl3deq.py and v3/yaqa/asym27/deq.py: torch.equal
         (rxq.deq replaces both: swjfit27 used the first, sibmom27 / c9e27 the second)
  c9     rxq.post c9 vs v3/e3/settle/w75/c9e27.py (DEV=cpu, the same writers via C9E27_ONLY / RXQ_C9_ONLY), alpha
         gdn_out=3 (and attn_o=-1 if the level has a full-attention layer): scales files and c9.jsonl (but "target") equal
  injob  the C9 the quant job computes (rxq.post.c9_tensor on the writer's tensors as the job holds them, W, the group's
         calib H_in; rxq.job._assemble_c9 -> LEVEL/c9) == c9e27.py's scales and c9.jsonl from the c9 unit (run c9
         first), and its time per writer (the per-tensor cost in the job, CPU here)
Needs RXM/volunteer/checker/exl3dec.py (exl3deq.py's import; copy it from rad-quant-at-home) and deq.py next to
c9e27.py (as gpubox's /data/c9w27/code holds it). Last line "DEQ C9 CHECKS n/m PASS"."""
import importlib.util, json, os, shutil, subprocess, sys

import torch
from safetensors import safe_open

W, RXM, LEVEL, MODEL, HESS = sys.argv[1:6]
only = sys.argv[6:]
os.makedirs(W, exist_ok=True)
sys.path.insert(0, f"{RXM}/pipeline")
from rxq import deq as R, post  # noqa: E402
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
    sys.path.insert(0, os.path.dirname(path)); sp.loader.exec_module(m); return m


def want(u): return not only or u in only


X = load("exl3dec", f"{RXM}/volunteer/checker/exl3dec.py")
if want("lut"):
    bad = [cb for cb in R.CODEBOOKS if not (R.lut(cb) == X.lut(cb)).all()]
    report("lut", not bad, f"{len(R.CODEBOOKS)} codebooks x 65,536 states, differing: {bad}")

if want("deq"):
    A = load("exl3deq", f"{RXM}/v3/yaqa/swj27/exl3deq.py")
    B = load("asym27deq", f"{RXM}/v3/yaqa/asym27/deq.py")
    la, lb, lr = A.Level(LEVEL, dev="cpu"), B.Level(LEVEL, dev="cpu"), R.Level(LEVEL, dev="cpu")
    keys = lr.keys(); bad = []
    for k in keys:
        r = lr.get(k)
        ea, eb = torch.equal(r, la.get(k)), torch.equal(r, lb.get(k))
        if not (ea and eb): bad.append((k, ea, eb))
        print(f"  {k}: K{lr.K(k):g} {lr.codebook} {tuple(r.shape)} == exl3deq {ea}, == asym27 deq {eb}", flush=True)
    report("deq", keys and not bad, f"{len(keys)} tensors, differing {bad[:2]}")

if want("c9"):
    ix = json.load(open(f"{LEVEL}/model.safetensors.index.json"))["weight_map"]
    wr = sorted({n[:-len(".trellis")] for n in ix if n.endswith(".trellis") and (".linear_attn.out_proj" in n or ".self_attn.o_proj" in n)})
    al = "gdn_out=3" + (",attn_o=-1" if any(".self_attn.o_proj" in k for k in wr) else "")
    env = dict(os.environ, DEV="cpu", C9E27_ONLY=",".join(wr), RXQ_C9_ONLY=",".join(wr), PYTHONPATH=f"{RXM}/pipeline")
    a, b = f"{W}/c9-ref", f"{W}/c9-rxq"
    for d in (a, b): shutil.rmtree(d, ignore_errors=True)
    ra = subprocess.call([sys.executable, f"{RXM}/v3/e3/settle/w75/c9e27.py", LEVEL, a, "--alpha", al, "--model", MODEL, "--hess", HESS],
                         env=env, stdout=open(f"{a}.log", "w"), stderr=subprocess.STDOUT)
    rb = subprocess.call([sys.executable, "-m", "rxq.post", "c9", LEVEL, b, "--alpha", al, "--model", MODEL, "--hess", HESS],
                         env=env, stdout=open(f"{b}.log", "w"), stderr=subprocess.STDOUT)
    bad = []
    if ra == rb == 0:
        for f in sorted(os.listdir(f"{a}/scales")):
            with safe_open(f"{a}/scales/{f}", "pt") as x, safe_open(f"{b}/scales/{f}", "pt") as y:
                if sorted(x.keys()) != sorted(y.keys()) or not all(torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys()): bad.append(f)
        ja = [json.loads(l) for l in open(f"{a}/c9.jsonl")]; jb = [{k: v for k, v in json.loads(l).items() if k != "target"} for l in open(f"{b}/c9.jsonl")]
        if ja != jb: bad.append("c9.jsonl")
    report("c9", ra == rb == 0 and not bad and wr, f"rc {ra}/{rb}; alpha {al}; {len(wr)} writers ({', '.join(k.split('.', 3)[3] for k in wr)}); differing {bad[:2]}")

if want("injob"):
    import time
    from rxq import job
    a, d = f"{W}/c9-ref", f"{W}/c9-job"
    shutil.rmtree(d, ignore_errors=True); os.makedirs(f"{d}/jobs")
    ref = {json.loads(l)["key"]: json.loads(l) for l in open(f"{a}/c9.jsonl")}
    ix = json.load(open(f"{LEVEL}/model.safetensors.index.json"))["weight_map"]
    mx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    cb = R.level_codebook(LEVEL); al = {"attn_o": -1.0, "gdn_out": 3.0, "down": 0.0}
    recs, files, bad, secs = {}, {}, [], []
    for key in ref:
        with safe_open(f"{LEVEL}/{ix[key + '.trellis']}", "pt") as f:
            tensors = {n: f.get_tensor(n) for n in (f"{key}.trellis", f"{key}.suh", f"{key}.svh")}
        with safe_open(f"{MODEL}/{mx[key + '.weight']}", "pt") as f: w = f.get_tensor(key + ".weight")
        from rxq import qwen35
        hname = qwen35.locate(key)[3]                                                    # the job's group name
        t0 = time.perf_counter()
        c, rec = post.c9_tensor(tensors, key, cb, w, f"{HESS}/{hname}.safetensors", al[post.c9_family(key)], 0.5, "cpu")
        secs.append(time.perf_counter() - t0)
        recs[key] = {"c9": {"key": key, "fam": post.c9_family(key), "layer": int(key.split(".")[3]), "alpha": al[post.c9_family(key)],
                            **rec, "target": "W/H"}}
        files[key] = f"{d}/jobs/c9-{key}.safetensors"
        from safetensors.torch import save_file
        save_file(post.c9_file(c, w.shape[1]), files[key], metadata={"rxq_settings": "test", "c9_alpha": json.dumps(al, sort_keys=True)})
        print(f"  {key}: {tuple(w.shape)} in-job C9 {secs[-1]:.1f} s (CPU, {torch.get_num_threads()} threads)", flush=True)
    job._assemble_c9(files, recs, list(ref), "test", f"{d}/c9", 64)
    for k in ref:
        with safe_open(f"{a}/scales/{k}.safetensors", "pt") as x, safe_open(f"{d}/c9/scales/{k}.safetensors", "pt") as y:
            if sorted(x.keys()) != sorted(y.keys()) or not all(torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys()): bad.append(k)
    jb = [{k: v for k, v in json.loads(l).items() if k != "target"} for l in open(f"{d}/c9/c9.jsonl")]
    if jb != [json.loads(l) for l in open(f"{a}/c9.jsonl")]: bad.append("c9.jsonl")
    report("injob", ref and not bad and os.path.exists(f"{d}/c9/.done"),
           f"{len(ref)} writers vs c9e27.py (scales + c9.jsonl), differing {bad[:2]}; {sum(secs) / max(1, len(secs)):.1f} s a writer on the CPU")

print(f"DEQ C9 CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if res and all(res) else 1)
