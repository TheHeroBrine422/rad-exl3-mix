#!/usr/bin/env python3
"""check_levelpass.py WORK RXM MODEL08 [UNIT ...] -- the level-pass ports against the research tools, on the CPU (colo):
the 0.8B (Qwen3.5-0.8B bf16, 24 layers) and a synthetic corpus (random token ids: no calib document is read), both
tools run as subprocesses in the same environment (P2B_BLOCK=causal_conv1d,fla: transformers' torch GDN path).
  swj    rxq.swjfit vs v3/yaqa/swj27/swjfit27.py: a :wq source (the 0.8B's gate / up + 3% noise, f16), 3 stages,
         --no-cache (the production forward), --ckpt-min 0 (a checkpoint after every document: the resume path);
         scales files, swiglu.jsonl and swjfit.json (but tool / created) equal
  swjc   the same with the fresh-DynamicCache forward (no --no-cache)
  swjb   rxq.swjfit alone vs in a batch: sources a (as swj) and b (seed 77) each fitted alone (3 stages) and both in one
         call (--split 0,12,24), --no-cache; every file of OUT/a and OUT/b (but swjfit.json: split, created) equal
  sib    rxq.sibmom vs v3/yaqa/asym27/sibmom27.py: --wq siblings (q/k/v or GDN qkv/z, gate / up + noise), every layer,
         256-token windows; every moment file (H, C, N, metadata), windows.json and the per-layer done markers equal
  hin    rxq.hin vs eval/gold27/mkhin27.py: 3 stages, 256-token chunks, a flush (checkpoint) every 2 documents; every
         OUT file byte for byte (97 Hessians), collect.json equal but tool / created / plan.out. Both sides run with
         OMP_NUM_THREADS=1: at 4 threads on a loaded host the CPU SDPA (full-attention layers) is not reproducible run to
         run, mkhin27.py against itself included (colo 11-07: 84 of 97 files differed between two mkhin27 runs)
Inputs are made under WORK once (meta/, wq/; wqb/ for swjb). Files compare by content: safetensors tensors torch.equal +
metadata dict (header bytes vary by process), other files by bytes. Last line "LEVELPASS CHECKS n/m PASS"."""
import filecmp, json, os, shutil, subprocess, sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

W, RXM, M08 = sys.argv[1:4]
only = sys.argv[4:]
os.makedirs(W, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, P2B_BLOCK="causal_conv1d,fla", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"),
           PYTHONPATH=os.pathsep.join([f"{RXM}/pipeline", f"{W}/../pylib", os.environ.get("PYTHONPATH", "")]),
           SWJ_DEV="cpu", DEV="cpu", MKH27_DEV="cpu", POOLS=f"{W}/meta/pools.json")
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


def run(name, argv, log, **env):
    with open(log, "w") as f: rc = subprocess.call(argv, env=dict(ENV, **env), stdout=f, stderr=subprocess.STDOUT, cwd=W)
    if rc: print(f"  {name}: rc {rc} (tail of {log}):\n" + "".join(open(log).readlines()[-8:]), flush=True)
    return rc


def same_st(pa, pb):
    with safe_open(pa, "pt") as x, safe_open(pb, "pt") as y:
        if x.metadata() != y.metadata() or sorted(x.keys()) != sorted(y.keys()): return False
        return all(x.get_tensor(n).dtype == y.get_tensor(n).dtype and torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys())


def same_tree(a, b, skip=()):
    """every file under a equal in b (recursive); -> (files compared, differing)"""
    n, bad = 0, []
    for root, _, files in os.walk(a):
        for f in files:
            pa = os.path.join(root, f); rel = os.path.relpath(pa, a); pb = os.path.join(b, rel)
            if rel in skip: continue
            n += 1
            if not os.path.exists(pb): bad.append(rel + " (missing)"); continue
            ok = same_st(pa, pb) if f.endswith(".safetensors") else filecmp.cmp(pa, pb, shallow=False)
            if not ok: bad.append(rel)
    return n, bad


def json_eq(pa, pb, drop):
    a, b = json.load(open(pa)), json.load(open(pb))
    for k in drop:                             # "plan.out": a nested key
        da, db = a, b; *head, last = k.split(".")
        for h in head: da, db = da.get(h, {}), db.get(h, {})
        da.pop(last, None); db.pop(last, None)
    return a == b


# ---------------------------------------------------------------- inputs (once)
Lp = "model.language_model.layers"
if not os.path.exists(f"{W}/meta/.done"):
    os.makedirs(f"{W}/meta", exist_ok=True)
    g = np.random.default_rng(76)
    lens = [300, 513, 129, 700, 64, 250, 420, 97]
    toks = g.integers(0, 248000, sum(lens)).astype(np.int32)
    toks.tofile(f"{W}/meta/tokens.i32")
    json.dump({"docs": [{"tokens": n} for n in lens]}, open(f"{W}/meta/kld.json", "w"))
    json.dump({"pool": ["qad"] * len(lens)}, open(f"{W}/meta/pools.json", "w"))
    open(f"{W}/meta/.done", "w").close()


def stand_in(d, seed):
    """a :wq source under d (once): every layer's attention siblings + gate / up of the 0.8B, + 3% noise (seed), f16"""
    if os.path.exists(f"{d}/.done"): return
    os.makedirs(f"{d}/wq", exist_ok=True)
    ix = json.load(open(f"{M08}/model.safetensors.index.json"))["weight_map"]
    cfg = json.load(open(f"{M08}/config.json")); lt = cfg["text_config"]["layer_types"]
    gen = torch.Generator().manual_seed(seed)
    for l in range(len(lt)):
        att = [f"self_attn.{s}_proj" for s in "qkv"] if lt[l] == "full_attention" else ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
        for s in att + ["mlp.gate_proj", "mlp.up_proj"]:
            k = f"{Lp}.{l}.{s}"
            with safe_open(f"{M08}/{ix[k + '.weight']}", "pt") as f: w = f.get_tensor(k + ".weight").float()
            q = w + 0.03 * w.std() * torch.randn(w.shape, generator=gen)
            save_file({"weight": q.half().contiguous()}, f"{d}/wq/{k}.safetensors")
    open(f"{d}/.done", "w").close()


stand_in(f"{W}/wq", 76)


def want(u): return not only or u in only


# ---------------------------------------------------------------- swj
for unit, extra in (("swj", ["--no-cache"]), ("swjc", [])):
    if not want(unit): continue
    outs = {}
    for side, argv0 in (("ref", [PY, "-I", f"{RXM}/v3/yaqa/swj27/swjfit27.py"]), ("rxq", [PY, "-m", "rxq.swjfit"])):
        d = f"{W}/{unit}-{side}"; shutil.rmtree(d, ignore_errors=True); shutil.rmtree(d + "-scr", ignore_errors=True)
        if side == "ref": argv0 = [PY, f"{RXM}/v3/yaqa/swj27/swjfit27.py"]     # -I would drop PYTHONPATH (P2B_BLOCK is env)
        rc = run(f"{unit}/{side}", argv0 + [M08, f"{W}/meta", d, "--src", f"a={W}/wq:wq", "--scratch", d + "-scr",
                                           "--split", "0,8,16,24", "--cap", "512", "--ckpt-min", "0"] + extra, d + ".log")
        outs[side] = (rc, d)
    (ra, da), (rb, db) = outs["ref"], outs["rxq"]
    n, bad = same_tree(f"{da}/a", f"{db}/a", skip=("swjfit.json",)) if ra == rb == 0 else (0, ["(a run failed)"])
    js = ra == rb == 0 and json_eq(f"{da}/a/swjfit.json", f"{db}/a/swjfit.json", ("tool", "created"))
    report(unit, ra == rb == 0 and not bad and js, f"rc {ra}/{rb}; {n} files compared, {len(bad)} differ {bad[:2]}; swjfit.json equal {js}")

# ---------------------------------------------------------------- swjb
if want("swjb"):
    stand_in(f"{W}/wqb", 77)
    rcs = {}
    for side, srcs, split in (("a", ["a"], "0,8,16,24"), ("b", ["b"], "0,8,16,24"), ("ab", ["a", "b"], "0,12,24")):
        d = f"{W}/swjb-{side}"; shutil.rmtree(d, ignore_errors=True); shutil.rmtree(d + "-scr", ignore_errors=True)
        sv = [x for s in srcs for x in ("--src", f"{s}={W}/{'wq' if s == 'a' else 'wqb'}:wq")]
        rcs[side] = run(f"swjb/{side}", [PY, "-m", "rxq.swjfit", M08, f"{W}/meta", d] + sv + ["--scratch", d + "-scr",
                        "--split", split, "--cap", "512", "--ckpt-min", "0", "--no-cache"], d + ".log")
    ok = not any(rcs.values())
    (na, ba), (nb, bb) = ((same_tree(f"{W}/swjb-{s}/{s}", f"{W}/swjb-ab/{s}", skip=("swjfit.json",)) if ok else (0, ["(a run failed)"]))
                          for s in "ab")
    report("swjb", ok and na > 0 and nb > 0 and not ba and not bb,
           f"rc {rcs}; a alone vs a in a + b: {na} files compared, {len(ba)} differ {ba[:2]}; b: {nb} compared, {len(bb)} differ {bb[:2]}")

# ---------------------------------------------------------------- sib
if want("sib"):
    outs = {}
    for side, argv0 in (("ref", [PY, f"{RXM}/v3/yaqa/asym27/sibmom27.py"]), ("rxq", [PY, "-m", "rxq.sibmom"])):
        d = f"{W}/sib-{side}"; shutil.rmtree(d, ignore_errors=True)
        rc = run(f"sib/{side}", argv0 + [M08, f"{W}/meta", d, "--wq", f"{W}/wq/wq", "--pools", "qad", "--tokens", "100000",
                                        "--win", "256", "--no-disk-check", "--snap-layers", "3"], d + ".log")
        outs[side] = (rc, d)
    (ra, da), (rb, db) = outs["ref"], outs["rxq"]
    n, bad = same_tree(da, db) if ra == rb == 0 else (0, ["(a run failed)"])
    report("sib", ra == rb == 0 and not bad and n > 0, f"rc {ra}/{rb}; {n} files compared (moments, half/, windows.json, .done/), {len(bad)} differ {bad[:2]}")

# ---------------------------------------------------------------- hin
if want("hin"):
    outs = {}
    for side, argv0 in (("ref", [PY, f"{RXM}/eval/gold27/mkhin27.py"]), ("rxq", [PY, "-m", "rxq.hin"])):
        d = f"{W}/hin-{side}"; shutil.rmtree(d, ignore_errors=True); shutil.rmtree(d + "-scr", ignore_errors=True)
        rc = run(f"hin/{side}", argv0 + [M08, f"{W}/meta", d, "--scratch", d + "-scr", "--split", "0,8,16,24", "--chunk", "256",
                                        "--flush", "2"], d + ".log", OMP_NUM_THREADS="1")
        outs[side] = (rc, d)
    (ra, da), (rb, db) = outs["ref"], outs["rxq"]
    ok = ra == rb == 0
    if ok:
        files = sorted(f for f in os.listdir(da) if f.endswith(".safetensors"))
        bad = [f for f in files if not filecmp.cmp(f"{da}/{f}", f"{db}/{f}", shallow=False)]
        js = json_eq(f"{da}/collect.json", f"{db}/collect.json", ("tool", "created", "plan.out"))
    else: files, bad, js = [], ["(a run failed)"], False
    report("hin", ok and not bad and js and len(files) > 0, f"rc {ra}/{rb}; {len(files)} Hessians byte-compared, {len(bad)} differ {bad[:2]}; collect.json equal {js}")

print(f"LEVELPASS CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if res and all(res) else 1)
