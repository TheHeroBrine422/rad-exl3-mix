# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The job unit and level assembly.

A quant job (JSON spec, what rad-quant-at-home hands out; schema in DESIGN.md "Job unit"):
  {"type": "rxq.quant", "version": 1, "level": "K2.5-i8", "K": 2.5, "codebook": "i8", "seed_offset": 0,
   "keys": [<checkpoint keys>], "settings_stamp": <16 hex>, "config": <settings file, by content in the datapack>,
   "inputs": {"ckpt": DIR, "hessians": DIR, "hin": DIR|null, "wpre": DIR|null, "hout": DIR|null}}
keys: one tensor, or several (any subset of one layer, usually a whole Hessian group: the group Hessian is then
loaded once a tensor anyway; results do not depend on the split, see quant.py).
Outputs in OUT: t-<key>.safetensors per tensor (its trellis / suh / svh / marker; metadata: settings stamp, code
stamp, K, codebook, level), log.jsonl (ourq's proxy-log record + stage times per tensor), job.json (spec, stamps,
device, seconds). Each tensor file is written .part -> fsync -> rename, so a killed job leaves only whole tensors and
a rerun skips them.

assemble(): job outputs of one level -> an ourq-layout level dir (q-LNN.safetensors per layer, index, config.json,
proxy.jsonl, stamp.json), the form rad-convert's exl3i importer (src=DIR) and every swap-test tool read."""
import json, os, resource, socket, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import config as C, engine, qwen35, quant
from .prof import Prof


def _fsync_put(write, path):
    write(path + ".part")
    fd = os.open(path + ".part", os.O_RDONLY); os.fsync(fd); os.close(fd)
    os.replace(path + ".part", path)


def tensor_file(out, key): return os.path.join(out, f"t-{key}.safetensors")


def run_quant(spec, out, device=0, cfg=None, profile=True, save_wq=False):
    cfg = cfg or C.load(spec.get("config"))
    st = C.settings_stamp(cfg)
    if spec.get("settings_stamp") and spec["settings_stamp"] != st:
        raise SystemExit(f"settings stamp {st} != the job's {spec['settings_stamp']}: wrong settings file")
    K = spec["K"]; cb = spec["codebook"]; so = spec.get("seed_offset", 0); inp = spec["inputs"]
    K = int(K) if float(K).is_integer() else float(K)        # ourq's _kp: integer K as int (the kernels' K argument)
    os.makedirs(out, exist_ok=True)
    keys = list(spec["keys"])
    s0 = C.quant_settings(cfg, K, cb, keys[0], so)
    tcd = (cfg["quant"]["tile_cd"]["sweeps"], cfg["quant"]["tile_cd"]["families"]) if cfg["quant"]["tile_cd"]["enable"] else None
    dev = torch.device("cuda", device)
    Q = engine.setup(dev, codebook=cb, qtf2=s0["qtf2"], drift=s0["drift"], block_trace=s0["block_trace"], tile_cd=tcd)
    code = engine.code_stamp(Q)
    prof = Prof(dev).install(Q) if profile else None
    ck = quant.Ckpt(inp["ckpt"])
    t_job = time.time(); done = 0
    for key in keys:
        tf = tensor_file(out, key)
        if os.path.exists(tf): continue
        s = C.quant_settings(cfg, K, cb, key, so)
        if prof: prof.reset()
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        tensors, rec, wq, rowerr = quant.quantize_tensor(Q, ck, key, s, inp["hessians"], dev, out, hin_dir=inp.get("hin"),
                                                         wpre_dir=inp.get("wpre"),
                                                         hout_dir=(inp.get("hout") or s["hout"]) if "hout" in s else None,
                                                         save_wq=save_wq, prof=prof)
        meta = {"rxq_settings": st, "rxq_code": code, "K": str(K), "codebook": cb, "level": spec.get("level", ""),
                "key": key}
        ts = time.perf_counter()
        _fsync_put(lambda p: save_file(tensors, p, metadata=meta), tf)
        if wq is not None:
            os.makedirs(os.path.join(out, "wq"), exist_ok=True)
            import numpy as np
            np.savez(os.path.join(out, "wq", f"{key}.rowerr.npz"), **rowerr)
            _fsync_put(lambda p: save_file({"weight": wq}, p), os.path.join(out, "wq", f"{key}.safetensors"))
        rec["rxq"] = {"settings": st, "code": code, "family": qwen35.family(key), "wall": round(time.perf_counter() - t0, 3),
                      "save": round(time.perf_counter() - ts, 3), "stages": prof.record() if prof else None,
                      "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3),
                      "maxrss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)}
        with open(os.path.join(out, "log.jsonl"), "a") as fh: fh.write(json.dumps(rec) + "\n")
        print(f"  {key:60s} K{K} proxy {rec['proxy_err']:.3e} {rec['rxq']['wall']:.1f}s", flush=True)
        done += 1
    ck.close()
    jr = {"spec": spec, "settings_stamp": st, "code_stamp": code, "device": torch.cuda.get_device_name(dev),
          "host": socket.gethostname() if os.environ.get("RXQ_RECORD_HOST") else None, "tensors": done,
          "seconds": round(time.time() - t_job, 1)}
    with open(os.path.join(out, "job.json"), "w") as fh: json.dump(jr, fh, indent=1)
    return jr


def plan_level(cfg, K, codebook=None, split="tensor", n_layers=64, seed_offset=0, inputs=None, level=None):
    """the job specs of one level: split = tensor (one job per linear), group (one per Hessian group) or layer"""
    K = int(K) if float(K).is_integer() else float(K)
    cb = codebook or C.codebook_for(cfg, K); st = C.settings_stamp(cfg)
    inputs = inputs or {"ckpt": "weights/", "hessians": "hessians/", "hin": None, "wpre": None, "hout": None}
    units = []
    for l in range(n_layers):
        plan = qwen35.layer_plan(l)
        if split == "layer": units.append([k for _, ks in plan for k in ks])
        elif split == "group": units += [list(ks) for _, ks in plan]
        else: units += [[k] for _, ks in plan for k in ks]
    lv = level or f"K{K}-{cb}" + (f"-s{seed_offset}" if seed_offset else "")
    return [{"type": "rxq.quant", "version": 1, "level": lv, "K": K, "codebook": cb, "seed_offset": seed_offset,
             "keys": ks, "settings_stamp": st, "inputs": dict(inputs)} for ks in units]


def assemble(job_dirs, dst, template=None, n_layers=64, head=False):
    """job outputs -> an ourq-layout level dir. Refuses mixed settings / code stamps and missing tensors."""
    files = {}
    for d in job_dirs:
        for f in os.listdir(d):
            if f.startswith("t-") and f.endswith(".safetensors"): files[f[2:-len(".safetensors")]] = os.path.join(d, f)
    want = qwen35.trunk_keys(n_layers) + (["lm_head"] if head else [])
    miss = [k for k in want if k not in files]
    if miss: raise SystemExit(f"{len(miss)} tensors missing (first {miss[0]})")
    stamps = set()
    for k in want:
        with safe_open(files[k], "pt") as f: m = f.metadata(); stamps.add((m["rxq_settings"], m["rxq_code"], m["K"], m["codebook"]))
    if len(stamps) != 1: raise SystemExit(f"mixed stamps in one level: {sorted(stamps)[:3]}")
    (st, code, K, cb), = stamps
    stamp = f"{st[:8]}{code[:8]}"
    os.makedirs(dst, exist_ok=True)
    plan = [(f"q-L{l:02d}.safetensors", [k for _, ks in qwen35.layer_plan(l) for k in ks]) for l in range(n_layers)]
    if head: plan.append(("q-head.safetensors", ["lm_head"]))
    wm = {}
    for name, keys in plan:
        t = {}
        for k in keys:
            with safe_open(files[k], "pt") as f: t.update({n: f.get_tensor(n) for n in f.keys()})
        for n in t: wm[n] = name
        _fsync_put(lambda p: save_file(t, p, metadata={"ourq_stamp": stamp, "rxq_settings": st, "rxq_code": code}),
                   os.path.join(dst, name))
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
    if template:
        c = json.load(open(template))
        c.setdefault("quantization_config", {"quant_method": "exl3"})
        Kf = float(K)
        c["quantization_config"].update({"bits": Kf, "head_bits": int(Kf) if Kf.is_integer() else Kf,
                                         "calibration": {"source": "challenge calib Hessians"}})
        json.dump(c, open(os.path.join(dst, "config.json"), "w"), indent=1)
    logs = []
    for d in job_dirs:
        p = os.path.join(d, "log.jsonl")
        if os.path.exists(p): logs += [json.loads(l) for l in open(p) if l.strip()]
    last = {r["key"]: r for r in logs}
    with open(os.path.join(dst, "proxy.jsonl"), "w") as fh:
        for k in want:
            if k in last: fh.write(json.dumps(last[k]) + "\n")
    json.dump([{"stamp": stamp, "rxq_settings": st, "rxq_code": code, "K": K, "codebook": cb,
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}], open(os.path.join(dst, "stamp.json"), "w"), indent=1)
    return stamp
