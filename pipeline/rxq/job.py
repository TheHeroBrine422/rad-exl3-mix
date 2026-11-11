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
device, seconds). With post.c9 on, a writer's job also computes its C9 row scales right after the quantization (W, Q
and the group's H_in at hand: rxq.post.c9_tensor, seconds) into c9-<key>.safetensors (swjapply's "c" / "r"), written
before the tensor file; log.jsonl's record carries c9.jsonl's line ("c9") and its time (rxq.c9_s). Each file is
written .part -> fsync -> rename, so a killed job leaves only whole tensors and a rerun skips them.
Diagnostics (#139, rxq.refit; RXQ_DIAG=0 turns them off): diag-<key>.safetensors per tensor (per-output-channel errors,
the 128 x 128 error map, the pre-refit scales; written before the tensor file, never read by assemble) and the record's
"diag" (error trajectory incl. post-C9, the proxy check, overload counters, g_scale). Experiments only: with
RXQ_REFIT_ARMS (+ RXQ_REFIT_ARMS_WQ=DIR) the other refit modes' scores (and decoded weights, DIR/<arm>/wq/<key>) too.

assemble(): job outputs of one level -> an ourq-layout level dir (q-LNN.safetensors per layer, index, config.json,
proxy.jsonl, stamp.json), the form rad-convert's exl3i importer (src=DIR) and every swap-test tool read."""
import json, os, resource, socket, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import config as C, engine, qwen35, quant, refit
from .prof import Prof


def _fsync_put(write, path):
    write(path + ".part")
    fd = os.open(path + ".part", os.O_RDONLY); os.fsync(fd); os.close(fd)
    os.replace(path + ".part", path)


def tensor_file(out, key): return os.path.join(out, f"t-{key}.safetensors")


def c9_path(out, key): return os.path.join(out, f"c9-{key}.safetensors")


def diag_path(out, key): return os.path.join(out, f"diag-{key}.safetensors")


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
    tcd = C.tile_cd(cfg)
    hout = None
    if C.hout_applies(cfg, K):
        hout = inp.get("hout") or s0["hout"]
        if not hout: raise SystemExit(f"K{K}: H_out applies (inputs.hout) but neither the job nor the settings name its dir")
    elif inp.get("hout"):
        raise SystemExit(f"K{K}: the job names an H_out dir but H_out does not apply at this level "
                         f"(inputs.hout enable / apply_k_min / apply_k_max)")
    dev = torch.device("cpu") if device == "cpu" else torch.device("cuda", device)
    cuda = dev.type == "cuda"
    Q = engine.setup(dev, codebook=cb, qtf2=s0["qtf2"], drift=s0["drift"], block_trace=s0["block_trace"], tile_cd=tcd,
                     block_order=C.block_order(cfg))
    code = engine.code_stamp(Q); kern = engine.kernels(Q)
    prof = Prof(dev).install(Q) if profile else None
    ck = quant.Ckpt(inp["ckpt"])
    t_job = time.time(); done = 0
    for key in keys:
        tf = tensor_file(out, key)
        if os.path.exists(tf): continue
        s = C.quant_settings(cfg, K, cb, key, so)
        if prof: prof.reset()
        if cuda: torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        dg = {}
        tensors, rec, wq, rowerr = quant.quantize_tensor(Q, ck, key, s, inp["hessians"], dev, out, hin_dir=inp.get("hin"),
                                                         wpre_dir=inp.get("wpre"),
                                                         hout_dir=hout,
                                                         save_wq=save_wq, prof=prof, diag_out=dg)
        meta = {"rxq_settings": st, "rxq_code": code, "K": str(K), "codebook": cb, "level": spec.get("level", ""),
                "key": key}
        if kern != "gpu": meta["rxq_kernels"] = kern     # cpu-standin: test only, assemble refuses it without allow_standin
        al = C.c9_injob(cfg, key, inp.get("hin"), inp.get("wpre"))
        c9s = None
        if al is not None:                      # S5b inside the job: the writer's C9 row scales (rxq.post.c9_tensor)
            meta["c9"] = str(al)                # this writer's alpha: assembly then requires its c9 file
            from . import post
            tc = time.perf_counter()
            _, _, _, hname, _ = qwen35.locate(key)
            c, c9rec = post.c9_tensor(tensors, key, cb, ck.get(key + ".weight"), f"{inp['hessians']}/{hname}.safetensors",
                                      al, cfg["post"]["c9"]["e_clip"], dev)
            c9rec = {"key": key, "fam": post.c9_family(key), "layer": int(key.split(".")[3]), "alpha": al, **c9rec,
                     "target": "W/H"}
            if "e1" in dg and "diag" in rec:          # the trajectory's post-C9 point: e1 rescaled by c (x1 / d1)
                ec = refit.rescaled(dg, c); h = dg.get("h")
                rec["diag"].setdefault("traj", {})["post_c9"] = {
                    "f1": float(ec.sum() / dg["r1"].sum()), "f2": None if h is None else float((h * ec).sum() / (h * dg["r1"]).sum())}
            c9meta = dict(meta, c9_alpha=json.dumps(cfg["post"]["c9"]["alpha"], sort_keys=True))
            _fsync_put(lambda p: save_file(post.c9_file(c, tensors[f"{key}.suh"].numel()), p, metadata=c9meta),
                       c9_path(out, key))
            del c
            if cuda: torch.cuda.empty_cache()
            c9s = round(time.perf_counter() - tc, 3)
        ts = time.perf_counter()
        if "e1" in dg:                                # #139: before the tensor file (a rerun redoes both)
            dmeta = dict(meta, rxq_diag="1", refit=json.dumps(s.get("refit", True)), d_off=repr(dg.get("d_off", 0.0)))
            _fsync_put(lambda p: save_file(refit.tensors(key, dg), p, metadata=dmeta), diag_path(out, key))
            rec["diag"]["bytes"] = os.path.getsize(diag_path(out, key))
        aw = os.environ.get("RXQ_REFIT_ARMS_WQ")
        if aw and dg.get("arm_wq"):                   # experiments (#138): each refit arm's decoded weight, f16 (out, in)
            for arm, x in dg["arm_wq"].items():
                os.makedirs(os.path.join(aw, arm, "wq"), exist_ok=True)
                _fsync_put(lambda p: save_file({"weight": x}, p), os.path.join(aw, arm, "wq", f"{key}.safetensors"))
        _fsync_put(lambda p: save_file(tensors, p, metadata=meta), tf)
        if wq is not None:
            os.makedirs(os.path.join(out, "wq"), exist_ok=True)
            import numpy as np
            np.savez(os.path.join(out, "wq", f"{key}.rowerr.npz"), **rowerr)
            _fsync_put(lambda p: save_file({"weight": wq}, p), os.path.join(out, "wq", f"{key}.safetensors"))
        if al is not None: rec["c9"] = c9rec
        rec["rxq"] = {"settings": st, "code": code, "kernels": kern, "family": qwen35.family(key), "wall": round(time.perf_counter() - t0, 3),
                      "save": round(time.perf_counter() - ts, 3), "stages": prof.record() if prof else None,
                      "c9_s": c9s, "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 3) if cuda else None,
                      "maxrss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)}
        with open(os.path.join(out, "log.jsonl"), "a") as fh: fh.write(json.dumps(rec) + "\n")
        print(f"  {key:60s} K{K} proxy {rec['proxy_err']:.3e} {rec['rxq']['wall']:.1f}s", flush=True)
        done += 1
    ck.close()
    jr = {"spec": spec, "settings_stamp": st, "code_stamp": code, "kernels": kern,
          "device": torch.cuda.get_device_name(dev) if cuda else "cpu",
          "host": socket.gethostname() if os.environ.get("RXQ_RECORD_HOST") else None, "tensors": done,
          "seconds": round(time.time() - t_job, 1)}
    with open(os.path.join(out, "job.json"), "w") as fh: json.dump(jr, fh, indent=1)
    return jr


def plan_level(cfg, K, codebook=None, split="tensor", n_layers=64, seed_offset=0, inputs=None, level=None):
    """the job specs of one level: split = tensor (one job per linear), group (one per Hessian group) or layer"""
    K = int(K) if float(K).is_integer() else float(K)
    cb = codebook or C.codebook_for(cfg, K); st = C.settings_stamp(cfg)
    inputs = dict(inputs or {"ckpt": "weights/", "hessians": "hessians/", "hin": None, "wpre": None, "hout": None})
    if C.hout_applies(cfg, K): inputs["hout"] = inputs.get("hout") or cfg["inputs"]["hout"]["dir"] or "hout/"
    else: inputs["hout"] = None
    units = []
    for l in range(n_layers):
        plan = qwen35.layer_plan(l)
        if split == "layer": units.append([k for _, ks in plan for k in ks])
        elif split == "group": units += [list(ks) for _, ks in plan]
        else: units += [[k] for _, ks in plan for k in ks]
    lv = level or f"K{K}-{cb}" + (f"-s{seed_offset}" if seed_offset else "")
    return [{"type": "rxq.quant", "version": 1, "level": lv, "K": K, "codebook": cb, "seed_offset": seed_offset,
             "keys": ks, "settings_stamp": st, "inputs": dict(inputs)} for ks in units]


def assemble(job_dirs, dst, template=None, n_layers=64, head=False, partial=False, allow_standin=False):
    """job outputs -> an ourq-layout level dir. Refuses mixed settings / code stamps, missing tensors and CPU stand-in
    results. partial (dry runs): exactly the tensors the job dirs hold, shards only for their layers; allow_standin:
    accept rxq/cpukern.py's results (test only)."""
    files = {}
    for d in job_dirs:
        for f in os.listdir(d):
            if f.startswith("t-") and f.endswith(".safetensors"): files[f[2:-len(".safetensors")]] = os.path.join(d, f)
    want = qwen35.trunk_keys(n_layers) + (["lm_head"] if head else [])
    if partial:
        want = [k for k in qwen35.trunk_keys(n_layers) + ["lm_head"] if k in files]
        if not want: raise SystemExit("no tensors in the job dirs")
    miss = [k for k in want if k not in files]
    if miss: raise SystemExit(f"{len(miss)} tensors missing (first {miss[0]})")
    stamps, c9want = set(), []
    for k in want:
        with safe_open(files[k], "pt") as f: m = f.metadata(); stamps.add((m["rxq_settings"], m["rxq_code"], m["K"], m["codebook"]))
        if m.get("rxq_kernels", "gpu") != "gpu" and not allow_standin:
            raise SystemExit(f"{k}: made with the {m['rxq_kernels']} kernels (test only), not for a level")
        if "c9" in m: c9want.append(k)
    if len(stamps) != 1: raise SystemExit(f"mixed stamps in one level: {sorted(stamps)[:3]}")
    (st, code, K, cb), = stamps
    stamp = f"{st[:8]}{code[:8]}"
    os.makedirs(dst, exist_ok=True)
    plan = [(f"q-L{l:02d}.safetensors", [k for _, ks in qwen35.layer_plan(l) for k in ks if k in want]) for l in range(n_layers)]
    plan = [(n, ks) for n, ks in plan if ks]
    if "lm_head" in want: plan.append(("q-head.safetensors", ["lm_head"]))
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
    c9 = {}
    for d in job_dirs:
        for f in os.listdir(d):
            if f.startswith("c9-") and f.endswith(".safetensors"): c9[f[3:-len(".safetensors")]] = os.path.join(d, f)
    for d in job_dirs:
        p = os.path.join(d, "log.jsonl")
        if os.path.exists(p): logs += [json.loads(l) for l in open(p) if l.strip()]
    last = {r["key"]: r for r in logs}
    with open(os.path.join(dst, "proxy.jsonl"), "w") as fh:
        for k in want:
            if k in last: fh.write(json.dumps(last[k]) + "\n")
    if c9 or c9want: _assemble_c9(c9, last, want, st, os.path.join(dst, "c9"), n_layers, c9want)
    # "args": what rad-convert's exl3i importer reads (exl3quant stamp_agrees: args.codebook, ourq's default mul1 when
    # absent, so an i8 / i4 level without it is refused, and a recipe saying mul1 would be accepted). rxq_settings in
    # args also makes `rxq levelargs` refuse the level (it is not an ourq level; S3 note)
    Kn = int(float(K)) if float(K).is_integer() else float(K)
    json.dump([{"stamp": stamp, "rxq_settings": st, "rxq_code": code, "K": K, "codebook": cb,
                "args": {"K": Kn, "codebook": cb, "rxq_settings": st},
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}], open(os.path.join(dst, "stamp.json"), "w"), indent=1)
    return stamp


def _assemble_c9(files, recs, want, st, dst, n_layers, required=()):
    """the jobs' C9 sidecars -> DST in rxq.post c9's layout (scales/<key>.safetensors, c9.jsonl in its writer order,
    .done): `rxq.post apply LEVEL[,..] LEVEL/c9 OUT` folds them into svh. Refuses a missing writer or other stamps."""
    from . import post
    miss = [k for k in required if k not in files]
    if miss: raise SystemExit(f"C9 on: {len(miss)} writers' jobs made C9 scales that are missing (first {miss[0]})")
    alphas = set()
    for k, p in files.items():
        with safe_open(p, "pt") as f: m = f.metadata()
        if m["rxq_settings"] != st: raise SystemExit(f"C9 scales of {k}: settings {m['rxq_settings']} != the level's {st}")
        alphas.add(m["c9_alpha"])
    if len(alphas) != 1: raise SystemExit(f"C9 scales with mixed alphas {sorted(alphas)}")
    keys = [k for k, _, _ in post.writer_keys(json.loads(alphas.pop()), n_layers) if k in want]
    miss = [k for k in keys if k not in files or "c9" not in recs.get(k, {})]
    if miss: raise SystemExit(f"C9 on: {len(miss)} writers without their scales / c9 record (first {miss[0]})")
    os.makedirs(os.path.join(dst, "scales"), exist_ok=True)
    for k in keys:
        with safe_open(files[k], "pt") as f: t = {n: f.get_tensor(n) for n in f.keys()}
        _fsync_put(lambda p: save_file(t, p), os.path.join(dst, "scales", f"{k}.safetensors"))
    def jl(p):
        with open(p, "w") as fh: fh.writelines(json.dumps(recs[k]["c9"]) + "\n" for k in keys)
    _fsync_put(jl, os.path.join(dst, "c9.jsonl"))
    open(os.path.join(dst, ".done"), "w").close()


def levelargs(level, cfg):
    """an ourq level's settings (stamp.json's last "args", rad-exl3-mix v3/sib27/levelargs.py's source) as rxq quant
    arguments under cfg: -> (argv list, problems). A level whose settings the config cannot express (another sigma, no
    qtf2, ...) gives problems: a requant of its tensors with rxq would not reproduce them (the no-op gate would fail)."""
    st = json.load(open(os.path.join(level, "stamp.json")))
    A = dict(st[-1]["args"]); q = cfg["quant"]; bad = []
    known = {"K", "frac_ext", "codebook", "seed_offset", "sigma_reg", "hout", "hout_diag", "sigma_reg_in", "sigma_in_sweep",
             "gain", "sigma_reg_out", "out_scales_off", "head_perm", "scaled_hessian", "qtf2", "head_mean_row", "hin", "wpre",
             "only", "tile_cd", "tile_cd_families", "hout_blocks"}
    if set(A) - known: bad.append(f"unknown ourq args {sorted(set(A) - known)}")
    if A.get("tile_cd"): bad.append(f"made with --tile-cd {A['tile_cd']} (tile CD settings are rxq's; compare by hand)")
    for k in ("hin", "wpre", "only", "hout_diag", "out_scales_off", "sigma_in_sweep", "head_perm"):
        if A.get(k): bad.append(f"made with --{k.replace('_', '-')} {A[k]}")
    K = A["K"]; K = int(K) if float(K).is_integer() else float(K)
    cb = A.get("codebook", "mul1")
    if bool(A.get("qtf2")) != bool(q["qtf2"]): bad.append(f"qtf2 {A.get('qtf2')} vs settings {q['qtf2']}")
    if bool(A.get("scaled_hessian")) != bool(q["scaled_hessian"]): bad.append(f"scaled_hessian {A.get('scaled_hessian')}")
    if A.get("gain", 1.0) not in (None, 1.0): bad.append(f"gain {A['gain']}")
    s = C.quant_settings(cfg, K, cb, qwen35.trunk_keys(1)[0], A.get("seed_offset", 0) or 0)
    if A.get("sigma_reg") != q["sigma_reg"]: bad.append(f"sigma_reg {A.get('sigma_reg')} vs {q['sigma_reg']}")
    if A.get("sigma_reg_in") != s["sigma_reg_in"]: bad.append(f"sigma_reg_in {A.get('sigma_reg_in')} vs {s['sigma_reg_in']} at K{K}")
    argv = ["-K", str(K), "--codebook", cb, "--seed-offset", str(A.get("seed_offset", 0) or 0)]
    if A.get("hout"):
        if not C.hout_applies(cfg, K): argv += ["--set", "inputs.hout.enable=true", "--set", f"inputs.hout.apply_k_min={float(K)}"]
        so = A.get("sigma_reg_out")
        if so is not None and so != cfg["inputs"]["hout"]["sigma_reg_out"]: argv += ["--set", f"inputs.hout.sigma_reg_out={so}"]
        if so is None: bad.append("--hout without --sigma-reg-out (ourq then uses sigma_reg for it)")
        argv += ["--hout", A["hout"]]
        hb = A.get("hout_blocks")                    # ourq --hout-blocks (w81, #115); unset = the dense path (every level before 11-08)
        if (hb or None) != C.hout_ldlq(cfg):
            argv += ["--set", "inputs.hout.ldlq=blocks", "--set", f"inputs.hout.schedule={hb}"] if hb else \
                    ["--set", "inputs.hout.ldlq=dense"]
    elif C.hout_applies(cfg, K):
        argv += ["--set", "inputs.hout.enable=false"]
    if C.block_order(cfg) is not None:               # #117 (11-08): ourq never reorders the readers' blocks
        argv += ["--set", "quant.block_order.enable=false"]
    return argv, bad
