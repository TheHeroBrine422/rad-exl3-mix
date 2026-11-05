# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Shared by quant_layer and quant_head: parameters, the settings stamp, the staging of inputs into the layout
quantize/ourq.py reads (RXM_CKPT / RXM_WORK / RXM_TEMPLATE), the ourq command line, and the shard structure check."""
import json, os, re, shutil, time
from common.util import stamp, st_header, write_json_atomic
from . import qwen

ID = r"[a-z0-9][a-z0-9._-]{0,63}"
SETTINGS = {"type": "object", "required": ["sigma_reg", "sigma_reg_in", "sigma_reg_out", "scaled_hessian", "qtf2",
                                           "hout", "seed_offset"], "properties": {
    "sigma_reg": {"type": "number", "minimum": 0, "maximum": 1},
    "sigma_reg_in": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "sigma_reg_out": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "scaled_hessian": {"type": "boolean"},
    "qtf2": {"type": "boolean"},
    "hout": {"type": ["string", "null"], "pattern": ID, "description": "H_out snapshot id (models/<m>@<rev>/hout/<id>/)"},
    "seed_offset": {"type": "integer", "minimum": 0, "maximum": 1000}}}


def schema(with_layer):
    props = {"model": {"type": "string", "pattern": ID}, "revision": {"type": "string", "pattern": ID},
             "arch": {"type": "string", "enum": ["qwen3_5"]},
             "K": {"type": "number", "minimum": 0.125, "maximum": 8, "multipleOf": 0.125},
             "codebook": {"type": "string", "enum": ["mul1", "i8", "i4", "i8c"]},
             "level": {"type": "string", "pattern": ID},
             "settings": SETTINGS, "stamp": {"type": "string", "pattern": "[0-9a-f]{16}"}}
    req = ["model", "revision", "arch", "K", "codebook", "level", "settings", "stamp"]
    if with_layer:
        props["layer"] = {"type": "integer", "minimum": 0, "maximum": 255}; req.append("layer")
    return {"type": "object", "required": req, "properties": props}


def settings_stamp(p):
    """one per ladder level: what the quantizer is asked to do, layer excluded"""
    return stamp({k: p[k] for k in ("model", "revision", "arch", "K", "codebook", "settings")})


def check_extra(p):
    if p["stamp"] != settings_stamp(p): raise ValueError("settings stamp does not match the parameters")
    K = p["K"]
    if K < 1 and not p["settings"]["qtf2"]: raise ValueError("K < 1 needs qtf2")
    if K != int(K) and (K * 16) % 2: raise ValueError("fractional K must be KA + n/16 with n even")
    if p["settings"]["qtf2"] and p["codebook"] not in ("mul1", "i8", "i8c"): raise ValueError("qtf2 needs mul1, i8 or i8c")


def root(p): return f"models/{p['model']}@{p['revision']}"


def inputs_for(p, plan, wfile):
    r = root(p)
    req = [f"{r}/config.json", f"{r}/weights/{wfile}"] + [f"{r}/hessians/{h}.safetensors" for h, _ in plan]
    opt = []
    if p["settings"]["hout"]:
        alt = "|".join(re.escape(k) for k in qwen.keys(plan))
        opt.append(re.escape(f"{r}/hout/{p['settings']['hout']}/") + f"(?:{alt})\\.safetensors")
    return req, opt


def shard_cap(p, n_params):
    return 32 * 2 ** 20 + int(n_params * p["K"] / 8 * 1.05)


LOG_CAP = 1 << 20
SMALL = {"proxy.jsonl": 256 << 10, "stamp.json": 1 << 20, "job.json": 64 << 10, "ourq.log": LOG_CAP, "rowerr.npz": 16 << 20}


def stage(p, ctx, plan, wfile):
    """symlink the cached inputs into a fixed layout (fixed paths keep ourq's code stamp the same for every job)"""
    r = root(p); s = ctx.stage_dir
    if os.path.exists(s): shutil.rmtree(s)
    ck, hs, ho = f"{s}/ckpt", f"{s}/work/challenge/hessians/calib", f"{s}/hout"
    for d in (ck, hs, ho, f"{s}/cwd"): os.makedirs(d)
    os.symlink(ctx.input(f"{r}/weights/{wfile}"), f"{ck}/{wfile}")
    wm = {f"{k}.weight": wfile for k in qwen.keys(plan)}
    with open(f"{ck}/model.safetensors.index.json", "w") as f: json.dump({"metadata": {}, "weight_map": wm}, f)
    for h, _ in plan: os.symlink(ctx.input(f"{r}/hessians/{h}.safetensors"), f"{hs}/{h}.safetensors")
    os.symlink(ctx.input(f"{r}/config.json"), f"{s}/template.json")
    nh = 0
    if p["settings"]["hout"]:
        pre = f"{r}/hout/{p['settings']['hout']}/"
        for n in ctx.input_names:
            if n.startswith(pre): os.symlink(ctx.input(n), f"{ho}/{n[len(pre):]}"); nh += 1
    return {"ckpt": ck, "work": f"{s}/work", "template": f"{s}/template.json", "hout": ho if nh else None, "cwd": f"{s}/cwd"}


def ourq_argv(p, cfg, out, st, which):
    S = p["settings"]
    K = p["K"]
    pre = [cfg.python] + ([cfg.capwrap, repr(float(cfg.vram_cap_gb))] if cfg.vram_cap_gb else [])
    a = pre + [cfg.ourq, "-K", str(int(K)) if float(K).is_integer() else repr(float(K)),
         "-o", out, "--device", str(cfg.device), "--codebook", p["codebook"], "--save-wq"] + which
    if S["qtf2"] or p["codebook"] != "mul1" or p["K"] != int(p["K"]):
        a += ["--frac-ext", cfg.exl3_src]
    if S["qtf2"]: a.append("--qtf2")
    if S["scaled_hessian"]: a.append("--scaled-hessian")
    a += ["--sigma-reg", repr(float(S["sigma_reg"]))]
    if S["sigma_reg_in"] is not None: a += ["--sigma-reg-in", repr(float(S["sigma_reg_in"]))]
    if st["hout"]:
        a += ["--hout", st["hout"]]
        if S["sigma_reg_out"] is not None: a += ["--sigma-reg-out", repr(float(S["sigma_reg_out"]))]
    if S["seed_offset"]: a += ["--seed-offset", str(S["seed_offset"])]
    return a


def run_ourq(p, ctx, plan, wfile, which, shard_name):
    t0 = time.time()
    st = stage(p, ctx, plan, wfile)
    out = ctx.out_dir
    shard = os.path.join(out, shard_name)
    argv = ourq_argv(p, ctx.cfg, out, st, which)
    if not (os.path.exists(shard) and os.path.exists(os.path.join(out, "rowerr.npz"))):
        for f in os.listdir(out):              # a killed earlier try: start clean (ourq appends to proxy.jsonl)
            q = os.path.join(out, f)
            shutil.rmtree(q) if os.path.isdir(q) else os.remove(q)
        env = dict(ctx.cfg.env)
        env.update({"RXM_CKPT": st["ckpt"], "RXM_WORK": st["work"], "RXM_TEMPLATE": st["template"]})
        rc = ctx.run_cmd(argv, env=env, cwd=st["cwd"], log=os.path.join(out, "ourq.log"))
        if rc != 0: raise RuntimeError(f"ourq.py exit {rc} (see ourq.log)")
        t1 = time.time()
        pack_rowerr(ctx, plan, st, out)
        t_rowerr = time.time() - t1
    else: t_rowerr = None
    hdr = st_header(shard)
    ls = re.findall(r"^layer \d+: (\d+)s", open(os.path.join(out, "ourq.log")).read(), re.M)
    meta = {"type": ctx.jtype, "version": ctx.jversion, "settings_stamp": p["stamp"], "level": p["level"],
            "ourq_stamp": (hdr.get("__metadata__") or {}).get("ourq_stamp"), "image": ctx.cfg.image,
            "gpu": ctx.cfg.gpu_name, "seconds": round(time.time() - t0, 1),
            "ourq_layer_s": int(ls[-1]) if ls else None, "rowerr_s": None if t_rowerr is None else round(t_rowerr, 1), "argv": argv[argv.index(ctx.cfg.ourq):]}
    write_json_atomic(os.path.join(out, "job.json"), meta)
    for f in ("model.safetensors.index.json", "config.json"):        # the server rebuilds them on export
        if os.path.exists(os.path.join(out, f)): os.remove(os.path.join(out, f))
    lg = os.path.join(out, "ourq.log")
    if os.path.getsize(lg) > LOG_CAP:
        with open(lg, "rb") as f: f.seek(-LOG_CAP + 4096, 2); tail = f.read()
        with open(lg, "wb") as f: f.write(b"[truncated]\n" + tail)
    return {k: meta[k] for k in ("ourq_stamp", "seconds", "ourq_layer_s", "rowerr_s")}


def pack_rowerr(ctx, plan, st, out):
    """the per-output-row claim (rowerr.npz, jobs/rowerr_tool.py) from ourq --save-wq's dequantized weights; the
    weights themselves stay local and are deleted"""
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rowerr_tool.py")
    pre = [ctx.cfg.python] + ([ctx.cfg.capwrap, repr(float(ctx.cfg.vram_cap_gb))] if ctx.cfg.vram_cap_gb else [])
    argv = pre + [tool, st["ckpt"], f"{st['work']}/challenge/hessians/calib", os.path.join(out, "wq"),
            os.path.join(out, "rowerr.npz"), str(ctx.cfg.device)] + [f"{k}:{h}" for h, ks in plan for k in ks]
    rc = ctx.run_cmd(argv, env=dict(ctx.cfg.env), cwd=st["cwd"], log=os.path.join(out, "ourq.log"))
    if rc != 0: raise RuntimeError(f"rowerr_tool exit {rc} (see ourq.log)")
    shutil.rmtree(os.path.join(out, "wq"))


def expected_tensors(p, plan, weights_hdr):
    """{tensor name: (dtype, shape)} a shard must hold exactly"""
    K = p["K"]; words = int(round(16 * K))
    exp = {}
    for k in qwen.keys(plan):
        n, kin = weights_hdr[f"{k}.weight"]["shape"]          # checkpoint (out, in)
        exp[f"{k}.trellis"] = ("I16", [kin // 16, n // 16, words])
        exp[f"{k}.suh"] = ("F16", [kin]); exp[f"{k}.svh"] = ("F16", [n])
        exp[f"{k}.mul1"] = ("I32", [])
    return exp


def check_shard(p, plan, shard, weights_file):
    """structure of a returned shard against the bf16 weights it was made from; returns the shard header"""
    hdr = st_header(shard)
    exp = expected_tensors(p, plan, st_header(weights_file))
    got = {k: (v["dtype"], v["shape"]) for k, v in hdr.items() if k != "__metadata__"}
    if set(got) != set(exp):
        raise ValueError(f"shard keys: missing {sorted(set(exp) - set(got))[:3]}, extra {sorted(set(got) - set(exp))[:3]}")
    for k, (dt, sh) in exp.items():
        if tuple(got[k][0:1]) != (dt,) or list(got[k][1]) != sh: raise ValueError(f"{k}: {got[k]} != {(dt, sh)}")
    if not (hdr.get("__metadata__") or {}).get("ourq_stamp"): raise ValueError("shard has no ourq_stamp")
    return hdr


def check_log(p, plan, proxy_path, job_path):
    """the proxy log covers every tensor once with the job's settings; job.json carries the job's stamp"""
    rows = [json.loads(l) for l in open(proxy_path) if l.strip()]
    keys = qwen.keys(plan)
    if sorted(r["key"] for r in rows) != sorted(keys): raise ValueError("proxy.jsonl: keys differ from the layer plan")
    for r in rows:
        if float(r["K"]) != float(p["K"]) or r["codebook"] != p["codebook"] or bool(r["scaled_hessian"]) != p["settings"]["scaled_hessian"]:
            raise ValueError(f"proxy.jsonl {r['key']}: settings differ from the job")
        if not (isinstance(r["proxy_err"], float) and 0 <= r["proxy_err"] < 10): raise ValueError(f"{r['key']}: bad proxy_err")
    meta = json.load(open(job_path))
    if meta.get("settings_stamp") != p["stamp"]: raise ValueError("job.json: settings stamp differs")
    return rows, meta
