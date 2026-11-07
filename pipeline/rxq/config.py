# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The settings file (config/v3.toml) and what a quant job derives from it.

settings_stamp(cfg): sha256 of the canonical JSON of every result-affecting section (meta, inputs, quant, post), first
16 hex digits. A level's jobs all carry it; assembly refuses mixed stamps.
quant_settings(cfg, K, codebook, key): the resolved per-tensor settings (ourq's arguments for that tensor)."""
import copy, hashlib, json, os

try:
    import tomllib
except ModuleNotFoundError:                  # python < 3.11
    tomllib = None

from . import qwen35

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "config", "v3.toml")
STAMPED = ("meta", "inputs", "quant", "post")


def load(path=None):
    p = path or os.environ.get("RXQ_CONFIG") or DEFAULT
    if tomllib is None: raise SystemExit("rxq needs python >= 3.11 (tomllib) to read the settings file")
    with open(p, "rb") as f: cfg = tomllib.load(f)
    validate(cfg)
    cfg["_path"] = os.path.abspath(p)
    return cfg


def validate(cfg):
    q = cfg["quant"]
    assert q["out_scales"] == "always", "only out_scales = always is implemented (#48)"
    assert q["drift"] in ("interp", "exact")
    assert cfg["inputs"]["hin"]["blend"] in ("none", "xcorp")
    assert cfg["inputs"]["hout"]["estimator"] in ("bn", "dy")
    if q["tile_cd"]["enable"] and cfg["inputs"]["hout"]["enable"]:
        raise ValueError("tile_cd and H_out together: tilecd.py refuses ldlq_2hess (w50)")
    if q["codebook"]["hash_codebook"]: raise ValueError("#100 hash codebook: not implemented")
    if cfg["post"]["healing"]["enable"]: raise ValueError("healing: not part of the pipeline yet (deferred, #89)")


def settings_stamp(cfg):
    body = {k: cfg[k] for k in STAMPED}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def codebook_for(cfg, K):
    """the ladder planner's default codebook of level K (job specs carry their own)"""
    K = float(K)
    if K in cfg["quant"]["codebook"]["mul1_levels"]: return "mul1"
    for r in cfg["quant"]["codebook"]["by_level"]:
        if r["k_min"] <= K <= r["k_max"]: return r["codebook"]
    return "mul1"


def quant_settings(cfg, K, codebook, key, seed_offset=0):
    """ourq.py's arguments for one tensor, resolved from the settings file. Keys mirror ourq's argparse names so a
    job's record reads like an ourq stamp (levelargs.py)."""
    q = cfg["quant"]
    _, _, _, hname, _ = qwen35.locate(key)
    inp = qwen35.is_input_group(hname)
    s = {"K": K, "codebook": codebook, "qtf2": q["qtf2"], "scaled_hessian": q["scaled_hessian"],
         "sigma_reg": q["sigma_reg"], "sigma_reg_in": q["sigma_reg_in"], "seed_offset": seed_offset,
         "drift": q["drift"], "block_trace": q["block_trace"], "refit": q["refit"]}
    if q["apply_sub1_damping"] and float(K) < 1 and inp: s["sigma_reg_in"] = q["sigma_reg_in_sub1"]
    if q["gain"] != 1.0: s["gain"] = q["gain"]
    ho = cfg["inputs"]["hout"]
    if ho["enable"] and float(K) <= ho["apply_k_max"]:
        s["hout"] = ho["dir"]; s["sigma_reg_out"] = ho["sigma_reg_out"]
    cd = q["tile_cd"]
    if cd["enable"] and qwen35.family(key) in cd["families"]:
        s["tile_cd"] = cd["sweeps"]
    return s


def with_overrides(cfg, **kv):
    """a copy with dotted-path overrides, e.g. with_overrides(cfg, **{"inputs.hout.enable": True})"""
    c = copy.deepcopy(cfg)
    for path, v in kv.items():
        d = c
        *head, last = path.split(".")
        for h in head: d = d[h]
        assert last in d, f"unknown setting {path}"
        d[last] = v
    validate(c)
    return c
