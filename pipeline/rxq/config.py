# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The settings file (config/v3.toml) and what a quant job derives from it.

settings_stamp(cfg): sha256 of the canonical JSON of every result-affecting section (meta, inputs, quant, post), first
16 hex digits. A level's jobs all carry it; assembly refuses mixed stamps.
quant_settings(cfg, K, codebook, key): the resolved per-tensor settings (ourq's arguments for that tensor).
Settings the engine reads per level: H_out between inputs.hout.apply_k_min and apply_k_max (hout_applies), the sub-1
input damping (quant.apply_sub1_damping: sigma_reg_in_sub1 for the layer-input group below K1), tile CD (tile_cd)."""
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
    q = cfg["quant"]; ho = cfg["inputs"]["hout"]
    assert q["out_scales"] == "always", "only out_scales = always is implemented (#48)"
    assert q["drift"] in ("interp", "exact")
    assert cfg["inputs"]["hin"]["blend"] in ("none", "xcorp")
    assert ho["estimator"] in ("bn", "dy")
    assert ho["apply_k_min"] <= ho["apply_k_max"], "inputs.hout: apply_k_min > apply_k_max"
    if q["codebook"]["hash_codebook"]: raise ValueError("#100 hash codebook: not implemented")
    if cfg["post"]["healing"]["enable"]: raise ValueError("healing: not part of the pipeline yet (deferred, #89)")
    a = cfg["post"]["asym"]
    assert a["scope"] in ("per-mix", "per-level"), "post.asym.scope: per-mix | per-level"
    if a["enable"] and a["scope"] == "per-level":
        raise ValueError("post.asym: per-level writers next to siblings of another K fail (#111); per-mix only")
    c9 = cfg["post"]["c9"]
    assert set(c9["alpha"]) <= {"attn_o", "gdn_out", "down"}, c9["alpha"]
    assert 0 < c9["e_clip"] < 1


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


def hout_applies(cfg, K):
    """H_out on for level K: the switch and apply_k_min <= K <= apply_k_max (H27: K >= 4 only, 11-07)"""
    ho = cfg["inputs"]["hout"]
    return bool(ho["enable"]) and ho["apply_k_min"] <= float(K) <= ho["apply_k_max"]


def tile_cd(cfg):
    """engine.setup's tile_cd argument: None or (sweeps, families, module path)"""
    cd = cfg["quant"]["tile_cd"]
    return (cd["sweeps"], list(cd["families"]), cd["module"]) if cd["enable"] else None


def c9_injob(cfg, key, hin=None, wpre=None):
    """C9's alpha for KEY when its quant job computes the C9 scales (post.c9 on, the key a writer with alpha != 0, plain
    writers: no asym (then C9 runs per mix on the final writers against W F / G: rxq.post c9 --dn), no hin / wpre
    job); else None"""
    c9 = cfg["post"]["c9"]
    if not c9["enable"] or cfg["post"]["asym"]["enable"] or hin or wpre: return None
    fam = {"attn_output": "attn_o", "ssm_out": "gdn_out", "ffn_down": "down"}.get(qwen35.family(key))
    al = c9["alpha"].get(fam, 0.0) if fam else 0.0
    return float(al) if al else None


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
    if hout_applies(cfg, K):
        s["hout"] = cfg["inputs"]["hout"]["dir"]; s["sigma_reg_out"] = cfg["inputs"]["hout"]["sigma_reg_out"]
    cd = q["tile_cd"]
    if cd["enable"] and qwen35.family(key) in cd["families"] and float(K) <= cd["k_max"]:
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
