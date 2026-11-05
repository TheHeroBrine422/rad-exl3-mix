# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""quant_head: the LM head of a ladder level with quantize/ourq.py --only-head (~22 GB host RAM at the 27B: the server
leases it only to workers that declared ram_gb >= 24). Inputs: weights/head.safetensors (lm_head.weight), the
lm_head.input Hessian, config.json. Outputs as quant_layer with q-head.safetensors."""
from . import qwen, quant_common as Q

NAME, VERSION, IMPLEMENTED = "quant_head", 2, True
SCHEMA = Q.schema(with_layer=False)
check_extra = Q.check_extra
HEAD_PARAMS_MAX = 1_300_000_000          # 248k x 5120


def requires(p): return {"gpu": True, "ram_gb": 24, "vram_gb": 24}
def plan(p): return qwen.head_plan(p["arch"])
def inputs(p): return Q.inputs_for(p, plan(p), "head.safetensors")
def outputs(p): return {"q-head.safetensors": Q.shard_cap(p, HEAD_PARAMS_MAX), **Q.SMALL}
def required_outputs(p): return {"q-head.safetensors", "proxy.jsonl", "stamp.json", "job.json", "rowerr.npz"}


def run(p, ctx):
    return Q.run_ourq(p, ctx, plan(p), "head.safetensors", ["--only-head"], "q-head.safetensors")


def accept(p, files, result, data_path):
    Q.check_shard(p, plan(p), files["q-head.safetensors"], data_path(f"{Q.root(p)}/weights/head.safetensors"))
    Q.check_log(p, plan(p), files["proxy.jsonl"], files["job.json"])
