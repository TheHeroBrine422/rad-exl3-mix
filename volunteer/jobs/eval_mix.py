# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""eval_mix (v0: interface only, IMPLEMENTED = False; workers refuse it and the server never leases it to them).

A mix = a container recipe (which level's shard each tensor comes from, models/<m>@<rev>/mixes/<recipe>.json) built
on the worker from cached ladder shards (levels/<level>/q-*.safetensors, fetched like any input); scored on a
golden slice (goldens/<slice>/: tokens + bf16 teacher top-k logits, built from calib only) -> kl.npy: one float32
per scored position, rows in the slice's order, plus a summary. The spot check re-scores a random sample of
documents on our hardware."""
from .quant_common import ID

NAME, VERSION, IMPLEMENTED = "eval_mix", 0, False
SCHEMA = {"type": "object", "required": ["model", "revision", "recipe", "golden", "kv"], "properties": {
    "model": {"type": "string", "pattern": ID}, "revision": {"type": "string", "pattern": ID},
    "recipe": {"type": "string", "pattern": ID}, "golden": {"type": "string", "pattern": ID},
    "kv": {"type": "string", "enum": ["bf16"]},
    "max_positions": {"type": "integer", "minimum": 1, "maximum": 10_000_000}}}


def requires(p): return {"gpu": True, "ram_gb": 16, "vram_gb": 24}


def inputs(p):
    r = f"models/{p['model']}@{p['revision']}"
    return ([f"{r}/config.json", f"{r}/mixes/{p['recipe']}.json", f"goldens/{p['golden']}/tokens.npy",
             f"goldens/{p['golden']}/teacher.safetensors"],
            [rf"{r}/levels/[a-z0-9._-]+/q-(?:L\d\d|head)\.safetensors"])


def outputs(p): return {"kl.npy": 4 * 10_000_000 + 4096, "summary.json": 64 << 10}
def required_outputs(p): return {"kl.npy", "summary.json"}


def run(p, ctx):
    raise NotImplementedError("eval_mix is not implemented in v0")


def accept(p, files, result, data_path):
    raise ValueError("eval_mix is not implemented in v0")
