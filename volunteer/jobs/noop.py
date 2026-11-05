# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""noop: sleep, read its inputs, write a deterministic file. For tests of the queue, the cache and the uploads."""
import hashlib, os, time

NAME, VERSION, IMPLEMENTED = "noop", 1, True
SCHEMA = {"type": "object", "required": ["seconds", "size", "seed"], "properties": {
    "seconds": {"type": "number", "minimum": 0, "maximum": 3600},
    "size": {"type": "integer", "minimum": 0, "maximum": 64 << 20},
    "seed": {"type": "integer", "minimum": 0, "maximum": 2 ** 31},
    "inputs": {"type": "array", "maxItems": 16, "items": {"type": "string", "pattern": r"test/[A-Za-z0-9][A-Za-z0-9._-]{0,63}"}},
    "fail": {"type": "boolean"},
    "tag": {"type": "string", "pattern": r"[A-Za-z0-9._-]{0,64}"}}}


def requires(p): return {"gpu": False, "ram_gb": 0, "vram_gb": 0}
def inputs(p): return list(p.get("inputs", [])), []
def outputs(p): return {"out.bin": p["size"]}
def required_outputs(p): return {"out.bin"}


def content(seed, size):
    out, i = bytearray(), 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{i}".encode()).digest(); i += 1
    return bytes(out[:size])


def run(p, ctx):
    h = hashlib.sha256()
    for n in inputs(p)[0]:
        with open(ctx.input(n), "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    t_end = time.time() + p["seconds"]
    while time.time() < t_end:
        if ctx.cancelled(): raise InterruptedError("cancelled")
        time.sleep(min(0.2, max(0.0, t_end - time.time())))
    if p.get("fail"): raise RuntimeError("noop: asked to fail")
    path = os.path.join(ctx.out_dir, "out.bin")
    with open(path + ".part", "wb") as f: f.write(content(p["seed"], p["size"]))
    os.replace(path + ".part", path)
    return {"inputs_sha256": h.hexdigest(), "slept": p["seconds"]}


def accept(p, files, result, data_path=None):
    if os.path.getsize(files["out.bin"]) != p["size"]: raise ValueError("out.bin: wrong size")
