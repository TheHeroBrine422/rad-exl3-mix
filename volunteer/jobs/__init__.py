# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The built-in job types. This registry is the whole list of things a worker can be asked to do: the server sends a
type name and a parameter object, never code. A worker refuses a type it does not know, parameters that do not match
the type's schema exactly, and input files outside what the type allows for those parameters.

Every job module defines
    NAME, VERSION, SCHEMA       the type, its version (bumped on any change of meaning), its parameter schema
    IMPLEMENTED                 False: the interface exists, the worker refuses to run it (eval_mix in v0)
    requires(p)                 {"gpu": bool, "ram_gb": float, "vram_gb": float}: the server leases only to workers
                                that declared at least that much
    inputs(p)                   (required relpaths, [regex for optional relpaths]) under the server's data root
    outputs(p)                  {file name: max bytes}: the only files a worker uploads and the server accepts
    required_outputs(p)         the subset that must be present at completion
    run(p, ctx) -> dict         the work; ctx is worker.worker.JobContext; returns a small JSON result
    accept(p, files, result)    cheap structural check on the server at completion (files: name -> staged path);
                                raises ValueError to reject. The real verification is checker/."""
import re
from common.schema import validate, SchemaError
from . import noop, quant_layer, quant_head, eval_mix

REGISTRY = {m.NAME: m for m in (noop, quant_layer, quant_head, eval_mix)}


class Rejected(ValueError):
    """a job this worker / server will not run: unknown type, bad parameters, bad inputs"""


def get(jtype):
    m = REGISTRY.get(jtype) if isinstance(jtype, str) else None
    if m is None: raise Rejected(f"unknown job type {jtype!r}")
    return m


def check_params(jtype, params):
    m = get(jtype)
    try:
        validate(params, m.SCHEMA)
        if hasattr(m, "check_extra"): m.check_extra(params)
    except (SchemaError, ValueError) as e:
        raise Rejected(f"{jtype}: {e}")
    return m


def check_inputs(jtype, params, names):
    """the server's input list for a job must be exactly the type's required files plus optional ones it allows"""
    m = get(jtype)
    req, opt = m.inputs(params)
    names = list(names)
    if len(set(names)) != len(names): raise Rejected("duplicate input")
    missing = [r for r in req if r not in names]
    if missing: raise Rejected(f"inputs missing {missing[:3]}")
    for n in names:
        if n in req: continue
        if not any(re.fullmatch(rx, n) for rx in opt): raise Rejected(f"input {n!r} not allowed for {jtype}")
    return m
