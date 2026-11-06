# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""swj (C10): gate / up output scales refit jointly through the SwiGLU on frozen codes (#85; 27B #104 PASS at K2.5).

Ported here (tests/check_cpu.py: byte-identical to the originals):
- acc / fit / resid: the per-channel grid sums and the joint fit, rad-exl3-mix proxy2b/exp/w02-fixedscale/fsutil.py
  swiglu_acc / swiglu_fit / swiglu_resid (the same ops in the same order);
- apply_level: the container form, rad-exl3-mix v3/yaqa/swj27/swjapply.py (svh *= c, suh *= r, rounded to f16 as
  stored; everything else byte for byte the level's).
Not ported yet (called in place): the level pass that feeds acc() -- the bf16 27B forward in staged layer passes with
the MLP hooks and the fused Triton grid (v3/yaqa/swj27/swjfit27.py, swjtri.py). It is a whole-model job (needs the
bf16 forward over ~0.93M qad tokens: ~1.5 card-h on the R9700 per two levels), not a per-tensor one; DESIGN.md
"Level passes". fit_level() runs it as a subprocess with the settings file's grid / pool / token count.

python -m rxq.swj apply LEVEL SCALES DST"""
import hashlib, json, os, shutil, subprocess, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def acc(s, aa, gq, uq, grid):
    """add one block of positions: aa = silu(g) u (target), gq / uq the quantized gate / up outputs, [T, J];
    s = [S_ap (G, J), S_pp (G, J), S_aa (J)] float64"""
    s[2] += aa.pow(2).sum(0).double()
    for i in range(len(grid)):
        p = torch.nn.functional.silu(grid[i] * gq) * uq
        s[0][i] += (aa * p).sum(0).double(); s[1][i] += p.pow(2).sum(0).double()


def fit(s, i1):
    """-> (grid index of c_g, c_u there) per channel for the joint fit, and c_u at c_g = grid[i1] (= 1): up-only"""
    sap, spp, saa = s
    cu = torch.where(spp > 1e-30, sap / spp.clamp(min=1e-30), torch.ones_like(spp))
    gi = (saa[None] - sap * cu).argmin(0)
    return gi, cu.gather(0, gi[None])[0], cu[i1]


def resid(s, gi, cu):
    """(sum over channels of sum_t (a - cu p(c_g))^2, sum a^2) for given grid indices and c_u"""
    sap, spp, saa = s
    a_ = sap.gather(0, gi[None])[0]; p_ = spp.gather(0, gi[None])[0]
    return float((saa - 2 * cu * a_ + cu * cu * p_).sum()), float(saa.sum())


def apply_level(src, sc, dst):
    """DST = LEVEL with suh *= r and svh *= c of every tensor that has SCALES/scales/<key>.safetensors"""
    if os.path.exists(f"{dst}/.done"): print(f"{dst} done"); return 0
    os.makedirs(dst, exist_ok=True)
    sd = f"{sc}/scales"
    keys = sorted(f[:-len(".safetensors")] for f in os.listdir(sd) if f.endswith(".safetensors"))
    h = hashlib.sha256()
    for k in keys: h.update(k.encode()); h.update(open(f"{sd}/{k}.safetensors", "rb").read())
    meta = json.dumps({"src": os.path.abspath(src), "scales": os.path.abspath(sc), "scales_sha256": h.hexdigest()[:16]})
    for f in ("config.json", "model.safetensors.index.json", "stamp.json"):
        if os.path.exists(f"{src}/{f}"): shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    used, edited = set(), 0
    for f in sorted(os.listdir(src)):
        if not (f.startswith("q-") and f.endswith(".safetensors")): continue
        with safe_open(f"{src}/{f}", "pt", device="cpu") as fh:
            t = {k: fh.get_tensor(k) for k in fh.keys()}; md = dict(fh.metadata() or {})
        for k in [k for k in t if k[:-4] in keys and k[-4:] in (".suh", ".svh")]:
            with safe_open(f"{sd}/{k[:-4]}.safetensors", "pt") as s: mult = s.get_tensor("r" if k.endswith(".suh") else "c")
            assert mult.shape == t[k].shape and t[k].dtype == torch.float16, (k, tuple(mult.shape), tuple(t[k].shape), t[k].dtype)
            if not bool((mult == 1).all()): t[k] = (t[k].float() * mult.float()).half(); edited += 1
            used.add(k[:-4])
        if os.path.exists(f"{dst}/{f}"): continue
        md["swj27"] = meta
        save_file(t, f"{dst}/{f}.part", metadata=md); os.replace(f"{dst}/{f}.part", f"{dst}/{f}")
    miss = sorted(set(keys) - used)
    if miss: raise SystemExit(f"{len(miss)} scales without a tensor in {src} (first {miss[0]}): {dst} left without .done")
    open(f"{dst}/.done", "w").close()
    print(f"SWJAPPLY DONE {len(used)} tensors ({edited} scale vectors changed) -> {dst}")
    return edited


def fit_level(cfg, model, meta, out, sources, scratch, rxm=None, extra=()):
    """the level pass (swjfit27.py, in place): sources = {name: level dir}"""
    rxm = rxm or os.environ.get("RXQ_RXM", "/workspace/rad-exl3-mix")
    s = cfg["post"]["swj"]
    argv = [sys.executable, "-I", f"{rxm}/v3/yaqa/swj27/swjfit27.py", model, meta, out, "--scratch", scratch, "--grid", s["grid"],
            "--no-cache"] + (["--fused"] if s["fused"] else [])
    for n, d in sources.items(): argv += ["--src", f"{n}={d}:level"]
    return subprocess.call(argv + list(extra))


if __name__ == "__main__":
    if sys.argv[1:2] == ["apply"]: apply_level(*sys.argv[2:5])
    else: raise SystemExit(__doc__)
