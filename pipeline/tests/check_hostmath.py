#!/usr/bin/env python3
"""check_hostmath.py SPEC OUT [--device N] -- the CPU BLAS / LAPACK call sites of a quant job: the spec's job runs
in-process (as `rxq job`: VRAM_CAP_GB applies) with torch's BLAS / LAPACK entry points wrapped; a call whose tensor
operands are all on the CPU, one of them floating point with 256+ elements, goes through the host's math library
(OpenBLAS / MKL), whose rounding differs between hosts with the same card (GitHub issue 2):
  hostmath  no such call site (FAIL lists each: function, count, largest operand, file:line and its two callers)
  untouched the traced job's t-* / c9-* files equal a plain `python -m rxq job` run of the same spec in a fresh
            process (tensors plus metadata): the wrapping does not change the math
Outputs in OUT/traced and OUT/plain (fresh dirs). --device cpu with RXQ_CPU_STANDIN=1 (the stand-in runs everything on
the CPU: hostmath must FAIL there). Last line: "HOSTMATH CHECKS n/m PASS"."""
import argparse, collections, functools, json, os, shutil, subprocess, sys, traceback

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PIPE)
from rxq import job  # noqa: E402
from rxq.__main__ import _cap, _dev  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("spec"); ap.add_argument("out"); ap.add_argument("--device", type=_dev, default=0)
a = ap.parse_args()
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


# ---- the wrapping: same math, every CPU float call of 256+ elements counted by site
seen, big = collections.Counter(), {}
ME = os.path.abspath(__file__)


def site():
    fr = [f for f in traceback.extract_stack()[:-2][::-1] if os.path.abspath(f.filename) != ME and "/torch/" not in f.filename]
    if not fr: return "?"
    loc = lambda f: f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}"
    return f"{loc(fr[0])}  {(fr[0].line or '').strip()[:60]}  <- " + " <- ".join(loc(f) for f in fr[1:3])


def wrap(owner, name, label):
    fn = getattr(owner, name, None)
    if fn is None or isinstance(fn, type) or not callable(fn): return

    @functools.wraps(fn)
    def w(*x, **k):
        ts = [t for t in list(x) + list(k.values()) if isinstance(t, torch.Tensor)]
        if ts and all(t.device.type == "cpu" for t in ts) and any(t.is_floating_point() and t.numel() >= 256 for t in ts):
            s = (label, site()); seen[s] += 1; big[s] = max(big.get(s, 0), max(t.numel() for t in ts))
        return fn(*x, **k)
    try: setattr(owner, name, w)
    except (AttributeError, TypeError): pass


BLAS = ("matmul", "mm", "bmm", "mv", "dot", "vdot", "addmm", "addmv", "addbmm", "baddbmm", "addr", "ger", "outer", "inner", "einsum",
        "tensordot", "chain_matmul", "cholesky", "cholesky_inverse", "cholesky_solve", "inverse", "pinverse", "lu", "lu_solve",
        "qr", "svd", "triangular_solve", "lstsq", "cdist", "kron")
for n in BLAS: wrap(torch, n, f"torch.{n}")
for n in BLAS + ("__matmul__", "__rmatmul__", "addmm_", "addmv_", "addbmm_", "baddbmm_", "addr_"): wrap(torch.Tensor, n, f"Tensor.{n}")
for n in dir(torch.linalg):
    if not n.startswith("_"): wrap(torch.linalg, n, f"linalg.{n}")
wrap(torch.nn.functional, "linear", "F.linear")

# ---- the traced job, in-process; then the plain one in a fresh process
tr, pl = os.path.join(a.out, "traced"), os.path.join(a.out, "plain")
for d in (tr, pl): shutil.rmtree(d, ignore_errors=True)
spec = json.load(open(a.spec))
_cap(a.device); job.run_quant(spec, tr, a.device)
lines = [f"  {lb:22s} x{n:<5d} largest operand {big[(lb, s)]:>10d} elements | {s}" for (lb, s), n in sorted(seen.items(), key=lambda kv: -big[kv[0]])]
report("hostmath", not seen, f"{len(seen)} call sites (torch {torch.__version__})" + "".join("\n" + l for l in lines))

env = dict(os.environ, PYTHONPATH=PIPE + os.pathsep + os.environ.get("PYTHONPATH", ""))
r = subprocess.run([sys.executable, "-m", "rxq", "job", a.spec, "-o", pl, "--device", str(a.device)], cwd=PIPE, env=env, capture_output=True, text=True)
files = sorted(f for f in os.listdir(tr) if f.endswith(".safetensors") and f.startswith(("t-", "c9-")))
diff = []
for f in files:
    q = os.path.join(pl, f)
    if not os.path.exists(q): diff.append(f"{f} missing"); continue
    with safe_open(os.path.join(tr, f), "pt") as x, safe_open(q, "pt") as y:
        if x.metadata() != y.metadata() or sorted(x.keys()) != sorted(y.keys()) or not all(
                x.get_tensor(n).dtype == y.get_tensor(n).dtype and torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys()): diff.append(f)
extra = sorted(set(f for f in os.listdir(pl) if f.endswith(".safetensors") and f.startswith(("t-", "c9-"))) - set(files))
report("untouched", r.returncode == 0 and files and not diff and not extra,
       f"plain job exit {r.returncode}; {len(files) - len(diff)}/{len(files)} files equal (tensors + metadata)"
       + (f"; differ {diff[:5]}" if diff else "") + (f"; only in plain {extra[:5]}" if extra else "")
       + (f"\n{r.stderr[-2000:]}" if r.returncode else ""))
print(f"HOSTMATH CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if res and all(res) else 1)
