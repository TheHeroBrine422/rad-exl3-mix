#!/usr/bin/env python3
"""test_w78.py -- regression checks for w78's sibmom27 / asym27.sh fix (CPU, seconds; run on any box with torch):
  F  save_lean in row blocks = safetensors' save_file (tensors, dtypes, metadata) and byte for byte the pre-w78
     save_lean's file, for block sizes that split the rows unevenly, one row per block, and the default (one block),
     from f64 sums as sibmom27 holds them
  G  sibmom27 refuses the expandable_segments allocator on ROCm (the guard is in place and comes before any GPU work),
     and asym27.sh neither sets it nor inherits it (unsets the three allocator-config variables)
  H  sibmom27's load() drops its open checkpoint shards (the embedding's 2.5 GB host copy no longer lives through layer 0)
  I  mkdenoise's lean-path solve (Cholesky + blocked triangular solves, replacing torch.linalg.solve, which takes > 20 min
     at 17408 f64 on ROCm) = torch.linalg.solve within 1e-11 relative, f64, for uneven / single / many blocks, and the
     lean path calls it (with the LU fallback)
The GPU side (layers 0-1 at 400k tokens with and without expandable segments) is dbg/dbgmom.sh; results in dbg/README.md."""
import json, os, re, sys, tempfile
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{HERE}/sibmom27.py").read()
res = {}

# F
fn = src[src.index("def save_lean("):src.index("\n\n\ndef save(")]
ns = {"json": json, "os": os, "torch": torch}; exec(fn, ns)
OLD = open(f"{HERE}/dbg/save_lean.pre-w78.py").read()      # the unchunked save_lean (git 10a26c4), for a byte compare
g = torch.Generator().manual_seed(78)
tot = {s: torch.randn(301 + i, 301 + i, dtype=torch.float64, generator=g)[:301, :301].contiguous() for i, s in enumerate("HCN")}
md = {"n": "1234", "mode": "sib", "wq": "x", "kernels": "{\"a\": 1}", "level": "None"}
ok = True
with tempfile.TemporaryDirectory() as d:
    save_file({s: v.float().contiguous() for s, v in tot.items()}, f"{d}/ref.safetensors", metadata=md)
    ons = {"json": json, "os": os, "torch": torch}; exec(OLD, ons); ons["save_lean"](f"{d}/old.safetensors", tot, md)
    old_bytes = open(f"{d}/old.safetensors", "rb").read()
    for cb in (4 * 301 * 7, 4 * 301, 1, 64 << 20):          # 7 rows a block (301 = 43 x 7), 1 row, below 1 row, default
        ns["save_lean"](f"{d}/a.safetensors", tot, md, chunk_bytes=cb)
        with safe_open(f"{d}/a.safetensors", "pt") as A, safe_open(f"{d}/ref.safetensors", "pt") as B:
            same = A.metadata() == B.metadata() and sorted(A.keys()) == sorted(B.keys()) and \
                all(torch.equal(A.get_tensor(s), B.get_tensor(s)) and A.get_tensor(s).dtype == torch.float32 for s in "HCN")
        same = same and open(f"{d}/a.safetensors", "rb").read() == old_bytes     # byte for byte the pre-w78 save_lean's file
        print(f"F chunk_bytes {cb}: {'identical' if same else 'DIFFER'}", flush=True); ok &= same
res["F chunked save_lean = save_file"] = ok

# G
m = re.search(r"\nif DEV\.type == \"cuda\" and torch\.version\.hip and any\(\"expandable_segments:true\".*?sys\.exit\(", src, re.S)
first_gpu = min(i for i in (src.find("torch.cuda.set_per_process_memory_fraction"), src.find(".to(DEV)")) if i > 0)
sh = open(f"{HERE}/asym27.sh").read()
code_lines = [l for l in sh.splitlines() if not l.lstrip().startswith("#")]
g_ok = bool(m) and m.start() < first_gpu and not any("expandable_segments" in l for l in code_lines) and \
    any(re.match(r"\s*unset PYTORCH_HIP_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF\b", l) for l in code_lines)
print(f"G expandable_segments guard in sibmom27 before GPU work: {bool(m) and m.start() < first_gpu}; "
      f"asym27.sh sets none and unsets the allocator config: {g_ok}", flush=True)
res["G no expandable_segments"] = g_ok

# H
ld = src[src.index("def load(prefix, mod):"):src.index("def unload(")]
res["H load() drops open shards"] = "_open.clear()" in ld
print(f"H load() clears _open: {res['H load() drops open shards']}", flush=True)

# I
ms = open(f"{HERE}/mkdenoise.py").read()
fn = ms[ms.index("def _chol_solve_blocked("):ms.index("def lean_group(")]
ns = {"torch": torch}; exec(fn, ns)
worst = 0.0
for n, bs in ((1500, 1024), (1500, 128), (700, 1024), (513, 64)):
    X = torch.randn(n, n // 3, dtype=torch.float64, generator=g); G = X @ X.T / n; G.diagonal().add_(0.025 * float(G.diagonal().mean()))
    B = torch.randn(n, n, dtype=torch.float64, generator=g)
    ref = torch.linalg.solve(G, B); got = ns["_chol_solve_blocked"](torch.linalg.cholesky(G), B.clone(), bs=bs)
    worst = max(worst, float((got - ref).abs().max() / ref.abs().max()))
lg = ms[ms.index("def lean_group("):ms.index("for mf in files:")]
calls = "_chol_solve_blocked(L, A.T.contiguous())" in lg and "torch.linalg.solve(Gs, A.T)" in lg and "cholesky_ex(Gs)" in lg
res["I lean solve = linalg.solve"] = worst < 1e-11 and calls
print(f"I blocked Cholesky solve vs torch.linalg.solve: max rel {worst:.1e}; lean path uses it with the LU fallback: {calls}", flush=True)

for k, v in res.items(): print(f"{'PASS' if v else 'FAIL'}  {k}")
print("ALL PASS" if all(res.values()) else "FAILED")
sys.exit(0 if all(res.values()) else 1)
