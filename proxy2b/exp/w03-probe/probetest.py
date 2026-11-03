#!/usr/bin/env python3
"""probetest.py PATCHED_COLLECT.py ORIG_COLLECT.py -- CPU self-test of the A10 patch (collect-probe.patch):
 1. the probe's covariance: Monte Carlo E[d dᵀ] = diag(p) - p pᵀ (= the sampled label's) for gauss and rad, on a peaky
    and a flat distribution; and the per-draw spread of a quadratic form vᵀ d dᵀ v vs the one-hot label's;
 2. the patched chunk_probe itself (taken from the patched file with ast, run with a small linear "head"): its
    gradient w.r.t. the logits equals sqrt(p) xi - p (sqrt(p)·xi) with that seed's xi; the same seed gives the same
    draw; torch.utils.checkpoint (as collect.py calls it) gives the gradient w.r.t. h of the plain call, bit for bit;
 3. no-op when YAQA_PROBE is unset: the patch removes exactly one line (the hdiag metadata dict), whose replacement
    evaluates to the same dict with PROBE = None; every other hunk only adds lines, all behind PROBE / an assert that
    holds with PROBE = None. (Bit-identical GPU output is checked by run.sh step 0.)"""
import ast, re, subprocess, sys, textwrap
import torch
from torch.utils.checkpoint import checkpoint
patched, orig = sys.argv[1:3]
ok = True
torch.manual_seed(0)

print("1. covariance and quadratic-form spread (R = 200000 draws, V = 64)")
V, R = 64, 200000
for name, logit in (("peaky", torch.randn(V, dtype=torch.float64) * 4), ("flat", torch.randn(V, dtype=torch.float64) * 0.5)):
    p = torch.softmax(logit, 0); sp = p.sqrt(); F = torch.diag(p) - torch.outer(p, p)
    v = torch.randn(V, dtype=torch.float64)
    ds = {"sample": p[None] - torch.nn.functional.one_hot(torch.multinomial(p, R, replacement=True), V).double()}
    for kind in ("gauss", "rad"):
        xi = torch.randn(R, V, dtype=torch.float64) if kind == "gauss" else torch.randint(0, 2, (R, V)).double() * 2 - 1
        ds[kind] = sp * xi - p * (xi @ sp)[:, None]
    for k, d in ds.items():
        C = d.T @ d / R; err = float((C - F).norm() / F.norm())
        qf = (d @ v) ** 2; cv = float(qf.std() / qf.mean())
        print(f"   {name:5s} {k:6s}: |E[dd^T] - F| / |F| {err:.4f}   v^T d d^T v: mean/v^T F v {float(qf.mean() / (v @ F @ v)):.4f}, CV {cv:.2f}")
        ok &= err < 0.02

print("2. the patched chunk_probe")
src = open(patched).read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "chunk_probe")
for kind in ("gauss", "rad"):
    head = torch.nn.Linear(16, 300, bias=False)
    model = type("M", (), {"lm_head": head})()
    ns = {"torch": torch, "model": model, "PROBE": kind}
    exec(compile(ast.Module([fn], []), patched, "exec"), ns)
    chunk_probe = ns["chunk_probe"]
    h = torch.randn(8, 16, requires_grad=True)
    z = head(h).float(); z.retain_grad(); seed = 12345
    zz = model.lm_head(h).float(); zz.retain_grad()
    ns["model"] = type("M2", (), {"lm_head": lambda self_, x, zz=zz: zz})()
    exec(compile(ast.Module([fn], []), patched, "exec"), ns)       # same function, a head that returns the tracked z
    ns["chunk_probe"](h, seed).backward()
    g = torch.Generator(); g.manual_seed(seed)
    xi = torch.randn(zz.shape, generator=g) if kind == "gauss" else torch.randint(0, 2, zz.shape, generator=g, dtype=torch.float32) * 2 - 1
    sp = torch.softmax(zz.detach(), -1).sqrt(); d = sp * xi - sp ** 2 * (sp * xi).sum(-1, keepdim=True)
    e_d = float((zz.grad - d).abs().max())
    ns["model"] = model; exec(compile(ast.Module([fn], []), patched, "exec"), ns); chunk_probe = ns["chunk_probe"]
    h1 = h.detach().clone().requires_grad_(True); chunk_probe(h1, seed).backward()
    h2 = h.detach().clone().requires_grad_(True); checkpoint(chunk_probe, h2, seed, use_reentrant=False).backward()
    h3 = h.detach().clone().requires_grad_(True); chunk_probe(h3, seed + 1).backward()
    same = torch.equal(h1.grad, h2.grad); diff_seed = not torch.equal(h1.grad, h3.grad)
    print(f"   {kind}: grad wrt logits vs closed form max abs {e_d:.2e}; checkpoint == plain bit for bit: {same}; "
          f"another seed differs: {diff_seed}")
    ok &= e_d < 1e-6 and same and diff_seed

print("3. no-op with YAQA_PROBE unset")
dl = subprocess.run(["diff", orig, patched], capture_output=True, text=True).stdout.splitlines()
removed = [l[2:] for l in dl if l.startswith("< ")]; added = [l[2:] for l in dl if l.startswith("> ")]
print(f"   diff: {len(removed)} line(s) removed, {len(added)} added")
ok &= len(removed) == 1 and "metadata=" in removed[0]
L, FP32, PREC, SEED, PROBE, DSTATE = 2048, True, "fp32", 0, None, {"n": 7}
old = eval(re.search(r"metadata=(\{.*\})\)", removed[0]).group(1))
i = next(j for j, l in enumerate(added) if 'metadata={"seqs"' in l)
new = eval((added[i] + added[i + 1]).split("metadata=", 1)[1].rsplit(")", 1)[0])
print(f"   metadata with PROBE=None identical: {old == new}")
ok &= old == new
FULL, DIAG, LABELS = set(), True, None
exec("assert not PROBE or ((DIAG or FULL) and not LABELS)")         # the patch's guard holds when PROBE is unset
print("   guard assert holds with PROBE=None: True")
print("PROBETEST", "PASS" if ok else "FAIL")
