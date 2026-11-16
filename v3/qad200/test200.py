#!/usr/bin/env python3
"""test200.py -- CPU unit checks of #200's switches (no GPU, no student). The functions are taken from qad200.py itself
(AST extraction, so the tests run the shipped code): lr_main / group_lr (schedule in plan steps and in optimizer steps)
and the parser (defaults, new flags). Plus torch checks of the optimizer and accumulation math that the trainer relies on:
AdamW's decoupled decay (closed form), W = 0 (AdamW with no decay = Adam), gradient accumulation over micro-batches.
Run: python3 test200.py   (exit 0 = all pass)."""
import ast, argparse, json, math, os, sys
from fractions import Fraction
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = ast.parse(open(os.path.join(HERE, "qad200.py")).read())
ns = {"os": os, "argparse": argparse, "json": json, "math": math, "Fraction": Fraction, "np": np, "torch": torch, "HERE": HERE}
for node in TREE.body:
    if isinstance(node, ast.FunctionDef) and node.name in {"lr_main", "group_lr", "parser"}:
        exec(compile(ast.Module([node], []), "qad200.py", "exec"), ns)
lr_main, group_lr, parser = ns["lr_main"], ns["group_lr"], ns["parser"]
fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
    if not ok: fails.append(name)

# 1. N = 1: group_lr == lr_main(s - k0, last - k0, fork - k0) at every plan step, with and without a block offset
for last, fork, warm, k0 in [(49, 39, 8, 0), (194, 155, 32, 0), (49, 39, 8, 5), (30, 20, 3, 4)]:
    ok = all(group_lr(s, k0, last, fork, 1e-3, warm, 1) == lr_main(s - k0, last - k0, fork - k0, 1e-3, warm) for s in range(k0, last))
    check(f"N=1: group_lr == lr_main (last {last}, fork {fork}, warm {warm}, k0 {k0})", ok)

# 2. 4 x batch at 4 M rows: 194 plan steps -> 49 optimizer steps; warmup 8 of 49; decay from group 39 (= ceil(155 / 4))
last, fork, N = 194, 155, 4
U = -(-last // N); FU = -(-fork // N); W = max(1, round(float(Fraction("8/49")) * U))
lrs = [group_lr(g * N, 0, last, fork, 1e-3, W, N) for g in range(U)]
check("B4 shape: 49 optimizer steps, warmup 8, decay from group 39", U == 49 and W == 8 and FU == 39, f"U {U} W {W} FU {FU}")
check("B4 warmup ramps lr/8 .. 8lr/8, then constant", all(abs(lrs[i] - 1e-3 * (i + 1) / 8) < 1e-15 for i in range(8))
      and all(abs(lrs[i] - 1e-3) < 1e-15 for i in range(8, FU)))
check("B4 last group: lr * (U - (U-1)) / (U - FU) = lr / 10", abs(lrs[-1] - 1e-3 / 10) < 1e-15, f"{lrs[-1]:.3e}")
# the 4 M arm at normal batch: 194 plan steps, warmup round(8/49 * 194) = 32
check("A4M warmup = round(8/49 x 194) = 32", max(1, round(float(Fraction("8/49")) * 194)) == 32)

# 3. parser: defaults keep #196's behaviour; the new flags parse
p = parser()
d = p.parse_args(["OUT", "--model", "m", "--level", "l", "--chunks", "c"])
check("defaults: wd 0, accum 1, rows None, warmup_frac None, order_seed None, seed 0",
      d.wd == 0.0 and d.accum == 1 and d.rows is None and d.warmup_frac is None and d.order_seed is None and d.seed == 0)
d = p.parse_args(["OUT", "--model", "m", "--level", "l", "--chunks", "c", "--wd", "20", "--accum", "4", "--rows", "4e6",
                  "--order-seed", "3", "--warmup-frac", "8/49"])
check("flags parse (wd 20, accum 4, rows 4e6, order-seed 3, warmup-frac 8/49)",
      d.wd == 20.0 and d.accum == 4 and d.rows == 4e6 and d.order_seed == 3 and d.warmup_frac == "8/49")

# 4. AdamW decoupled decay: each step is "p <- p (1 - lr wd)" then the Adam update (closed form, float64)
torch.manual_seed(0)
lr, wd = 1e-3, 20.0
p0 = torch.randn(64, dtype=torch.float64) * 0.1
g = torch.randn(64, dtype=torch.float64)
pa = torch.nn.Parameter(p0.clone())
oa = torch.optim.AdamW([{"params": [pa], "weight_decay": wd}], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
pc = torch.nn.Parameter(p0.clone()); oc = torch.optim.Adam([pc], lr=lr, betas=(0.9, 0.99))
for _ in range(3):
    pa.grad = g.clone(); oa.step()
    with torch.no_grad(): pc.mul_(1 - lr * wd)
    pc.grad = g.clone(); oc.step()
check("AdamW(wd) == decay by (1 - lr wd) then the Adam step (3 steps, float64)", torch.allclose(pa.detach(), pc.detach(), rtol=0, atol=1e-14),
      f"max diff {float((pa - pc).abs().max()):.2e}")
pz = torch.nn.Parameter(p0.clone()); oz = torch.optim.AdamW([{"params": [pz], "weight_decay": wd}], lr=lr, weight_decay=0.0)
pz.grad = torch.zeros_like(pz); oz.step()
check("zero gradient: the step is the decay alone, p (1 - lr wd) = 2 % shrink at wd 20", torch.allclose(pz.detach(), p0 * (1 - lr * wd)),
      f"ratio {float((pz / p0).mean()):.6f}")

# 5. W = 0: AdamW with no decay is Adam (bitwise); the trainer keeps torch.optim.Adam for W = 0 (information)
pw = torch.nn.Parameter(p0.clone()); ow = torch.optim.Adam([{"params": [pw]}], lr=lr, betas=(0.9, 0.99))
pv = torch.nn.Parameter(p0.clone()); ov = torch.optim.AdamW([{"params": [pv], "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
for _ in range(3):
    pw.grad = g.clone(); ow.step(); pv.grad = g.clone(); ov.step()
check("AdamW(weight_decay=0) == Adam bitwise (3 steps)", torch.equal(pw.detach(), pv.detach()))

# 6. gradient accumulation: micro-batch backward passes with loss / (rows of the whole optimizer step) == one backward
def kl_sum(w, x, y):                                  # a smooth stand-in for the per-row KL sum of a linear model
    z = x @ w
    return (torch.nn.functional.softplus(z) - y * z).sum()
torch.manual_seed(1)
w0 = torch.randn(8, dtype=torch.float64)
micro = [(torch.randn(n, 8, dtype=torch.float64), torch.rand(n, dtype=torch.float64)) for n in (5, 7, 4, 6)]
rows = sum(x.shape[0] for x, _ in micro)
wa = torch.nn.Parameter(w0.clone())
for x, y in micro: (kl_sum(wa, x, y) / rows).backward()           # accumulate across micro-batches, one update after
wb = torch.nn.Parameter(w0.clone())
(kl_sum(wb, torch.cat([x for x, _ in micro]), torch.cat([y for _, y in micro])) / rows).backward()
check("accumulated gradient (unequal micro-batches) == one backward over the step's rows", torch.allclose(wa.grad, wb.grad, rtol=0, atol=1e-13),
      f"max diff {float((wa.grad - wb.grad).abs().max()):.2e}")
eq = [(torch.randn(6, 8, dtype=torch.float64), torch.rand(6, dtype=torch.float64)) for _ in range(4)]
w1 = torch.nn.Parameter(w0.clone())
for x, y in eq: (kl_sum(w1, x, y) / (6 * 4)).backward()          # N = 4 equal micro-batches: 1/N of each micro-batch mean
m1 = torch.nn.Parameter(w0.clone())
(kl_sum(m1, torch.cat([x for x, _ in eq]), torch.cat([y for _, y in eq])) / 24).backward()
check("N = 4 equal micro-batches: the accumulated grad == the grad of the mean over the 4 micro-batches",
      torch.allclose(w1.grad / 4, m1.grad / 4 * 1, rtol=0, atol=1e-13) and torch.allclose(w1.grad, m1.grad, rtol=0, atol=1e-13))
print("FAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
