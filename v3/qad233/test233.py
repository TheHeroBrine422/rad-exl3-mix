#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test233.py [DEVICE] -- w233's host-side changes vs the code they replace, bit for bit, on DEVICE (default cuda:0 when a card is visible, else cpu):
  use_ram_embed (token table in host RAM, pinned async rows) = qmodel.Student.embed (memmap rows, pageable copy)
  gold_targets (pinned async targets) = data181.Gold.targets
Run on a GPU box (dad1 / box2) and on the CPU. Exit 0 = all pass."""
import json, os, sys, tempfile, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../qad220", "../qad", "../qad181"): sys.path.insert(0, os.path.join(HERE, p))
import qad220 as Q  # noqa: E402
import qmodel  # noqa: E402
import data181 as D  # noqa: E402

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
    if not ok: fails.append(name)

tmp = tempfile.mkdtemp(prefix="test233-")
rg = np.random.default_rng(0)
# token table as a memmap of bf16 bits (int16), like Student.emb
Vt, H = 5000, 256
tab = rg.standard_normal((Vt, H)).astype(np.float32)
bits = torch.from_numpy(tab).to(torch.bfloat16).view(torch.int16).numpy()
bits.tofile(f"{tmp}/emb.i16")
stu = types.SimpleNamespace(emb=np.memmap(f"{tmp}/emb.i16", np.int16, "r", shape=(Vt, H)), devs=[dev])
ref_embed = qmodel.Student.embed
Q.use_ram_embed(stu)
ok = True
for n in (1, 7, 1024, 3000):
    ids = rg.integers(0, Vt, n).astype(np.int64)
    a = stu.embed(ids); b = ref_embed(stu, ids)
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    ok &= a.dtype == b.dtype and a.shape == b.shape and a.device == b.device and torch.equal(a.view(torch.int16), b.view(torch.int16))
ids32 = rg.integers(0, Vt, 100).astype(np.int32)                    # Gold.ids gives int64; int32 ids too
ok &= torch.equal(stu.embed(ids32).view(torch.int16), ref_embed(stu, ids32).view(torch.int16))
check(f"use_ram_embed = Student.embed bit for bit on {dev} (1 / 7 / 1024 / 3000 / int32 ids)", ok)

# async reuse: many back-to-back embeds before one sync (pinned buffers must not be reused while their copy is in flight)
outs, refs = [], []
for k in range(50):
    ids = rg.integers(0, Vt, 1024); outs.append(stu.embed(ids)); refs.append(ref_embed(stu, ids))
if dev.type == "cuda": torch.cuda.synchronize(dev)
check("50 embeds in flight, then one sync: every one bit for bit", all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(outs, refs)))

# a synthetic golden (test220's recipe)
Vg, K = 64, 8
gdir = os.path.join(tmp, "gold"); os.makedirs(gdir)
nt = [3000, 900]; sf = [0, 5]; R = sum(n - 1 - s for n, s in zip(nt, sf))
rec = D.rec(K); top = np.empty((R, K), rec); rows = np.zeros((R, 3), np.float32)
for r in range(R):
    pr = rg.dirichlet(np.ones(Vg) * 0.3); ix = np.argsort(-pr)[:K]; top["id"][r] = ix; top["lp"][r] = np.log(pr[ix]); rows[r, 2] = 1 - pr[ix].sum()
top.tofile(f"{gdir}/top.bin"); rows.tofile(f"{gdir}/rows.f32"); rg.integers(0, Vg, sum(nt)).astype(np.int32).tofile(f"{gdir}/tokens.i32")
r0, docs = 0, []
for n, s in zip(nt, sf): docs.append({"tokens": n, "score_from": s, "row0": r0}); r0 += n - 1 - s
json.dump({"docs": docs, "top": K, "rows": R}, open(f"{gdir}/kld.json", "w"))
(np.arange(R) % 3 != 0).astype(np.uint8).tofile(f"{gdir}/gen.u8")
gold = D.Gold(gdir, "toy")
ok = True
for i, (a_, b_) in ((0, (0, 1024)), (0, (1024, 2999)), (1, (0, 900)), (1, (3, 40))):
    _, r = gold.scored(i, a_, b_)
    x = Q.gold_targets(gold, r, dev); y = gold.targets(r, dev)
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    ok &= all(u.dtype == v.dtype and u.shape == v.shape and torch.equal(u, v) for u, v in zip(x, y))
check(f"gold_targets = Gold.targets bit for bit on {dev} (dtype, shape, values; 4 windows)", ok)
print("\nALL PASS" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
