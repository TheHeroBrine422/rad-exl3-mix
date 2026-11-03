#!/usr/bin/env python3
"""CPU self-test of w10-hout-attrib: mksub.py on the real c18 file list (empty stand-in files), idcheck.py on synthetic
wq dirs, summ.py on synthetic score files with planted per-subset gains (incl. the adaptive-seed listing)."""
import json, math, os, re, subprocess, sys, tempfile
import numpy as np, torch
from safetensors.torch import save_file
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = tempfile.mkdtemp(); ok = True
def check(c, msg):
    global ok; ok &= bool(c); print(("PASS " if c else "FAIL ") + msg)
run = lambda *a, **k: subprocess.run([sys.executable, *a], capture_output=True, text=True, **k)

# 1. mksub on the real 150 names
names = [l.strip() for l in open(f"{X}/selftest/c18-dy-files.txt") if l.strip()]
src = f"{T}/src"; os.makedirs(src)
for n in names: open(f"{src}/{n}", "w").close()
cnt = {}
for nm, sub in (("gdn_in", "fam:gdn_in"), ("attn_in", "fam:attn_in"), ("mlp_in", "fam:mlp_in"), ("writers", "fam:writers"),
                ("d0", "depth:0-7"), ("d1", "depth:8-15"), ("d2", "depth:16-23")):
    r = run(f"{X}/mksub.py", src, f"{T}/sub/{nm}", sub); assert r.returncode == 0, r.stderr
    cnt[nm] = set(l.strip() for l in open(f"{T}/sub/{nm}/subset.txt"))
    check(all(os.path.islink(f"{T}/sub/{nm}/{k}.safetensors") for k in cnt[nm]), f"{nm}: {len(cnt[nm])} symlinks")
fam = [cnt[k] for k in ("gdn_in", "attn_in", "mlp_in", "writers")]; dep = [cnt[k] for k in ("d0", "d1", "d2")]
check([len(s) for s in fam] == [36, 18, 48, 48], f"family sizes {[len(s) for s in fam]} == [36, 18, 48, 48]")
check(set().union(*fam) == set(n[:-12] for n in names) and sum(map(len, fam)) == 150, "families partition the 150 linears")
check(set().union(*dep) == set(n[:-12] for n in names) and [len(s) for s in dep] == [50, 50, 50], f"depth thirds partition: {[len(s) for s in dep]}")
# no Hessian group split: q/k/v, qkv/z, gate/up land in the same subset
grp = lambda k: re.sub(r"(q_proj|k_proj|v_proj)$", "qkv", re.sub(r"(in_proj_qkv|in_proj_z)$", "gin", re.sub(r"(gate_proj|up_proj)$", "gu", k)))
for nm, s in cnt.items():
    gs = {grp(k) for k in s}; full = {k for k in (n[:-12] for n in names) if grp(k) in gs}
    check(full == s, f"{nm}: no Hessian group split")
r = run(f"{X}/mksub.py", src, f"{T}/sub/x", "depth:30-31"); check(r.returncode != 0, "empty subset refused")

# 2. idcheck
keys = sorted(n[:-12] for n in names); sub = cnt["mlp_in"]
for d in ("b", "a", "c"): os.makedirs(f"{T}/wq{d}")
g = torch.Generator().manual_seed(0)
for k in keys:
    w = torch.randn(8, 16, generator=g).half(); save_file({"weight": w}, f"{T}/wqb/{k}.safetensors")
    wa = w + 0.01 if k in sub else w.clone(); save_file({"weight": wa}, f"{T}/wqa/{k}.safetensors")
    wc = wa.clone()
    if k == "model.language_model.layers.5.mlp.down_proj": wc[0, 0] += 0.001
    save_file({"weight": wc}, f"{T}/wqc/{k}.safetensors")
r = run(f"{X}/idcheck.py", f"{T}/wqb", f"{T}/wqa", f"{T}/sub/mlp_in/subset.txt"); j = json.loads(r.stdout)
check(r.returncode == 0 and j["outside_identical"] == 102 and j["subset_changed"] == 48, f"idcheck pass case: {j}")
r = run(f"{X}/idcheck.py", f"{T}/wqb", f"{T}/wqc", f"{T}/sub/mlp_in/subset.txt"); j = json.loads(r.stdout)
check(r.returncode == 1 and j["first_changed"] == ["model.language_model.layers.5.mlp.down_proj"], "idcheck catches one changed outside weight")

# 3. summ on synthetic score files: per-row KL = base_s * exp(-gain) ; shares planted
D = f"{T}/res"; os.makedirs(D); rng = np.random.default_rng(1)
N = 20000; gen = (rng.random(N) < 0.8).astype(np.float32)
def write(tag, scale):
    kl = (rng.gamma(2.0, 0.05, N) * 0 + 0.1) * scale     # constant rows: the mean is exactly 0.1*scale
    np.stack([kl, np.ones(N), gen, np.zeros(N)], 1).astype(np.float32).tofile(f"{D}/{tag}.f32")
G = {3: 0.09, 2: 0.065}; SH = {"gdn_in": 0.30, "attn_in": 0.05, "mlp_in": 0.45, "writers": 0.155, "d0": 0.10, "d1": 0.30, "d2": 0.60}
sx = lambda s: "" if s == 0 else f"_s{s}"
noise = {0: 0.004, 1: -0.004, 2: 0.0}
for K in (3, 2):
    for s in (0, 1, 2):
        b = 1 + 0.01 * s; write(f"k{K}base{sx(s)}", b); write(f"k{K}edy{sx(s)}", b * math.exp(-G[K] + noise[s]))
        for n, sh in SH.items():
            if s < 2: write(f"k{K}h-{n}{sx(s)}", b * math.exp(-sh * G[K] + noise[s] / 2))
for K, gk in ((4, 0.05), (5, 0.02)):
    for s in (0, 1): write(f"k{K}base{sx(s)}", 1.0); write(f"k{K}edy{sx(s)}", math.exp(-gk + noise[s]))
r = run(f"{X}/summ.py", D); print(r.stdout[:2500]); out = json.loads(r.stdout.strip().splitlines()[-1])
check(abs(out["full"]["3"]["g"] - 0.09) < 1e-6 and abs(out["full"]["2"]["g"] - 0.065) < 1e-6, "full gains recovered")
check(all(abs(out["subsets"][n]["3"]["share"] - SH[n]) < 1e-4 for n in SH), "planted K3 shares recovered")
V = out["verdicts"]
check(V["attn_in"].startswith("one-sided (minor") and V["mlp_in"] == "two-sided" and V["K5"].startswith("skip") and V["K4"] == "keep two-sided",
      f"verdicts {V}")
check(abs(out["additivity"]["K3 families"] - (0.30 + 0.05 + 0.45 + 0.155)) < 1e-3, "additivity = sum of shares")
need = run(f"{X}/summ.py", D, "--need-seeds").stdout.split("\n")
# writers sits at 0.155, |0.155 - 0.15| < SE -> one more seed (seed 2); attn_in at 0.05 is clear; K5 0.02 vs 0.03 is 2.5 SE away: no seed
check("3 h-writers 2" in need and "3 h-attn_in 2" not in need, f"adaptive seeds listed: {[l for l in need if l]}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
