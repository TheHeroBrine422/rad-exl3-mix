#!/usr/bin/env python3
"""test_summ.py WORK -- summ.py on synthetic score files (a 60-document golden, evalq.py's [kl, top1, gen, 0] rows):
pick returns the arm with the lowest fit KL; verdict gives PASS / FAIL / INCONCLUSIVE as its three rules say, and reads
a base from Y when DIR has none."""
import json, os, subprocess, sys
import numpy as np
W = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__)); S = f"{HERE}/../summ.py"
for d in ("g", "d", "y"): os.makedirs(f"{W}/{d}", exist_ok=True)
rng = np.random.default_rng(0); docs, r0 = [], 0
for i in range(60):
    t = int(rng.integers(100, 400)); docs.append({"tokens": t, "score_from": 0, "row0": r0}); r0 += t - 1
json.dump({"rows": r0, "docs": docs}, open(f"{W}/g/kld.json", "w"))
gen = (rng.random(r0) < 0.8).astype(np.float32); base = rng.gamma(2.0, 0.05, r0).astype(np.float32)
def put(path, scale, noise=0.01):
    kl = base * scale * (1 + noise * rng.standard_normal(r0)).astype(np.float32)
    np.stack([kl, np.zeros(r0, np.float32), gen, np.zeros(r0, np.float32)], 1).astype(np.float32).tofile(path)
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {info}", flush=True)
def run(*a): return subprocess.run([sys.executable, S, *a], capture_output=True, text=True)
put(f"{W}/d/k3base.fit.f32", 1.0); put(f"{W}/d/k3a.fit.f32", 0.99); put(f"{W}/d/k3b.fit.f32", 0.95)
r = run("pick", f"{W}/d", "3", "a,b")
check("pick: lowest fit KL", r.stdout.strip() == "b", r.stderr.strip().replace("\n", " | "))
sfx = lambda s: "" if s == 0 else f"_s{s}"
for s, bs in enumerate((1.0, 1.01, 0.99)): put(f"{W}/y/k3base{sfx(s)}.f32", bs)      # base read from Y
for arm, sc in (("good", 0.95), ("meh", 0.985), ("bad", 1.02)):
    for s in range(3): put(f"{W}/d/k3{arm}{sfx(s)}.f32", sc * (1.0, 1.01, 0.99)[s])
for c, sc in (("rs7", 0.999), ("rs8", 1.002), ("rs9", 1.0)): put(f"{W}/d/k3{c}.f32", sc)
def verdict(arm, rule, bar, *c):
    r = run("verdict", f"{W}/d", f"{W}/g", f"{W}/y", "3", arm, rule, str(bar), *c)
    return r.stdout.strip().splitlines()[-1].split("=>")[-1].strip().split()[0] if r.returncode == 0 else "ERR " + r.stderr[-300:]
check("seed rule: 5% gain at bar 0.97 -> PASS", verdict("good", "seed", 0.97) == "PASS")
check("seed rule: 1.5% gain at bar 0.97 -> INCONCLUSIVE", verdict("meh", "seed", 0.97) == "INCONCLUSIVE")
check("seed rule: 2% loss -> FAIL", verdict("bad", "seed", 0.97) == "FAIL")
check("mean rule: 5% gain at bar 0.97 -> PASS", verdict("good", "mean", 0.97) == "PASS")
check("mean rule: 1.5% gain at bar 0.97 -> INCONCLUSIVE", verdict("meh", "mean", 0.97) == "INCONCLUSIVE")
check("ctrl rule: 1.5% gain, control sd 0.15% at bar 0.98 -> INCONCLUSIVE (bar 0.98 not met)", verdict("meh", "ctrl", 0.98, "rs7,rs8,rs9") == "INCONCLUSIVE")
check("ctrl rule: 1.5% gain, control sd 0.15% at bar 0.99 -> PASS", verdict("meh", "ctrl", 0.99, "rs7,rs8,rs9") == "PASS")
check("summary.jsonl written", sum(1 for _ in open(f"{W}/d/summary.jsonl")) == 7)
print("ALL PASS" if ok else "SOME FAIL")
