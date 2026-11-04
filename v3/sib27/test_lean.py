#!/usr/bin/env python3
"""test_lean.py CALIB MOM MODEL WORK -- no-op test of mkdenoise-lean.patch (v3/sib27/lean/mkdenoise.py = the patched
file) against the repo's proxy2b/exp/w07-denoise/mkdenoise.py, on real moments (the 0.8B's w07-seqsib mom-k3base, 1M
tokens: layers 0, 3, 10, 11, 14, which include groups whose G needs clipping), CPU fp64:
  1. patched, default path (every 0.8B group is below MKD_LEAN_K): hin / wpre files bit-identical to the repo's, stats equal;
  2. patched, MKD_LEAN=1 (the 27B path forced on every group): hin and wpre bit-identical (tensors and metadata), stats
     equal within 1e-9 (the lean path sums the norms of hsub / F_dev by blocks);
  3. lam = rho = 0 on the lean path: hin = calib's hessian_upper bit-exact (the no-op gate's premise);
  4. peak RSS of the lean run vs the default one (reported; the lean path's point at the 27B's 17408-wide groups).
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python3 test_lean.py /data/h08/calib /data/yb/w07-seqsib/mom-k3base \
      /data/models/qwen35-08b /tmp/w12/lean"""
import glob, json, os, resource, shutil, subprocess, sys
import torch
from safetensors import safe_open
CALIB, MOM, MODEL, WORK = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "..", "..", "proxy2b", "exp", "w07-denoise", "mkdenoise.py"); NEW = os.path.join(HERE, "lean", "mkdenoise.py")
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(f"{WORK}/mom")
for l in (0, 3, 10, 11, 14):
    for f in glob.glob(f"{MOM}/model.language_model.layers.{l}.*.input.safetensors"): os.symlink(f, f"{WORK}/mom/{os.path.basename(f)}")
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {info}", flush=True)
def mk(script, out, lam, rho, **env):
    r = subprocess.run([sys.executable, script, CALIB, f"{WORK}/mom", out, "--lam", str(lam), "--rho", str(rho), "--ckpt", MODEL],
                       env=dict(os.environ, DEV="cpu", **env), capture_output=True, text=True)
    if r.returncode: print(r.stderr[-1500:])
    return r.returncode == 0, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss >> 10
def files_equal(a, b):
    fa = sorted(os.path.relpath(f, a) for f in glob.glob(f"{a}/*/*.safetensors")); fb = sorted(os.path.relpath(f, b) for f in glob.glob(f"{b}/*/*.safetensors"))
    if fa != fb or not fa: return False, f"file lists differ ({len(fa)} / {len(fb)})"
    bad = []
    for f in fa:
        with safe_open(f"{a}/{f}", "pt") as x, safe_open(f"{b}/{f}", "pt") as y:
            same = x.metadata() == y.metadata() and sorted(x.keys()) == sorted(y.keys()) and all(torch.equal(x.get_tensor(k), y.get_tensor(k)) for k in x.keys())
        if not same: bad.append(f)
    return not bad, f"{len(fa)} files, {len(bad)} differ {bad[:2]}"
def stats(d): return {json.loads(l)["group"]: json.loads(l) for l in open(f"{d}/stats.jsonl")}
def stats_close(a, b, tol):
    A, B = stats(a), stats(b); worst = 0.0
    if A.keys() != B.keys(): return False, "groups differ"
    for g in A:
        if A[g].keys() != B[g].keys(): return False, f"{g}: keys differ"
        for k, v in A[g].items():
            if isinstance(v, float): worst = max(worst, abs(v - B[g][k]) / max(abs(v), 1e-30))
            elif v != B[g][k]: return False, f"{g} {k}: {v} != {B[g][k]}"
    return worst <= tol, f"max rel {worst:.1e}"

o1, rss_lean = mk(NEW, f"{WORK}/lean", 1, 0.5, MKD_LEAN="1")
o0, rss_both = mk(ORIG, f"{WORK}/orig", 1, 0.5)
o2, _ = mk(NEW, f"{WORK}/default", 1, 0.5)
check("runs", o0 and o1 and o2)
check("1. patched default path = repo (files)", *files_equal(f"{WORK}/orig", f"{WORK}/default"))
check("1. patched default path = repo (stats)", *stats_close(f"{WORK}/orig", f"{WORK}/default", 0.0))
check("2. lean path = repo (hin, wpre: tensors and metadata)", *files_equal(f"{WORK}/orig", f"{WORK}/lean"))
check("2. lean path = repo (stats within 1e-9)", *stats_close(f"{WORK}/orig", f"{WORK}/lean", 1e-9))
print("   clip shares (lean):", {g.split("layers.")[1]: r["clip"] for g, r in stats(f"{WORK}/lean").items() if r["clip"]})
o3, _ = mk(NEW, f"{WORK}/noop", 0, 0, MKD_LEAN="1")
bad = []
for f in glob.glob(f"{WORK}/noop/hin/*.safetensors"):
    with safe_open(f, "pt") as x, safe_open(f"{CALIB}/{os.path.basename(f)}", "pt") as y:
        if not torch.equal(x.get_tensor("hessian_upper"), y.get_tensor("hessian_upper")): bad.append(os.path.basename(f))
check("3. lam = rho = 0 (lean): hin = calib H bit-exact", o3 and not bad and not os.path.exists(f"{WORK}/noop/wpre"), f"{len(bad)} differ")
print(f"   4. peak RSS: lean run {rss_lean} MiB; default run <= {rss_both} MiB (max over both; > lean means the default's)")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
