#!/usr/bin/env python3
"""check_cpu.py OUTDIR -- bit-identity of the CPU-side ports against the rad-exl3-mix originals, on data on the 4070
box (0.8B statistics, the 27B hconv H_out snapshots). CPU only. Each check prints PASS / FAIL and a reason; the last
line is "CPU CHECKS n/m PASS".
  (files compared by content: tensors torch.equal + metadata dict; header bytes vary by process)
  blend   rxq.stats.blend_hin(h08 calib, d1111 hinX, 0.5)      vs d1111/hi-X0.5 (made by d1108/blend.py): file bytes
  hout    rxq.stats.hout_blocks(hconv snaps, dy, norm)          vs mkhalf.py (MKHALF_NORM=1) run now on the same snapshots
  swjmath rxq.swj.acc / fit / resid                             vs fsutil.swiglu_acc / _fit / _resid on random blocks
  swjapp  rxq.swj.apply_level(q-k3base, w36 k3lv scales)        vs swjapply.py on the same inputs: file bytes
  denoise rxq.asym.Denoiser on 4 groups of sib27test/ref2 (default and lean path) vs mkdenoise.py (DEV=cpu): bytes"""
import filecmp, glob, json, os, shutil, subprocess, sys

import torch

R = "/workspace/rad-exl3-mix"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rxq import stats, swj, asym  # noqa: E402

O = sys.argv[1]; os.makedirs(O, exist_ok=True)
only = sys.argv[2:]
res = []


def report(name, ok, why):
    res.append(ok); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


def same_st(pa, pb):
    """two safetensors files hold the same thing: tensor names, dtypes, values (torch.equal) and the metadata dict.
    Not file bytes: safetensors writes __metadata__ keys in hash order, which differs from process to process"""
    from safetensors import safe_open
    with safe_open(pa, "pt") as x, safe_open(pb, "pt") as y:
        if x.metadata() != y.metadata() or sorted(x.keys()) != sorted(y.keys()): return False
        return all(x.get_tensor(n).dtype == y.get_tensor(n).dtype and torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys())


def same_tree(a, b, names=None):
    """every name in a equal in b: safetensors by content (same_st), other files by bytes, symlinks by target"""
    names = names or sorted(os.listdir(a))
    bad = []
    for n in names:
        pa, pb = os.path.join(a, n), os.path.join(b, n)
        if os.path.isdir(pa): continue
        if os.path.islink(pa) or os.path.islink(pb):
            if os.path.realpath(pa) != os.path.realpath(pb): bad.append(n)
        elif not os.path.exists(pb): bad.append(n)
        elif n.endswith(".safetensors"):
            if not same_st(pa, pb): bad.append(n)
        elif not filecmp.cmp(pa, pb, shallow=False): bad.append(n)
    return len(names), bad


def run(name):
    return not only or name in only


if run("blend"):
    d = f"{O}/blend"; shutil.rmtree(d, ignore_errors=True)
    n = stats.blend_hin("/data/h08/calib", "/data/yb/d1111/hinX", 0.5, d)
    tot, bad = same_tree("/data/yb/d1111/hi-X0.5", d, [f for f in os.listdir("/data/yb/d1111/hi-X0.5") if f.endswith(".safetensors")])
    report("blend", not bad, f"{n} blended, {tot} files compared, {len(bad)} differ {bad[:2]}")

if run("hout"):
    d = f"{O}/hout"; shutil.rmtree(d, ignore_errors=True)
    stats.hout_blocks("/data/yb/hconv/snaps", d, "dy", norm=True, split=True)
    ref = f"{O}/hout-orig"; shutil.rmtree(ref, ignore_errors=True)     # hconv/half-bn predates the last snapshots: rerun mkhalf
    subprocess.run([sys.executable, "-I", f"{R}/v3/yaqa/hconv/mkhalf.py", "/data/yb/hconv/snaps", ref, "dy"], check=True,
                   capture_output=True, env=dict(os.environ, MKHALF_NORM="1"))
    tot = 0; bad = []
    for sub in sorted(os.listdir(ref)):
        if os.path.isdir(f"{ref}/{sub}"):
            if not os.path.isdir(f"{d}/{sub}"): bad.append(sub + "/ (missing)"); continue
            t, b = same_tree(f"{ref}/{sub}", f"{d}/{sub}", [f for f in os.listdir(f"{ref}/{sub}") if f.endswith(".safetensors")])
            tot += t; bad += [f"{sub}/{x}" for x in b]
    mj = json.load(open(f"{ref}/mkhalf.json")) == json.load(open(f"{d}/mkhalf.json"))
    report("hout", not bad and mj, f"{tot} files compared, {len(bad)} differ {bad[:2]}; mkhalf.json equal {mj}")

if run("swjmath"):
    sys.path.insert(0, f"{R}/proxy2b/exp/w02-fixedscale"); import fsutil as U
    g = torch.Generator().manual_seed(52); grid = [0.6 + 0.01 * i for i in range(81)]; J = 96
    sa = [torch.zeros(81, J, dtype=torch.float64), torch.zeros(81, J, dtype=torch.float64), torch.zeros(J, dtype=torch.float64)]
    sb = [x.clone() for x in sa]
    for _ in range(4):
        aa, gq, uq = (torch.randn(257, J, generator=g) for _ in range(3))
        U.swiglu_acc(sa, aa, gq, uq, grid); swj.acc(sb, aa, gq, uq, grid)
    eq = all(torch.equal(x, y) for x, y in zip(sa, sb))
    fa, fb = U.swiglu_fit(sa, 40), swj.fit(sb, 40)
    eq2 = all(torch.equal(x, y) for x, y in zip(fa, fb))
    ra, rb = U.swiglu_resid(sa, fa[0], fa[1]), swj.resid(sb, fb[0], fb[1])
    report("swjmath", eq and eq2 and ra == rb, f"sums equal {eq}, fit equal {eq2}, resid equal {ra == rb}")

if run("swjapp"):
    lv, sc = "/data/yb/q-k3base", "/data/yb/w36/fit/k3lv"
    a, b = f"{O}/swjapp-orig", f"{O}/swjapp-rxq"
    for d in (a, b): shutil.rmtree(d, ignore_errors=True)
    subprocess.run([sys.executable, "-I", f"{R}/v3/yaqa/swj27/swjapply.py", lv, sc, a], check=True, capture_output=True)
    swj.apply_level(lv, sc, b)
    tot, bad = same_tree(a, b)
    report("swjapp", not bad, f"{tot} files compared, {len(bad)} differ {bad[:2]}")

if run("denoise"):
    src = "/data/yb/sib27test/ref2"; sub = f"{O}/dn-mom"; shutil.rmtree(sub, ignore_errors=True); os.makedirs(sub)
    for f in sorted(glob.glob(f"{src}/model.language_model.layers.[03].*.input.safetensors")): os.symlink(f, f"{sub}/{os.path.basename(f)}")
    env = dict(os.environ, DEV="cpu")
    for lean in ("0", "1"):
        a, b = f"{O}/dn-orig-{lean}", f"{O}/dn-rxq-{lean}"
        for d in (a, b): shutil.rmtree(d, ignore_errors=True)
        e = dict(env, MKD_LEAN=lean) if lean == "1" else env
        subprocess.run([sys.executable, "-I", f"{R}/proxy2b/exp/w07-denoise/mkdenoise.py", "/data/h08/calib", sub, a,
                        "--lam", "1", "--rho", "0", "--ckpt", "/data/models/qwen35-08b"], check=True, capture_output=True, env=e)
        asym.Denoiser("/data/h08/calib", sub, b, 1.0, 0.0, ckpt="/data/models/qwen35-08b", dev=torch.device("cpu"),
                      lean=lean == "1").run()
        t1, b1 = same_tree(f"{a}/hin", f"{b}/hin"); t2, b2 = same_tree(f"{a}/wpre", f"{b}/wpre")
        sa = [json.loads(l) for l in open(f"{a}/stats.jsonl")]; sb = [json.loads(l) for l in open(f"{b}/stats.jsonl")]
        report(f"denoise{'-lean' if lean == '1' else ''}", not b1 and not b2 and sa == sb,
               f"{t1} hin + {t2} wpre files, {len(b1) + len(b2)} differ; stats equal {sa == sb}")

print(f"CPU CHECKS {sum(res)}/{len(res)} PASS")
