#!/usr/bin/env python3
"""test_cpu.py MODEL META WQDIR LEVEL WORK -- CPU check of sibmom27.py against moments.py --mode sib on the real 0.8B
(a few short qad windows; DEV=cpu, P2B_BLOCK=causal_conv1d,fla in both, so both take transformers' torch paths):
  A. sibmom27 --wq = moments.py --mode sib (same windows, same siblings, layers 0-3): H, C, N and n bit-exact
     (tolerance stated: max |diff| / max |ref| <= 1e-6; expected 0, the same ops in the same order);
  B. siblings decoded from the level's shards (--level, deq.py) instead of --save-wq files: (1) = --wq on the files
     `deq.py wq` writes, bit-exact (one pipeline); (2) vs the real --save-wq files: H bit-exact (it does not depend on
     the siblings); the decode differs from wq by the fp16 rounding of suh / svh (~4e-4 of W, deq.py check). Any change
     of the siblings re-draws the bf16 rounding of the rerun's outputs (~1e-3 of x per element) while eta is only
     ~0.1 of x, so C and N move by percents whatever the change: the floor is measured, a control run with the wq files
     multiplied by (1 + u), u ~ N(0, 3.6e-4^2) (the decode's own size). Pass: ||dC|| / ||C||, ||dN|| / ||N|| and
     ||dG|| / ||G|| (G = H + C + C^T + N, mkdenoise's metric) of the decode each <= 2x the control's, per group max;
  C. resume: layers 0-1 in one run, then 0-3 in the same OUT: layers 0-1 skipped, 2-3 = A's bit-exact; b02 deleted;
  D. --wins long = collect.py's window order (its list built here independently, as collect.py does);
  E. --snap-layers: the full sums unchanged (bit-exact vs a run without), half/ written at the flush boundary with
     n = the tokens of the first 32 windows.
Run where the 0.8B and its K3 base live (the 4070 box), CPU only:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python3 test_cpu.py /data/models/qwen35-08b /data/challenge/calib-meta \
      /data/yb/q-k3base/wq /data/yb/q-k3base /tmp/w12/t"""
import glob, json, os, shutil, subprocess, sys
import numpy as np, torch
from safetensors import safe_open
MODEL, META, WQ, LEVEL, WORK = sys.argv[1:6]
HERE = os.path.dirname(os.path.abspath(__file__))
MOM = os.path.join(HERE, "..", "..", "proxy2b", "exp", "w07-denoise", "moments.py")
SIB = os.path.join(HERE, "sibmom27.py")
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
ENV = dict(os.environ, DEV="cpu", P2B_BLOCK="causal_conv1d,fla")
ENV.setdefault("POOLS", os.path.join(HERE, "..", "..", "eval", "pools", "pools-v3.json"))
ARGS = ["--tokens", "2500", "--win", "512", "--seed", "3"]
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {info}", flush=True)
def run(script, out, *extra):
    r = subprocess.run([sys.executable, script, MODEL, META, out, *extra], env=ENV, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2500:])
    return r.returncode == 0, r.stderr
def load(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.input.safetensors")):
        with safe_open(f, "pt") as h: out[os.path.basename(f)[:-12]] = ({s: h.get_tensor(s) for s in "HCN"}, h.metadata())
    return out
def rel(a, b): return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))

R, S = f"{WORK}/ref", f"{WORK}/sib"
okr, er = run(MOM, R, "--mode", "sib", "--wq", WQ, "--layers", "0-3", *ARGS)
oks, es = run(SIB, S, "--wq", WQ, "--layers", "0-3", *ARGS)
check("A. both run", okr and oks)
ref, sib = load(R), load(S)
check("A. same groups (o/out + down of layers 0-3)", sorted(ref) == sorted(sib) and len(ref) == 8, f"{len(ref)} / {len(sib)}")
worst = max((rel(sib[g][0][s], ref[g][0][s]) for g in ref if g in sib for s in "HCN"), default=1.0)
check("A. H, C, N = moments.py's (tolerance 1e-6)", worst <= 1e-6, f"max rel {worst:.1e}")
check("A. n equal", all(sib[g][1]["n"] == ref[g][1]["n"] for g in ref if g in sib), ref[next(iter(ref))][1]["n"] if ref else "")
mk = ["n", "mode", "gamma", "sigma", "seed", "pools", "win", "wq", "model", "kernels"]
check("A. moments.py's metadata keys and values", all(all(sib[g][1].get(k) == ref[g][1].get(k) for k in mk) for g in ref if g in sib))
print("   attention:", [l for l in es.splitlines() if "attention" in l][:1])

D, DW = f"{WORK}/dec", f"{WORK}/decwq"
okd, _ = run(SIB, D, "--level", LEVEL, "--codebook", "i8", "--layers", "0-3", *ARGS)
kf = f"{WORK}/sibkeys.txt"
open(kf, "w").write("".join(f"model.language_model.layers.{l}.{s}\n" for l in range(4) for s in
    (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] if l % 4 == 3 else ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"])
    + ["mlp.gate_proj", "mlp.up_proj"]))
r = subprocess.run([sys.executable, os.path.join(HERE, "deq.py"), "wq", LEVEL, f"{WORK}/deqwq", kf], env=ENV, capture_output=True, text=True)
okw, _ = run(SIB, DW, "--wq", f"{WORK}/deqwq", "--layers", "0-3", *ARGS)
dec, decwq = load(D), load(DW)
check("B. --level runs", okd and sorted(dec) == sorted(sib))
if okd and okw and sorted(decwq) == sorted(dec):
    w1 = max(rel(dec[g][0][s], decwq[g][0][s]) for g in dec for s in "HCN")
    check("B1. --level = --wq on deq.py's files", w1 == 0, f"max rel {w1:.1e}")
else: check("B1. deq.py wq + --wq run", False, r.stderr[-500:])
if okd and sorted(dec) == sorted(sib):
    hx = max(rel(dec[g][0]["H"], sib[g][0]["H"]) for g in sib)
    fn = lambda t: float(t.double().norm())
    cn = {g: (fn(dec[g][0]["C"] - sib[g][0]["C"]) / fn(sib[g][0]["C"]), fn(dec[g][0]["N"] - sib[g][0]["N"]) / fn(sib[g][0]["N"]),
              fn((dec[g][0]["C"] - sib[g][0]["C"]) + (dec[g][0]["C"] - sib[g][0]["C"]).T + dec[g][0]["N"] - sib[g][0]["N"])
              / fn(sib[g][0]["H"] + sib[g][0]["C"] + sib[g][0]["C"].T + sib[g][0]["N"])) for g in sib}
    check("B2. H bit-exact vs the --save-wq siblings'", hx == 0, f"max rel {hx:.1e}")
    from safetensors.torch import save_file
    PW, PO = f"{WORK}/pertwq", f"{WORK}/pert"; os.makedirs(PW, exist_ok=True); gen = torch.Generator().manual_seed(0)
    for k in open(kf).read().split():
        with safe_open(f"{WQ}/{k}.safetensors", "pt") as h: w = h.get_tensor("weight").float()
        save_file({"weight": (w * (1 + 3.6e-4 * torch.randn(w.shape, generator=gen))).half().contiguous()}, f"{PW}/{k}.safetensors")
    okp, _ = run(SIB, PO, "--wq", PW, "--layers", "0-3", *ARGS); pert = load(PO)
    def dev3(x):
        return {g: (fn(x[g][0]["C"] - sib[g][0]["C"]) / fn(sib[g][0]["C"]), fn(x[g][0]["N"] - sib[g][0]["N"]) / fn(sib[g][0]["N"]),
                    fn((x[g][0]["C"] - sib[g][0]["C"]) + (x[g][0]["C"] - sib[g][0]["C"]).T + x[g][0]["N"] - sib[g][0]["N"])
                    / fn(sib[g][0]["H"] + sib[g][0]["C"] + sib[g][0]["C"].T + sib[g][0]["N"])) for g in sib}
    cn, cp = dev3(dec), dev3(pert) if okp else None
    fmt = lambda d: "; ".join(f"{g.split('layers.')[1]} C {c:.1e} N {n:.1e} G {gg:.1e}" for g, (c, n, gg) in d.items())
    print("   B2. decode vs --save-wq:", fmt(cn)); print("   B2. control (wq x (1 + 3.6e-4 noise)):", fmt(cp) if cp else "FAILED")
    mx = lambda d, j: max(v[j] for v in d.values())
    check("B2. decode's C / N / G deviations within 2x the measured floor", cp and all(mx(cn, j) <= 2 * mx(cp, j) for j in range(3)),
          "max C {:.1e} / {:.1e}, N {:.1e} / {:.1e}, G {:.1e} / {:.1e}".format(*(v for j in range(3) for v in (mx(cn, j), mx(cp, j) if cp else 0))))

C = f"{WORK}/res"
ok1, _ = run(SIB, C, "--wq", WQ, "--layers", "0-1", *ARGS)
ok2, e2 = run(SIB, C, "--wq", WQ, "--layers", "0-3", *ARGS)
res = load(C)
lines = [l for l in e2.splitlines() if l.startswith("layer ")]
check("C. resume runs, only layers 2-3 computed", ok1 and ok2 and [l.split(":")[0] for l in lines] == ["layer 2", "layer 3"], str([l.split(":")[0] for l in lines]))
w2 = max((rel(res[g][0][s], sib[g][0][s]) for g in sib if g in res for s in "HCN"), default=1.0)
check("C. resumed moments = A's", sorted(res) == sorted(sib) and w2 == 0, f"max rel {w2:.1e}")
check("C. consumed boundaries deleted", not os.path.exists(f"{C}/work/b02.bf16") and os.path.exists(f"{C}/work/b04.bf16"))

E = f"{WORK}/long"
okl, _ = run(SIB, E, "--wq", WQ, "--layers", "0-0", "--wins", "long", "--tokens", "5000", "--win", "512", "--seed", "3")
m = json.load(open(f"{META}/kld.json")); pools = json.load(open(ENV["POOLS"]))["pool"]
off = np.cumsum([0] + [d["tokens"] for d in m["docs"]]); cw = []
for i, d in enumerate(m["docs"]):              # collect.py's list
    if pools[i] != "qad": continue
    for s in range(0, d["tokens"] - 512 + 1, 512): cw.append(int(off[i] + s))
np.random.default_rng(3).shuffle(cw)
got = [w for w, _ in json.load(open(f"{E}/windows.json"))["wins"]] if okl else []
check("D. --wins long = collect.py's order", okl and got == cw[:len(got)] and len(got) == 10, f"{len(got)} windows")

F1, F2 = f"{WORK}/snap", f"{WORK}/nosnap"
SA = ["--wq", WQ, "--layers", "0-0", "--tokens", "6000", "--win", "128", "--seed", "3"]
okf1, _ = run(SIB, F1, *SA, "--snap-layers", "0"); okf2, _ = run(SIB, F2, *SA)
f1, f2, fh = load(F1), load(F2), load(f"{F1}/half")
check("E. snapshot runs", okf1 and okf2 and len(fh) == 2)
if okf1 and okf2 and len(fh) == 2:
    w3 = max(rel(f1[g][0][s], f2[g][0][s]) for g in f2 for s in "HCN")
    wl = json.load(open(f"{F1}/windows.json"))["wins"]; sa = max(32, int(round(len(wl) / 2 / 32)) * 32)
    check("E. full sums unchanged by the snapshot", w3 == 0, f"max rel {w3:.1e}")
    check("E. half n = tokens of the first snap windows", all(int(fh[g][1]["n"]) == sum(n for _, n in wl[:sa]) for g in fh),
          f"{len(wl)} windows, snapshot after {sa}, n {fh[next(iter(fh))][1]['n']}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
