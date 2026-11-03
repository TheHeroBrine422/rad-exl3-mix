#!/usr/bin/env python3
"""headarms.py DUMPDIR CKPT OUTDIR ARM [ARM ...] [--sens-perm PERM.npy] -- exact head-only KL of head arms (headsim.py's
method: real head inputs, the bf16 head as reference, f32 logits), for comparisons of quants at equal bytes.

ARM: NAME=QDIR[:PERM.npy]            QDIR/wq/lm_head.safetensors ((V, H) f16 rows, in PERM's order if given: row i =
                                     vocab id PERM[i], as ourq.py --head-perm writes it; else vocabulary order)
     NAME=plan:PLAN.json:PLAN:QDIR:PERM.npy   a headslab.py plan built from QDIR/K<k>/wq (PERM = the donors' order)
Per arm OUTDIR/NAME.kl.f32 (KL per position) and NAME.top1.u8 are written once and reused (delete to recompute).
Units for the bootstrap: documents (DUMPDIR/doc.i32, headx.py dumps) or 512-position blocks (the calib-A dump).
OUTDIR/report.json: per arm and half (sel = even units, rep = odd units, all) mean KL, top-1 agreement, guard KL
(positions whose bf16 argmax is outside the GUARD most sensitive rows of --sens-perm), and every arm vs the first:
ratio, diff, unit-bootstrap SE, z, 95% CI. env HA_BLOCK (256), HA_GUARD (20000)."""
import json, os, sys
import numpy as np, torch
from safetensors import safe_open

args = [x for x in sys.argv[1:] if not x.startswith("--")]
SP = sys.argv[sys.argv.index("--sens-perm") + 1] if "--sens-perm" in sys.argv else None
if SP: args.remove(SP)
DUMP, CK, OUT, ARMS = args[0], args[1], args[2], args[3:]
BLK = int(os.environ.get("HA_BLOCK", 256)); GUARD = int(os.environ.get("HA_GUARD", 20000))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT, exist_ok=True)
H_, bf, _ = open(f"{DUMP}/336.meta").read().split(); H_ = int(H_)
X = torch.from_numpy(np.fromfile(f"{DUMP}/336.x", np.uint16).astype(np.int16).reshape(-1, H_))
X = X.view(torch.bfloat16 if bf == "1" else torch.float16); N = X.shape[0]
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: Wref = f.get_tensor("lm_head.weight").to(dev)
V = Wref.shape[0]
def mm(x, w):
    try: return torch.mm(x, w.T, out_dtype=torch.float32)
    except (TypeError, RuntimeError): return torch.mm(x.float(), w.float().T)
def load(p): 
    with safe_open(p, "pt") as f: return f.get_tensor("weight")
def head(spec):                                    # -> (V, H) f16 in vocabulary order, on the CPU
    if spec.startswith("plan:"):
        _, pf, pn, qd, pp = spec.split(":"); perm = torch.from_numpy(np.load(pp))
        kb = np.array(json.load(open(pf))["plans"][pn]["K_blocks"]); rowK = np.repeat(kb, 128)
        Wp = torch.empty(V, H_, dtype=torch.float16)
        for k in sorted(set(rowK.tolist())):
            m = torch.from_numpy(np.nonzero(rowK == k)[0]); Wp[m] = load(f"{qd}/K{k:g}/wq/lm_head.safetensors")[m]
    else:
        qd, _, pp = spec.partition(":"); Wp = load(f"{qd}/wq/lm_head.safetensors")
        perm = torch.from_numpy(np.load(pp)) if pp else None
    if perm is None: return Wp
    W = torch.empty_like(Wp); W[perm] = Wp; return W
rank = None
if SP:
    p = np.load(SP); rank = torch.empty(V, dtype=torch.long); rank[torch.from_numpy(p)] = torch.arange(len(p))
ref_top = OUT + "/ref.top.i32"
for a in ARMS:
    name, spec = a.split("=", 1)
    if os.path.exists(f"{OUT}/{name}.kl.f32"): continue
    W = head(spec).to(dev); kl = np.zeros(N, np.float32); t1 = np.zeros(N, np.uint8); top = np.zeros(N, np.int32)
    with torch.no_grad():
        for b0 in range(0, N, BLK):
            x = X[b0:b0 + BLK].to(dev)
            zr = mm(x.to(torch.bfloat16), Wref); lp = torch.log_softmax(zr, -1); tr = zr.argmax(-1)
            zq = mm(x.to(torch.float16), W)
            kl[b0:b0 + len(x)] = (lp.exp() * (lp - torch.log_softmax(zq, -1))).sum(-1).cpu().numpy()
            t1[b0:b0 + len(x)] = (zq.argmax(-1) == tr).cpu().numpy(); top[b0:b0 + len(x)] = tr.cpu().numpy()
    kl.tofile(f"{OUT}/{name}.kl.f32"); t1.tofile(f"{OUT}/{name}.top1.u8"); top.tofile(ref_top)
    print(f"{name}: KL {kl.mean():.6f} top1 {t1.mean():.4f}", flush=True); del W
unit = np.fromfile(f"{DUMP}/doc.i32", np.int32) if os.path.exists(f"{DUMP}/doc.i32") else np.arange(N) // 512
_, unit = np.unique(unit, return_inverse=True); nu = unit.max() + 1
top = torch.from_numpy(np.fromfile(ref_top, np.int32).astype(np.int64))
guard = (rank[top] >= GUARD).numpy() if rank is not None else np.zeros(N, bool)
names = [a.split("=", 1)[0] for a in ARMS]
KL = {n: np.fromfile(f"{OUT}/{n}.kl.f32", np.float32).astype(np.float64) for n in names}
T1 = {n: np.fromfile(f"{OUT}/{n}.top1.u8", np.uint8) for n in names}
rng = np.random.default_rng(0); rep = {}
for half, us in (("sel", np.arange(0, nu, 2)), ("rep", np.arange(1, nu, 2)), ("all", np.arange(nu))):
    m = np.isin(unit, us); u = unit[m]; cnt = np.bincount(u, minlength=nu)[us]
    S = {n: np.bincount(u, KL[n][m], minlength=nu)[us] for n in names}
    boot = rng.integers(0, len(us), (2000, len(us)))
    r = {}
    for n in names:
        r[n] = {"kl": float(KL[n][m].mean()), "top1": float(T1[n][m].mean()),
                "guard_kl": float(KL[n][m & guard].mean()) if (m & guard).any() else None, "n": int(m.sum()), "units": len(us)}
        if n != names[0]:
            d = (S[n][boot].sum(1) - S[names[0]][boot].sum(1)) / cnt[boot].sum(1)
            diff = r[n]["kl"] - r[names[0]]["kl"]
            r[n].update(vs=names[0], ratio=r[n]["kl"] / r[names[0]]["kl"] if r[names[0]]["kl"] > 0 else None, diff=diff, se=float(d.std()),
                        z=float(diff / d.std()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
            if r[n]["guard_kl"] and r[names[0]]["guard_kl"]: r[n]["guard_ratio"] = r[n]["guard_kl"] / r[names[0]]["guard_kl"]
    rep[half] = r
json.dump({"dump": DUMP, "N": N, "guard_rows": GUARD if rank is not None else None, "arms": dict(a.split("=", 1) for a in ARMS),
           "halves": rep}, open(f"{OUT}/report.json", "w"), indent=1)
for half in ("sel", "rep"):
    for n, r in rep[half].items():
        v = f"  x{r['ratio'] or 0:.3f} z {r['z']:+.1f}" if "z" in r else ""
        print(f"{half} {n:16s} KL {r['kl']:.6f} top1 {r['top1']:.4f} guard {r['guard_kl'] or 0:.6f}{v}")
