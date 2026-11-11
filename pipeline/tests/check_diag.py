#!/usr/bin/env python3
"""check_diag.py FORK [WORK] -- #139 job diagnostics and #138 refit modes (rxq/refit.py) on the CPU (rxq/cpukern.py's
stand-in kernels, the fork's real quantize.py; FORK = the exllamav3 fork with patch 0008+), w98 2026-11-09. A tiny
two-layer model (layer 2 GDN + MLP, layer 3 attention + MLP; random weights / Hessians / H_out, check_cpujob's shapes)
under the V3 settings (H_out blocks, tile CD on attn_qg, C9 on the writers, block order), K0.5 and K2.5 i8:
  port     refit_port() with both switches off == the fork's refit_scales (torch.equal: Q, su, sv; same errors)
  proxy    per tensor: sum(e1) / sum(r1) of the diag file == the logged proxy_err to 1e-6 relative (the logged check
           too), and the pre-refit sums == the fork's e_before
  direct   e1 per channel == (W - Wq) H (W - Wq)^T recomputed in fp64 from the raw Hessian file and the job's wq
           (f16): f1 and the normalized channel profile to 2e-3; e2 == h e1; the tile map sums to sum(e1) (1e-6, fp32 cells); su_pre / sv_pre shapes
  noop     the t-files (tensors and metadata) and C9 files are identical with RXQ_DIAG=0 (and no diag file then)
  c9       writers' post-C9 trajectory point == the decoded stored weight with svh * c, recomputed (2e-3)
  arms     RXQ_REFIT_ARMS=none,hout,undamped,hout+undamped: every arm's in-job f2 == f2 of its written wq in fp64
           (2e-3); the cur arm's wq == the stored weight decoded (f16 rounding); the t-files still identical
  modes    quant.refit = false / hout / undamped as settings: same trellis as refit = true (fixed codes); the stored
           decoded weight's f2 == the matching in-job arm's (1e-4); false stores the pre-refit scales (f16 of su_pre)
  ladder   the settings stamp is unchanged with refit = true, and changes with any other mode
Last line: "DIAG CHECKS n/m PASS"."""
import json, os, re, shutil, subprocess, sys, tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PIPE)
from rxq import config as C, deq, job, qwen35  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "--run":          # child: one job (one engine setup a process)
    spec = json.load(open(sys.argv[2]))
    job.run_quant(spec, sys.argv[3], device="cpu", save_wq=True)
    sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1] == "--port":         # child: refit_port vs the fork's refit_scales
    os.environ["RXQ_CPU_STANDIN"] = "1"
    from rxq import cpukern, refit
    Q = cpukern.load(os.path.dirname(os.path.dirname(os.path.abspath(os.environ["RXQ_EXL3_SRC"]))))
    g = torch.Generator().manual_seed(98); ok = True
    for k, n, damp in ((256, 384, 0.0244), (384, 256, 0.0), (128, 128, 0.167)):
        A = torch.randn(3 * k, k, generator=g) * torch.linspace(0.2, 2.0, k); H = (A.T @ A) / (3 * k)
        H.diagonal().add_(damp / (1 - damp) * H.diagonal().mean() if damp else 0.0)
        W = torch.randn(k, n, generator=g) * 0.02
        Qw = W + torch.randn(k, n, generator=g) * 0.003
        su = torch.randn(k, 1, generator=g).sign() * (1 + 0.1 * torch.rand(k, 1, generator=g))
        sv = torch.randn(1, n, generator=g).sign() * (1 + 0.1 * torch.rand(1, n, generator=g))
        a = Q.refit_scales(W.clone(), Qw.clone(), H.clone(), su.clone(), sv.clone(), damping=damp)
        b = refit.refit_port(W.clone(), Qw.clone(), H.clone(), su.clone(), sv.clone(), damping=damp)
        ok &= all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])) and a[3:] == b[3:]
        h = torch.rand(n, generator=g) + 0.05
        for und, w in ((True, None), (False, refit._w_out(h, 0.01, "cpu")), (True, refit._w_out(h, 0.01, "cpu"))):
            c = refit.refit_port(W.clone(), Qw.clone(), H.clone(), su.clone(), sv.clone(), damping=damp, undamped=und, w_out=w)
            ok &= bool(torch.isfinite(c[0]).all() and torch.isfinite(c[1]).all() and torch.isfinite(c[2]).all())
    print("PORT", "OK" if ok else "BAD"); sys.exit(0 if ok else 1)

FB = sys.argv[1]
W = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="w98diag-")
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


ENV = dict(os.environ, RXQ_CPU_STANDIN="1", RXQ_EXL3_SRC=f"{FB}/exllamav3/exllamav3_ext", PYTHONPATH=PIPE, OMP_NUM_THREADS="2")
r = subprocess.run([sys.executable, __file__, "--port"], env=ENV, capture_output=True, text=True)
report("port", r.returncode == 0 and "PORT OK" in r.stdout, (r.stdout + r.stderr)[-300:].strip())

# ---- the tiny model (check_cpujob's shapes and seed)
SH = {"ssm_qkv": (384, 256), "ssm_z": (256, 256), "ssm_out": (256, 256), "attn_qg": (256, 256), "attn_k": (128, 256),
      "attn_v": (128, 256), "attn_output": (256, 256), "ffn_gate": (384, 256), "ffn_up": (384, 256), "ffn_down": (256, 384)}
g = torch.Generator().manual_seed(86)
ck, hs, ho = f"{W}/ckpt", f"{W}/hess", f"{W}/hout/bn1920"
for d in (ck, hs, ho): os.makedirs(d, exist_ok=True)
tens, keys, Hraw, hdiag = {}, [], {}, {}
for l in (2, 3):
    for hname, ks in qwen35.layer_plan(l):
        k_in = SH[qwen35.family(ks[0])][1]
        A = torch.randn(4 * k_in, k_in, generator=g) * torch.linspace(0.3, 2.0, k_in)
        H = (A.T @ A).float(); iu = torch.triu_indices(k_in, k_in); Hraw[hname] = H.double()
        save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{hs}/{hname}.safetensors", metadata={"k": str(k_in), "rows": str(4 * k_in)})
        for key in ks:
            n, k = SH[qwen35.family(key)]
            tens[f"{key}.weight"] = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
            hdiag[key] = (torch.rand(n, generator=g) + 0.1).float()
            save_file({"hout_diag": hdiag[key]}, f"{ho}/{key}.safetensors")
            keys.append((l, hname, ks, key))
save_file(tens, f"{ck}/model.safetensors")

base = open(os.path.join(PIPE, "config", "v3.toml")).read()
base = re.sub(r'^ldlq = "(dense|blocks)"', 'ldlq = "blocks"', base, flags=re.M)
cfgs = {}
for m, v in (("true", "true"), ("false", "false"), ("hout", '"hout"'), ("undamped", '"undamped"')):
    t = re.sub(r'^refit = (true|false|"[a-z+]+")', f"refit = {v}", base, flags=re.M)
    open(f"{W}/{m}.toml", "w").write(t); cfgs[m] = f"{W}/{m}.toml"
stamps = {m: C.settings_stamp(C.load(p)) for m, p in cfgs.items()}
report("ladder", stamps["hout"] == C.settings_stamp(C.load()) and len(set(stamps.values())) == 4,
       f"refit = \"hout\" (the adopted setting, #138-03) gives the repo's stamp {C.settings_stamp(C.load())}; all four modes differ: {sorted(set(stamps.values()))}")

LEVELS = [(0.5, "i8"), (2.5, "i8")]
ARMS = "none,hout,undamped,hout+undamped"


def run(tag, cfgname, K, cb, extra_env=None):
    out = f"{W}/out-{tag}-K{K}"
    groups = []
    for l in (2, 3): groups += [list(ks) for _, ks in qwen35.layer_plan(l)]
    st = stamps[cfgname]
    for gi, ks in enumerate(groups):
        spec = {"type": "rxq.quant", "version": 1, "level": f"t-k{K}-{cb}", "K": K, "codebook": cb, "seed_offset": 0, "keys": ks,
                "settings_stamp": st, "config": cfgs[cfgname], "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "wpre": None, "hout": ho}}
        sp = f"{W}/spec-{tag}-K{K}-{gi}.json"; json.dump(spec, open(sp, "w"))
        r = subprocess.run([sys.executable, __file__, "--run", sp, out], env=dict(ENV, **(extra_env or {})), capture_output=True, text=True)
        if r.returncode: print(r.stdout[-1500:], r.stderr[-3000:]); return out, False
    return out, all(os.path.exists(job.tensor_file(out, k)) for _, _, _, k in keys)


def tfile(out, key):
    with safe_open(job.tensor_file(out, key), "pt") as f: return {n.rsplit(".", 1)[1]: f.get_tensor(n) for n in f.keys()}, f.metadata()


def same_files(a, b, prefix):
    fa = sorted(f for f in os.listdir(a) if f.startswith(prefix)); fb = sorted(f for f in os.listdir(b) if f.startswith(prefix))
    if fa != fb or not fa: return False
    for f in fa:
        with safe_open(f"{a}/{f}", "pt") as x, safe_open(f"{b}/{f}", "pt") as y:
            if x.metadata() != y.metadata() or sorted(x.keys()) != sorted(y.keys()): return False
            if not all(torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys()): return False
    return True


table = {cb: torch.from_numpy(deq.codebook_lut(cb).astype(np.float32)) for cb in ("i8",)}


def decoded(out, key, K, cb="i8", c=None):
    t, _ = tfile(out, key)
    svh = t["svh"].float() * (c.float() if c is not None else 1.0)
    return deq.weight(t["trellis"], t["suh"], svh.half(), K, table[cb]).double()        # (k, n)


def f2_of(Wq_kn, key, hname):
    """w81 f2calc's arithmetic: Wq (k, n) vs W, raw H, raw h -> (f2, f1, e per channel)"""
    Wt = tens[f"{key}.weight"].double().T; H = Hraw[hname]; h = hdiag[key].double()
    E = Wt - Wq_kn; e = (E * (H @ E)).sum(0); r = (Wt * (H @ Wt)).sum(0)
    return float((h * e).sum() / (h * r).sum()), float(e.sum() / r.sum()), e


for K, cb in LEVELS:
    out, ok = run("diag", "true", K, cb)
    off, ok2 = run("nodiag", "true", K, cb, {"RXQ_DIAG": "0"})
    arms, ok3 = run("arms", "true", K, cb, {"RXQ_REFIT_ARMS": ARMS, "RXQ_REFIT_ARMS_WQ": f"{W}/armwq-K{K}"})
    report(f"run-K{K}", ok and ok2 and ok3, f"diag / RXQ_DIAG=0 / arms runs, {len(keys)} tensors each")
    if not (ok and ok2 and ok3): continue
    recs = {json.loads(x)["key"]: json.loads(x) for x in open(f"{out}/log.jsonl")}
    worst_p, worst_pre, worst_d, worst_t, worst_c9, nbytes, shapes_ok, ov_ok, traj_ok = 0.0, 0.0, 0.0, 0.0, 0.0, 0, True, True, True
    for l, hname, ks, key in keys:
        rc = recs[key]; dgr = rc.get("diag") or {}
        with safe_open(job.diag_path(out, key), "pt") as f: d = {n.rsplit(".", 1)[1]: f.get_tensor(n).double() for n in f.keys()}
        nbytes += os.path.getsize(job.diag_path(out, key))
        n, k = SH[qwen35.family(key)]
        shapes_ok &= d["e1"].shape == (n,) and d["su_pre"].shape == (k,) and d["sv_pre"].shape == (n,) and d["tile"].shape == (k // 128, n // 128) \
            and set(d) == {"e1", "e2", "e1_pre", "e2_pre", "x1", "d1", "r1", "tile", "su_pre", "sv_pre"}
        p = float(d["e1"].sum() / d["r1"].sum())
        worst_p = max(worst_p, abs(p - rc["proxy_err"]) / rc["proxy_err"], dgr["check"]["rel"])
        worst_pre = max(worst_pre, dgr["check"]["pre_rel"])
        with safe_open(f"{out}/wq/{key}.safetensors", "pt") as f: wq = f.get_tensor("weight").double().T
        _, f1w, e = f2_of(wq, key, hname)                 # raw H (a sum over rows); the job's H is a mean: compare normalized
        en, e1n = e / e.sum(), d["e1"] / d["e1"].sum()
        worst_d = max(worst_d, abs(f1w / p - 1), float((en - e1n).abs().max() / e1n.abs().max()),
                      float((d["e2"] - hdiag[key].double() * d["e1"]).abs().max() / d["e2"].abs().max()))
        worst_t = max(worst_t, abs(float(d["tile"].sum() / d["e1"].sum()) - 1))
        ov = dgr.get("overload") or {}
        ov_ok &= all(0 <= ov.get(x, -1) <= 1 for x in ("q_outer", "w_over", "tile_3x")) and ov.get("q_absmax", 0) > 0
        tj = dgr.get("traj") or {}
        traj_ok &= {"pre_refit", "post_refit"} <= set(tj) and (("tilecd_q" in tj) == (qwen35.family(key) == "attn_qg"))
        if "c9" in rc:
            traj_ok &= "post_c9" in tj
            with safe_open(job.c9_path(out, key), "pt") as f: c = f.get_tensor("c")
            f2d, f1d, _ = f2_of(decoded(out, key, K, c=c), key, hname)
            worst_c9 = max(worst_c9, abs(tj["post_c9"]["f1"] / f1d - 1), abs(tj["post_c9"]["f2"] / f2d - 1))
    report(f"proxy-K{K}", worst_p <= 1e-6 and worst_pre <= 1e-6,
           f"max rel |sum e1 / sum r1 - proxy_err| = {worst_p:.2e} (rule 1e-6), pre-refit vs the fork's e_before {worst_pre:.2e}")
    report(f"direct-K{K}", worst_d < 2e-3 and worst_t < 1e-6 and shapes_ok,
           f"e1 vs fp64 recompute from wq + raw H: {worst_d:.2e}; tile map sum {worst_t:.1e}; shapes / names {shapes_ok}")
    report(f"traj-K{K}", traj_ok and ov_ok and worst_c9 < 2e-3,
           f"trajectory points (tile CD on attn_qg only) {traj_ok}, overload counters {ov_ok}, post-C9 vs decoded svh*c: {worst_c9:.2e}; "
           f"diag bytes {nbytes} for {len(keys)} tensors")
    noop = same_files(out, off, "t-") and same_files(out, off, "c9-") and same_files(out, arms, "t-") and same_files(out, arms, "c9-") \
        and not any(f.startswith("diag-") for f in os.listdir(off))
    report(f"noop-K{K}", noop, "t- / c9- files identical (tensors + metadata) with RXQ_DIAG=0 and with RXQ_REFIT_ARMS; no diag file when off")
    arecs = {json.loads(x)["key"]: json.loads(x) for x in open(f"{arms}/log.jsonl")}
    worst_a, worst_cur = 0.0, 0.0
    for l, hname, ks, key in keys:
        A = arecs[key]["diag"]["arms"]
        for name, v in A.items():
            with safe_open(f"{W}/armwq-K{K}/{name}/wq/{key}.safetensors", "pt") as f: x = f.get_tensor("weight").double().T
            f2x, f1x, _ = f2_of(x, key, hname)
            worst_a = max(worst_a, abs(v["f2"] / f2x - 1), abs(v["f1"] / f1x - 1))
        with safe_open(f"{W}/armwq-K{K}/cur/wq/{key}.safetensors", "pt") as f: x = f.get_tensor("weight").double().T
        dq = decoded(out, key, K)
        worst_cur = max(worst_cur, float((x - dq).norm() / dq.norm()))
    report(f"arms-K{K}", worst_a < 2e-3 and worst_cur < 2e-3 and set(A) == {"cur", *ARMS.split(",")},
           f"in-job arm f1 / f2 vs fp64 on the arm's wq {worst_a:.2e}; cur wq vs the stored weight decoded {worst_cur:.2e}")
    for m, arm in (("false", "none"), ("hout", "hout"), ("undamped", "undamped")):
        mo, okm = run(f"mode-{m}", m, K, cb)
        if not okm: report(f"mode-{m}-K{K}", False, "run failed"); continue
        same_codes, worst_m, pre_ok = True, 0.0, True
        for l, hname, ks, key in keys:
            ta, _ = tfile(mo, key); tb, _ = tfile(out, key)
            same_codes &= torch.equal(ta["trellis"], tb["trellis"])
            f2m, _, _ = f2_of(decoded(mo, key, K), key, hname)
            worst_m = max(worst_m, abs(f2m / arecs[key]["diag"]["arms"][arm]["f2"] - 1))
            if m == "false":
                with safe_open(job.diag_path(out, key), "pt") as f: sp, vp = f.get_tensor(f"{key}.su_pre"), f.get_tensor(f"{key}.sv_pre")
                pre_ok &= torch.equal(ta["suh"], sp.half()) and torch.equal(ta["svh"], vp.half())
        mrec = json.loads(open(f"{mo}/log.jsonl").readline())
        report(f"mode-{m}-K{K}", same_codes and worst_m < 1e-4 and pre_ok and mrec["diag"]["refit"] == {"false": False}.get(m, m),
               f"trellis == refit true's {same_codes}; stored f2 vs in-job arm '{arm}' {worst_m:.2e}; pre-refit scales stored {pre_ok}")
print(f"DIAG CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
