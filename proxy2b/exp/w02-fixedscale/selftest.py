#!/usr/bin/env python3
"""selftest.py ORIG_QUANTIZE.py PATCHED_QUANTIZE.py [ORIG_OURQ.py PATCHED_OURQ.py] -- w02 CPU self-test (synthetic data; output in selftest.txt).
 1  patch no-op: the patched refit_scales with default arguments is bit-identical to the original
 2  refitq.refit (h = None, damped H, 2 rounds) = the original refit_scales (same algorithm, other summation order)
 3  the patched refit_scales with out_weight / fit_undamped = refitq.refit with h / the raw H
 4  h-weighted: the h-weighted error falls every round and ends below the unweighted refit's; constant h = unweighted
 5  fixed-code identity: W_q = diag(su) M diag(sv) refit to diag(r) W_q diag(c) is the same as suh *= r, svh *= c
 6  refitq.py end to end on a fake checkpoint / Hessian / quant dir: --rounds 0 links (bytes identical), a refit run
    lowers the logged errors, the edited shard reproduces the new wq, --alpha all=1 restores each row's H-norm of an
    LS-fitted row
 7  swiglu_acc / swiglu_fit against a brute-force per-channel search
 8  ourq.py patch: refit options off leave the stamp settings and quant_args as before
PATCHED_* = copies with refit-exllamav3.patch / refit-ourq.patch applied (git apply -p1 in a copy of each tree)."""
import ast, json, os, subprocess, sys, tempfile
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fsutil as U
torch.manual_seed(0); torch.set_num_threads(2)
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}", flush=True)

def load_fn(path, name):
    src = open(path).read(); tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch}; exec(compile(ast.Module([fn], []), path, "exec"), ns); return ns[name]
orig = load_fn(sys.argv[1], "refit_scales"); new = load_fn(sys.argv[2], "refit_scales")
# refitq.refit without running the script: take the function from its source
rq = {"torch": torch}; tree = ast.parse(open(f"{HERE}/refitq.py").read())
exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "refit"], []), "refitq", "exec"), rq)
refit = rq["refit"]

def problem(k=256, n=384, noise=0.15):
    X = torch.randn(4 * k, k) * torch.linspace(0.2, 3, k); H = X.T @ X / X.shape[0]
    W = torch.randn(k, n) * torch.linspace(0.5, 2, n)
    Q = (W + noise * W.std() * torch.randn(k, n)) * (1 + 0.1 * torch.randn(k, 1)) * (1 + 0.1 * torch.randn(1, n))
    sig = 0.025; Hd = H + sig * H.diagonal().mean() * torch.eye(k)
    return W, Q, H, Hd, sig / (1 + sig)
def herr(W, Q, H, h=None):
    e, r = U.colerr(W, Q, H); h = torch.ones_like(e) if h is None else h.double(); return float((h * e).sum() / (h * r).sum())

W, Q, H, Hd, ds = problem()
su, sv = torch.ones(W.shape[0], 1), torch.ones(1, W.shape[1])
o = orig(W, Q.clone(), Hd, su, sv, damping=ds); p = new(W, Q.clone(), Hd, su, sv, damping=ds)
p2 = new(W, Q.clone(), Hd, su, sv, rounds=2, damping=ds, out_weight=None, fit_undamped=False)
check("1 patch no-op (defaults, explicit defaults)", all(torch.equal(x, y) for x, y in zip(o[:3], p[:3])) and o[3:] == p[3:]
      and all(torch.equal(x, y) for x, y in zip(o[:3], p2[:3])) and o[3:] == p2[3:])
Q2 = Q.clone(); r, c = refit(W, Q2, Hd, None, 2)
d = max(float((Q2 - o[0]).abs().max() / o[0].abs().max()), float((r - o[1].flatten()).abs().max()), float((c - o[2].flatten()).abs().max()))
check("2 refitq.refit = refit_scales", d < 1e-4, f"max rel diff {d:.1e}")

h = torch.rand(W.shape[1]) ** 3 * 5 + 0.05; h = h / h.mean()
p = new(W, Q.clone(), Hd, su, sv, rounds=4, damping=ds, out_weight=h, fit_undamped=True)
Q3 = Q.clone(); r, c = refit(W, Q3, H, h, 4)
d = float((Q3 - p[0]).abs().max() / p[0].abs().max())
check("3 patched out_weight + fit_undamped = refitq.refit(h, raw H)", d < 1e-4, f"max rel diff {d:.1e}")

hist, Qh = [herr(W, Q, H, h)], Q.clone()
for _ in range(6): refit(W, Qh, H, h, 1); hist.append(herr(W, Qh, H, h))
Qu = Q.clone(); refit(W, Qu, H, None, 6)
mono = all(b <= a * (1 + 1e-6) for a, b in zip(hist, hist[1:]))
check("4a h-weighted error falls every round", mono, " ".join(f"{x:.5f}" for x in hist))
check("4b h-weighted refit beats unweighted in the h metric", hist[-1] < herr(W, Qu, H, h),
      f"{hist[-1]:.5f} vs {herr(W, Qu, H, h):.5f} (one-sided {herr(W, Qh, H):.5f} vs {herr(W, Qu, H):.5f})")
Qc = Q.clone(); refit(W, Qc, H, torch.full((W.shape[1],), 3.7) / 3.7, 6)
check("4c constant h = unweighted", float((Qc - Qu).abs().max() / Qu.abs().max()) < 1e-5)

# 5: codes fixed: the refit of the dequantized weight is a change of the stored scales
M = torch.randn(256, 384); s_u, s_v = torch.rand(256) + 0.5, torch.rand(384) + 0.5
Wq = s_u[:, None] * M * s_v[None]; Q5 = Wq.clone(); r, c = refit(W, Q5, H, h, 3)
d = float(((s_u * r)[:, None] * M * (s_v * c)[None] - Q5).abs().max() / Q5.abs().max())
check("5 refit = new (suh, svh) on the same codes", d < 1e-5, f"{d:.1e}")

# 6: refitq.py end to end on fake files (one layer: GDN-less, a down_proj and a gate_proj)
with tempfile.TemporaryDirectory() as T:
    P = "model.language_model.layers.0"; ks = {f"{P}.mlp.gate_proj": (256, 384), f"{P}.mlp.down_proj": (384, 256)}
    os.makedirs(f"{T}/m"); os.makedirs(f"{T}/h"); os.makedirs(f"{T}/q/wq"); os.makedirs(f"{T}/ho")
    Ws, sh, px = {}, {}, []
    for key, (k, n) in ks.items():
        Wt = torch.randn(n, k).to(torch.bfloat16); Ws[key + ".weight"] = Wt
        X = torch.randn(3 * k, k) * torch.linspace(0.3, 2, k); Hs = X.T @ X; iu = torch.triu_indices(k, k)
        save_file({"hessian_upper": Hs[iu[0], iu[1]].contiguous()}, f"{T}/h/{U.group(key)}.safetensors",
                  metadata={"k": str(k), "rows": str(3 * k)})
        Mq = Wt.float().T + 0.2 * torch.randn(k, n); s1, s2 = torch.rand(k) + 0.5, torch.rand(n) + 0.5
        Mq = Mq / s1[:, None] / s2[None]
        sh[key + ".suh"], sh[key + ".svh"] = s1.half(), s2.half(); sh[key + ".trellis"] = torch.zeros(4, dtype=torch.int16)
        Wq_ = (sh[key + ".suh"].float()[:, None] * Mq * sh[key + ".svh"].float()[None])
        save_file({"weight": Wq_.T.half().contiguous()}, f"{T}/q/wq/{key}.safetensors")
        save_file({"hout_diag": torch.rand(n) + 0.1}, f"{T}/ho/{key}.safetensors")
        px.append({"key": key, "proxy_err": U.colerr(Wt.float().T, Wq_.half().float(), Hs / (3 * k))[0].sum().item() /
                   U.colerr(Wt.float().T, Wq_.half().float(), Hs / (3 * k))[1].sum().item(), "sigma_reg": 0.025})
    save_file(Ws, f"{T}/m/model.safetensors"); save_file(sh, f"{T}/q/q-L00.safetensors", metadata={"ourq_stamp": "x"})
    open(f"{T}/q/proxy.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in px))
    run = lambda dst, *ar: subprocess.run([sys.executable, f"{HERE}/refitq.py", f"{T}/q", f"{T}/{dst}", "--model", f"{T}/m",
                                           "--hess", f"{T}/h", "--device", "cpu", *ar], capture_output=True, text=True)
    r0 = run("d0")
    same = all(os.path.samefile(f"{T}/q/wq/{k}.safetensors", f"{T}/d0/wq/{k}.safetensors") for k in ks)
    check("6a --rounds 0: every tensor linked, unchanged", r0.returncode == 0 and same, r0.stderr.strip()[-200:])
    r1 = run("d1", "--rounds", "6", "--undamped", "--hout", f"{T}/ho", "--shards")
    recs = [json.loads(l) for l in open(f"{T}/d1/refit.jsonl")] if r1.returncode == 0 else []
    check("6b refit run: proxy reconstructed, h-weighted error falls", recs and all(
        abs(x["e_before"] / x["src_proxy"] - 1) < 1e-4 and x["eh_after"] < x["eh_before"] for x in recs),
          r1.stderr.strip()[-300:] + " " + "; ".join(f"{x['key'][-14:]} e {x['e_before']:.4f}->{x['e_after']:.4f} eh {x['eh_before']:.4f}->{x['eh_after']:.4f}" for x in recs))
    with safe_open(f"{T}/d1/q-L00.safetensors", "pt") as f: s2 = {k: f.get_tensor(k) for k in f.keys()}
    d = 0.0
    for key in ks:
        with safe_open(f"{T}/q/wq/{key}.safetensors", "pt") as f: w0 = f.get_tensor("weight").float().T
        with safe_open(f"{T}/d1/wq/{key}.safetensors", "pt") as f: w1 = f.get_tensor("weight").float().T
        Mq = w0 / sh[key + ".suh"].float()[:, None] / sh[key + ".svh"].float()[None]
        rec = s2[key + ".suh"].float()[:, None] * Mq * s2[key + ".svh"].float()[None]
        d = max(d, float((rec - w1).abs().max() / w1.abs().max()))
    check("6c edited shard (suh*r, svh*c) reproduces the new wq", d < 3e-3, f"max rel diff {d:.1e} (f16 scales)")
    r2 = run("d2", "--rounds", "1", "--undamped", "--alpha", "all=1")      # last step input-side: rows not LS-fitted;
    key = f"{P}.mlp.gate_proj"                                              # check the multiplier formula instead
    with safe_open(f"{T}/d2/scales/{key}.safetensors", "pt") as f: c2 = f.get_tensor("c")
    r3 = run("d3", "--rounds", "1", "--undamped")
    with safe_open(f"{T}/d3/scales/{key}.safetensors", "pt") as f: c3 = f.get_tensor("c")
    with safe_open(f"{T}/d3/wq/{key}.safetensors", "pt") as f: wq3 = f.get_tensor("weight").float().T
    Hk = U.hessian(f"{T}/h/{U.group(key)}.safetensors", "cpu"); e, rr = U.colerr(Ws[key + ".weight"].float().T, wq3, Hk)
    d = float(((c2 / c3) - (1 - e / rr).pow(-0.5).float()).abs().max())
    check("6d --alpha all=1: c *= (1 - e)^(-1/2)", r2.returncode == 0 and r3.returncode == 0 and d < 2e-3, f"max diff {d:.1e}")
    Wr = Ws[key + ".weight"].float().T; Ql = wq3 * ((wq3 * (Hk @ Wr)).sum(0) / (wq3 * (Hk @ wq3)).sum(0))[None]
    e, rr = U.colerr(Wr, Ql, Hk); nq = (Ql * (Hk @ Ql)).sum(0) * (1 - e / rr).pow(-1).float()
    check("6e alpha = 1 restores the H-norm of an LS-fitted row", float((nq / rr.float() - 1).abs().max()) < 1e-3)

# 7: SwiGLU joint fit vs brute force
J, Tn = 64, 3000
g = torch.randn(Tn, J) * 2; u = torch.randn(Tn, J); aa = torch.nn.functional.silu(g) * u
gq = g * (1 + 0.1 * torch.randn(1, J)) + 0.3 * torch.randn(Tn, J); uq = u * (1 + 0.1 * torch.randn(1, J)) + 0.2 * torch.randn(Tn, J)
grid = torch.arange(0.7, 1.3 + 0.005, 0.01); i1 = int(torch.argmin((grid - 1).abs()))
s = [torch.zeros(len(grid), J, dtype=torch.float64), torch.zeros(len(grid), J, dtype=torch.float64), torch.zeros(J, dtype=torch.float64)]
for b in range(0, Tn, 1000): U.swiglu_acc(s, aa[b:b + 1000], gq[b:b + 1000], uq[b:b + 1000], grid)
gi, cu, cu1 = U.swiglu_fit(s, i1)
res = []                                   # direct per-position residuals, c_u by least squares in float64
for cg in grid:
    p = (torch.nn.functional.silu(cg * gq) * uq).double(); cv = (aa.double() * p).sum(0) / (p * p).sum(0)
    res.append(((aa.double() - cv * p) ** 2).sum(0))
bad = int(((torch.stack(res).argmin(0) - gi).abs() > 1).sum())
rj, ra = U.swiglu_resid(s, gi, cu); rb, _ = U.swiglu_resid(s, torch.full_like(gi, i1), torch.ones(J, dtype=torch.float64))
direct = float(((aa - torch.nn.functional.silu(gq) * uq) ** 2).sum())
check("7a joint fit = brute-force grid argmin (within one step)", bad == 0, f"{bad}/{J} channels off")
check("7b residual sums = direct", abs(rb / direct - 1) < 1e-5 and rj < rb, f"rel resid base {rb / ra:.4f} joint {rj / ra:.4f}")

# 8: ourq.py patch, options off: same parsed settings, same stamp settings (keep), same quant_args
if len(sys.argv) > 4:
    def argblock(path):
        t = open(path).read(); a0 = t.index("ap = argparse.ArgumentParser()"); a1 = t.index("a = ap.parse_args()")
        g = {"argparse": __import__("argparse")}; exec(t[a0:a1] + "a = ap.parse_args(['-K', '3', '-o', 'x', '--save-wq'])", g)
        k0 = t.index("    keep = {k: v")
        g["vars"] = vars; exec("def _k(a):\n" + t[k0:t.index("    h.update(json.dumps(keep", k0)] + "    return keep\n", g)
        return g["_k"](g["a"])
    ko, kn = argblock(sys.argv[3]), argblock(sys.argv[4])
    check("8 ourq patch, refit options off: stamp settings unchanged", ko == kn, f"{sorted(set(kn) ^ set(ko))}")
    t = open(sys.argv[4]).read()
    check("8b ourq patch: every new quant_args entry is conditional on its option", all(x in t for x in (
        'if a.refit_rounds is not None: qa["refit_rounds"]', 'if a.refit_undamped: qa["refit_undamped"]', 'if rh and (')))
print("SELFTEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
