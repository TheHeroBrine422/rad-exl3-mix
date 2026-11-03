#!/usr/bin/env python3
"""test_mkdenoise.py WORK -- synthetic check of mkdenoise.py's algebra (Theorem 1 of 2610.07522 with ourq's damping):
x and a perturbation eta correlated with it (so C != 0), the calib-layout H built from the same samples. Cases:
  1. lam = rho = 0: G = H (f32 rounding), no wpre file;
  2. lam = rho = 0.5: G = H + lam^2 N + lam (C + C^T), no wpre file (F = I);
  3. lam 0.55, rho 0.5: for random W^1, W^2 the sample objective E||W^(x + lam eta) - W(x + rho eta)||^2 + alpha||W^ - W||^2
     differs by exactly the difference of tr[(W^ - WF) G' (W^ - WF)^T] (G' = G + alpha I, alpha = sig x mean diag G,
     sig-in on a layer-input group), and W F (= (Ft @ W^T)^T, as ourq.py --wpre applies it) minimises it;
  4. independent noise (C = 0, N diagonal): ||W F||_F < ||W||_F (Theorem 2's shrink)."""
import json, os, subprocess, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
WORK = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0); k, n, m = 256, 6000, 48
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {info}", flush=True)
def write(d, g, x, eta):
    os.makedirs(f"{d}/calib", exist_ok=True); os.makedirs(f"{d}/mom", exist_ok=True)
    H = x.T @ x; iu = torch.triu_indices(k, k)
    save_file({"hessian_upper": H[iu[0], iu[1]].float().contiguous()}, f"{d}/calib/{g}.safetensors", metadata={"k": str(k), "rows": str(n)})
    save_file({"H": H.float(), "C": (x.T @ eta).float(), "N": (eta.T @ eta).float()}, f"{d}/mom/{g}.safetensors", metadata={"n": str(n)})
def mk(d, lam, rho, out):
    r = subprocess.run([sys.executable, f"{HERE}/../mkdenoise.py", f"{d}/calib", f"{d}/mom", out, "--lam", str(lam), "--rho", str(rho)],
                       env=dict(os.environ, DEV="cpu"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
def G_of(out, g):
    with safe_open(f"{out}/hin/{g}.safetensors", "pt") as f: up = f.get_tensor("hessian_upper").double(); rows = int(f.metadata()["rows"])
    G = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k); G[iu[0], iu[1]] = up; return (G + G.triu(1).T) / rows
def Ft_of(out, g):
    with safe_open(f"{out}/wpre/{g}.safetensors", "pt") as f: return f.get_tensor("Ft").double(), float(f.metadata()["alpha"])

sc = torch.ones(k); sc[:4] = 20.0                                   # a few large channels, as real inputs have
MIX = torch.eye(k, dtype=torch.float64) + 0.1 * torch.randn(k, k, dtype=torch.float64)
x = (torch.randn(n, k, dtype=torch.float64) @ MIX) * sc
eta = 0.05 * x @ torch.randn(k, k, dtype=torch.float64) / k ** 0.5 + 0.3 * torch.randn(n, k, dtype=torch.float64) * sc
W = torch.randn(m, k, dtype=torch.float64)
for g, sg in (("model.language_model.layers.5.post_attention.input", 0.025), ("model.language_model.layers.5.input", 0.005)):
    d = f"{WORK}/{g.split('layers.')[1]}"; write(d, g, x, eta)
    Hm, Cm, Nm = x.T @ x / n, x.T @ eta / n, eta.T @ eta / n
    mk(d, 0, 0, f"{d}/o0"); G0 = G_of(f"{d}/o0", g)
    check(f"1. [{sg}] lam = rho = 0: G = H, no wpre", float((G0 - Hm).norm() / Hm.norm()) < 1e-6 and not os.path.exists(f"{d}/o0/wpre"))
    mk(d, 0.5, 0.5, f"{d}/o5"); G5 = G_of(f"{d}/o5", g); Gw = Hm + 0.25 * Nm + 0.5 * (Cm + Cm.T)
    check(f"2. [{sg}] lam = rho = 0.5: G = H + lam^2 N + lam (C + C^T), no wpre", float((G5 - Gw).norm() / Gw.norm()) < 1e-6 and not os.path.exists(f"{d}/o5/wpre"))
    lam, rho = 0.55, 0.5
    mk(d, lam, rho, f"{d}/o"); G = G_of(f"{d}/o", g); Ft, al = Ft_of(f"{d}/o", g)
    check(f"3. [{sg}] alpha = sig x mean diag G", abs(al / (sg * float(G.diagonal().mean())) - 1) < 1e-5, f"(alpha {al:.4e})")
    Gp = G + al * torch.eye(k, dtype=torch.float64); WF = (Ft @ W.T).T
    def J(Wh): return float(((x + lam * eta) @ Wh.T - (x + rho * eta) @ W.T).pow(2).sum() / n + al * (Wh - W).pow(2).sum())
    def Q(Wh): E = Wh - WF; return float((E @ Gp * E).sum())
    W1, W2 = W + 0.3 * torch.randn(m, k, dtype=torch.float64), 0.5 * W + 0.1 * torch.randn(m, k, dtype=torch.float64)
    dj, dq = J(W1) - J(W2), Q(W1) - Q(W2)
    check(f"3. [{sg}] objective differences = trace-form differences", abs(dj - dq) / abs(dj) < 1e-5, f"(dJ {dj:.6e}, dQ {dq:.6e})")
    js = [J(WF + 1e-3 * torch.randn(m, k, dtype=torch.float64)) - J(WF) for _ in range(4)]
    check(f"3. [{sg}] W F minimises the objective", min(js) > 0, f"(J(WF + small) - J(WF) min {min(js):.2e})")
# 4. independent noise: the shrink
eta2 = 0.3 * torch.randn(n, k, dtype=torch.float64) * sc; g = "model.language_model.layers.5.post_attention.input"
d = f"{WORK}/ind"; write(d, g, x, eta2); mk(d, 0.55, 0.3, f"{d}/o"); Ft, _ = Ft_of(f"{d}/o", g)
r = float((Ft @ W.T).norm() / W.norm())
check("4. independent noise: ||W F|| < ||W||", r < 1, f"(ratio {r:.4f})")
# 5. calib H from other samples
x2 = (torch.randn(2 * n, k, dtype=torch.float64) @ MIX) * sc                  # the same distribution, other samples
d = f"{WORK}/cv"; write(d, g, x, eta)
H2 = x2.T @ x2; iu = torch.triu_indices(k, k)
save_file({"hessian_upper": H2[iu[0], iu[1]].float().contiguous()}, f"{d}/calib/{g}.safetensors", metadata={"k": str(k), "rows": str(2 * n)})
mk(d, 0.55, 0.5, f"{d}/o"); Ft, al = Ft_of(f"{d}/o", g); G = G_of(f"{d}/o", g)
Hs, Cm, Nm = x.T @ x / n, x.T @ eta / n, eta.T @ eta / n
Gs = Hs + 0.3025 * Nm + 0.55 * (Cm + Cm.T); als = 0.025 * float(Gs.diagonal().mean())
A = Hs + 0.55 * Cm + 0.5 * Cm.T + 0.275 * Nm + als * torch.eye(k, dtype=torch.float64)
want = torch.linalg.solve(Gs + als * torch.eye(k, dtype=torch.float64), A.T)
check("5. F = the subset's own filter (calib H not in it)", float((Ft - want).norm() / want.norm()) < 1e-5 and abs(al / als - 1) < 1e-6)
Gw = H2 / (2 * n) + 0.3025 * Nm + 0.55 * (Cm + Cm.T)
st = json.loads(open(f"{d}/o/stats.jsonl").readline())
check("5. G = H_calib + lam^2 N + lam (C + C^T), not clipped", float((G - Gw).norm() / Gw.norm()) < 1e-6 and st["clip"] == 0, f"(hsub_vs_calib {st['hsub_vs_calib']:.3f})")
print("ALL PASS" if ok else "SOME FAIL")
