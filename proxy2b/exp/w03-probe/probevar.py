#!/usr/bin/env python3
"""probevar.py MODEL META POOLS OUT.json [--win 8] [--tok 1024] [--draws 8] [--threads 16] -- CPU variance study for A10 on
real 0.8B distributions: how much of a diagonal estimate's single-sequence variance is label-draw noise (what the probe
removes) and how much is window-to-window variation (what it cannot).

Test linear: the last layer's down_proj input x_t (3584) against the gradient at the final normed hidden state,
g_t = W_headᵀ d_t (1024; the RMSNorm Jacobian and the residual path are left out: one exact head backward), with d_t the
logit gradient of one draw: sampled label p - e_y, or the probe sqrt(p) xi - p (sqrt(p)·xi) (gauss / rad). Per window
and draw: Sketch B's diagonal D[j] = ‖Σ_t g_t[j] x_t‖² and the per-token Σ_t g_t[j]² ‖x_t‖². Windows: the first WIN of
collect.py's calib qad order (seed 0), first TOK tokens. Reported per estimator, relative to ‖D̄‖² (D̄: grand mean):
nu = label-draw variance of one sequence, omega = window-to-window variance (shared), and the effective-dose factor
(nu_sample + omega) / (nu_probe + omega): one probe sequence is worth that many sampled sequences."""
import argparse, json, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("pools"); ap.add_argument("out")
ap.add_argument("--win", type=int, default=8); ap.add_argument("--tok", type=int, default=1024)
ap.add_argument("--draws", type=int, default=8); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
meta = json.load(open(f"{a.meta}/kld.json")); toks = np.fromfile(f"{a.meta}/tokens.i32", np.int32)
pools = json.load(open(a.pools))["pool"]; off = np.cumsum([0] + [d["tokens"] for d in meta["docs"]]); wins = []
for i, d in enumerate(meta["docs"]):
    if pools[i] != "qad": continue
    for s in range(0, d["tokens"] - 2048 + 1, 2048): wins.append(int(off[i] + s))
np.random.default_rng(0).shuffle(wins)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
dec, W = model.get_decoder(), model.get_output_embeddings().weight.detach()         # [V, 1024]
cap = {}
dec.layers[-1].mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0][0]))
EST = ("sample", "gauss", "rad")
D = {k: [] for k in EST}; P = {k: [] for k in EST}            # per window: [draws, 1024]
ent = []; t0 = time.time()
with torch.no_grad():
    for w in range(a.win):
        ids = torch.from_numpy(toks[wins[w]:wins[w] + a.tok].astype(np.int64))[None]
        hid = dec(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]; x = cap["x"][:-1]
        p = torch.softmax(hid @ W.T, -1); sp = p.sqrt(); pW = p @ W; x2 = x.pow(2).sum(1)
        ent.append(float(-(p * p.clamp_min(1e-30).log()).sum(1).mean()))
        for k in EST:
            dd, pp = [], []
            for r in range(a.draws):
                if k == "sample":
                    y = torch.multinomial(p, 1)[:, 0]; g = pW - W[y]
                else:
                    xi = torch.randn_like(p) if k == "gauss" else torch.randint(0, 2, p.shape, dtype=torch.float32) * 2 - 1
                    xi.mul_(sp); xi.addcmul_(p, xi.sum(1, keepdim=True), value=-1.0); g = xi @ W; del xi
                dd.append((g.T @ x).pow(2).sum(1).double()); pp.append((g.pow(2) * x2[:, None]).sum(0).double())
            D[k].append(torch.stack(dd)); P[k].append(torch.stack(pp))
        print(f"window {w + 1}/{a.win}: mean entropy {ent[-1]:.2f} nats, {time.time() - t0:.0f}s", flush=True)
res = {"windows": a.win, "tokens": a.tok, "draws": a.draws, "mean_entropy": float(np.mean(ent))}
for form, X in (("sketchB", D), ("pertoken", P)):
    A = {k: torch.stack(v) for k, v in X.items()}                          # [win, draws, 1024]
    grand = torch.cat([v.reshape(-1, v.shape[-1]) for v in A.values()]).mean(0); n2 = float(grand.pow(2).sum())
    cos = lambda u, v: float((u * v).sum() / (u.norm() * v.norm()))
    nu = {k: float(v.var(1, unbiased=True).sum(-1).mean()) / n2 for k, v in A.items()}
    wm = {k: v.mean(1) for k, v in A.items()}                              # per-window means
    om = {k: float(wm[k].var(0, unbiased=True).sum()) / n2 - nu[k] / a.draws for k in A}
    omega = float(np.mean(list(om.values())))
    r = {"nu": nu, "omega_by_est": om, "omega": omega,
         "bias_cos_vs_sample": {k: cos(A[k].mean((0, 1)), A["sample"].mean((0, 1))) for k in A},
         "dose_factor": {k: (nu["sample"] + omega) / (nu[k] + omega) for k in A},
         "draw_only_factor": {k: nu["sample"] / nu[k] for k in A}}
    res[form] = r
    print(f"{form}: nu {', '.join(f'{k} {v:.3f}' for k, v in nu.items())}; omega {omega:.3f}; dose factor vs sampled "
          f"{', '.join(f'{k} {v:.2f}' for k, v in r['dose_factor'].items())}; means cos vs sampled "
          f"{', '.join(f'{k} {v:.4f}' for k, v in r['bias_cos_vs_sample'].items())}")
json.dump(res, open(a.out, "w"), indent=1)
