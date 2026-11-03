#!/usr/bin/env python3
"""swiglufit.py MODEL SRC GOLDEN DST_JOINT DST_UP [--grid 0.6:1.4:0.01] [--cap 4096] -- ranked.md C10: gate / up output
scales fitted jointly through the SwiGLU, codes fixed (one GPU, 0.8B).

refit_scales fits gate and up each to its own output in the input-Hessian metric; what down_proj reads is
a = silu(g) * u. Per MLP channel j, with the bf16 model's own MLP inputs x (forward hooks; GOLDEN's documents, i.e.
the fit pool for g08-fit1), g = W_g x, u = W_u x and their quantized ĝ, û (SRC/wq):
    min over (c_g, c_u) of  sum_t (silu(g_t) u_t - c_u silu(c_g ĝ_t) û_t)^2
c_u is closed form for each c_g (sum a p / sum p², p = silu(c_g ĝ) û), c_g a grid search: the sums
sum a p, sum p² per (grid point, channel) and sum a² per channel are accumulated over all positions, so nothing
is stored per token. Channels are separable (the objective ignores down_proj's cross-channel terms), so no weights.
Writes DST_JOINT (gate rows x c_g, up rows x c_u) and DST_UP (c_g = 1, up rows x c_u(1)): wq for evalq.py, scales/
as refitq.py, swiglu.jsonl per layer with the relative residual sum (a - p)² / sum a² of: the stored quant, up-only,
joint (in-sample), and joint / up-only fitted on even documents scored on odd ones (held-out check)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b")
import fsutil as U

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("src"); ap.add_argument("golden"); ap.add_argument("dst_joint"); ap.add_argument("dst_up")
ap.add_argument("--grid", default="0.6:1.4:0.01"); ap.add_argument("--cap", type=int, default=4096)
ap.add_argument("--maxdocs", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--dry", action="store_true", help="no files: the per-layer residual table only (CPU estimate)")
a = ap.parse_args()
dev = torch.device(a.device); U.cap_vram()
g0, g1, gs = map(float, a.grid.split(":")); GRID = torch.arange(g0, g1 + gs / 2, gs, device=dev); NG = len(GRID)
i1 = int(torch.argmin((GRID - 1).abs()))
assert abs(float(GRID[i1]) - 1) < 1e-6, "grid must contain 1.0"

from common import load_model
model, dec, _ = load_model(a.model, dev=a.device)
m = json.load(open(f"{a.golden}/kld.json")); toks = np.memmap(f"{a.golden}/tokens.i32", np.int32, "r")
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
NL = len(dec.layers); P = "model.language_model.layers"
WQ = {}                                    # layer -> (gate, up) quantized, (out, in) fp32 on the card
for l in range(NL):
    WQ[l] = tuple(U.load_wq(a.src, f"{P}.{l}.mlp.{n}", dev).T.contiguous() for n in ("gate_proj", "up_proj"))
NJ = WQ[0][0].shape[0]
# sums per (half, layer): S_ap [NG, J], S_pp [NG, J], S_aa [J]; float64
S = {(hf, l): [torch.zeros(NG, NJ, dtype=torch.float64, device=dev), torch.zeros(NG, NJ, dtype=torch.float64, device=dev),
               torch.zeros(NJ, dtype=torch.float64, device=dev)] for hf in (0, 1) for l in range(NL)}
cur = {"half": 0}
silu = torch.nn.functional.silu
def hook(l):
    mlp = dec.layers[l].mlp
    def h(mod, inp):
        x = inp[0][0].float()                              # [T, hidden]
        g = x @ mlp.gate_proj.weight.float().T; u = x @ mlp.up_proj.weight.float().T
        aa = silu(g) * u; del g, u
        gq = x @ WQ[l][0].T; uq = x @ WQ[l][1].T
        U.swiglu_acc(S[(cur["half"], l)], aa, gq, uq, GRID)
    return h
for l in range(NL): dec.layers[l].mlp.register_forward_pre_hook(hook(l))
nd = len(m["docs"]) if not a.maxdocs else min(a.maxdocs, len(m["docs"]))
t0 = time.time(); ntok = 0
with torch.no_grad():
    for i in range(nd):
        tk = np.asarray(toks[toff[i]:toff[i + 1]])[:a.cap]; cur["half"] = i % 2; ntok += len(tk)
        dec(input_ids=torch.from_numpy(tk.astype(np.int64))[None].to(dev))
        if i % 20 == 0: print(f"{i + 1}/{nd} docs {ntok} tokens {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

fit, resid = U.swiglu_fit, U.swiglu_resid

if not a.dry:
    for d in (a.dst_joint, a.dst_up): os.makedirs(f"{d}/wq", exist_ok=True)
logj = open(f"{a.dst_joint}/swiglu.jsonl", "w") if not a.dry else open(os.devnull, "w")
CLAMP = (0.5, 2.0)
done = set()
for l in range(NL):
    tot = [x + y for x, y in zip(S[(0, l)], S[(1, l)])]
    gi, cu, cu1 = fit(tot, i1)
    one = torch.full_like(gi, i1)
    cu = cu.clamp(*CLAMP); cu1 = cu1.clamp(*CLAMP)
    rb, aa = resid(tot, one, torch.ones_like(cu)); ru, _ = resid(tot, one, cu1); rj, _ = resid(tot, gi, cu)
    ge, cue, cu1e = fit(S[(0, l)], i1)                         # fitted on even docs, scored on odd
    rbo, aao = resid(S[(1, l)], one, torch.ones_like(cu)); ruo, _ = resid(S[(1, l)], one, cu1e.clamp(*CLAMP))
    rjo, _ = resid(S[(1, l)], ge, cue.clamp(*CLAMP))
    cg = GRID[gi]
    rec = {"layer": l, "rel_base": rb / aa, "rel_up": ru / aa, "rel_joint": rj / aa,
           "heldout_base": rbo / aao, "heldout_up": ruo / aao, "heldout_joint": rjo / aao,
           "cg_mean": float(cg.mean()), "cg_sd": float(cg.std()), "cu_mean": float(cu.mean()), "cu_sd": float(cu.std()),
           "cg_at_edge": int(((gi == 0) | (gi == NG - 1)).sum())}
    logj.write(json.dumps(rec) + "\n"); logj.flush()
    print(f"L{l:02d} rel resid base {rb / aa:.4e} up {ru / aa:.4e} joint {rj / aa:.4e} | held-out {rbo / aao:.4e} "
          f"{ruo / aao:.4e} {rjo / aao:.4e} | c_g {float(cg.mean()):.3f}±{float(cg.std()):.3f} edge {rec['cg_at_edge']}", flush=True)
    kg, ku = f"{P}.{l}.mlp.gate_proj", f"{P}.{l}.mlp.up_proj"
    if a.dry: continue
    for dst, mg, mu in ((a.dst_joint, cg.float(), cu.float()), (a.dst_up, torch.ones_like(cu1).float(), cu1.float())):
        for key, mult, wq in ((kg, mg, WQ[l][0]), (ku, mu, WQ[l][1])):
            U.write_wq(dst, key, (wq * mult.view(-1, 1)).T)        # rows of (out, in) = output channels
            U.write_scales(dst, key, torch.ones(wq.shape[1]), mult)
    done |= {kg, ku}
for dst in (a.dst_joint, a.dst_up) if not a.dry else ():
    for key in U.wq_keys(a.src):
        if key not in done: U.link_wq(a.src, dst, key)
    json.dump({"src": a.src, "golden": a.golden, "grid": a.grid, "docs": nd, "tokens": ntok, "cap": a.cap},
              open(f"{dst}/swiglufit.json", "w"), indent=1)
    open(f"{dst}/.done", "w").close()
print(f"swiglufit done: {nd} docs, {ntok} tokens, {time.time() - t0:.0f}s", file=sys.stderr)
