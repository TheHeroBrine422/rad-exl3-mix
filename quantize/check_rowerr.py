#!/usr/bin/env python3
"""check_rowerr.py [FORK_ROOT] -- CPU test of ourq --save-wq's per-row error after v3/e3/settle/reval/ourq-rowerr.patch
(w46; applied by w90 11-08). Seconds, no GPU: the fork's real quantize.py (its in-place regularize, scaled LDL, LDLQ,
refit) on CPU through rxq's stand-in trellis kernels (pipeline/rxq/cpukern.py, RXQ_CPU_STANDIN=1), on a random
correlated H and a weight with uneven row / column scales, K4 i8, scaled_hessian, sigma_reg 0.025.
  premise  quantize_exl3 changes the weight tensor it is given (regularize works in place)
  old      ourq's save-wq arithmetic on that w (the code before the patch): sum err / sum ref ~ 1, not the proxy
  new      the same arithmetic on a copy taken before quantize_exl3 (the patch): sum err / sum ref = the proxy_err
           quantize_exl3 returns, within 5% (the proxy is taken under the damped H, rowerr under the raw H)
  same     wq and the stored tensors are identical with and without the copy (the patch changes no output)
  source   quantize/ourq.py holds the patch: the copy before quantize_exl3, w restored from it in the save-wq block
Last line "ROWERR CHECKS n/m PASS". FORK_ROOT default /workspace/forks/exllamav3-frac."""
import os, re, sys, tempfile

import torch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK = sys.argv[1] if len(sys.argv) > 1 else "/workspace/forks/exllamav3-frac"
os.environ["RXQ_CPU_STANDIN"] = "1"
os.environ.setdefault("RXQ_EXL3_SRC", os.path.join(FORK, "exllamav3", "exllamav3_ext"))
sys.path.insert(0, os.path.join(R, "pipeline"))
from rxq import engine  # noqa: E402

res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


torch.set_num_threads(2)
Q = engine.setup("cpu", codebook="i8", qtf2=True)
g = torch.Generator().manual_seed(11)
k, n = 512, 384                                            # (in_features, out_features), ourq's layout
X = torch.randn(8192, k, generator=g) @ (torch.eye(k) + 0.3 * torch.randn(k, k, generator=g) / k ** 0.5)
X *= torch.exp(torch.randn(k, generator=g) * 0.7)          # uneven input channels
Hraw = (X.T @ X).float()
W0 = (torch.randn(k, n, generator=g) * torch.exp(torch.randn(1, n, generator=g) * 0.5) * 0.02).float()


def H_data():
    return {"H": Hraw.clone(), "first_key": "test.input", "count": X.shape[0], "finalized": False,
            "num_total": X.shape[0], "inf_nan": torch.zeros(2, dtype=torch.long), "device": torch.device("cpu")}


def qa():
    return {"seed": 5, "K": 4, "devices": ["cpu"], "device_ratios": None, "apply_out_scales": True, "mul1": True,
            "debug_dir": tempfile.mkdtemp(prefix="rowerr-"), "scaled_hessian": True, "sigma_reg": 0.025}


def rowerr(w, wq):                                         # ourq's save-wq block, verbatim arithmetic
    H = Hraw; err, ref = [], []
    for c in range(0, w.shape[1], 8192):
        E = w[:, c:c + 8192] - wq[:, c:c + 8192]
        err.append(((H @ E) * E).sum(0).double().cpu())
        W = w[:, c:c + 8192]; ref.append(((H @ W) * W).sum(0).double().cpu())
    return torch.cat(err), torch.cat(ref)


# old: ourq before the patch
w = W0.clone()
wq1, perr1, t1 = Q.quantize_exl3(w, H_data(), qa(), True)
report("premise", not torch.equal(w, W0), f"quantize_exl3 changed its weight argument in place "
       f"(max |w - W0| {float((w - W0).abs().max()):.3g}, |W0| max {float(W0.abs().max()):.3g})")
e, r = rowerr(w, wq1.float()); old = float(e.sum() / r.sum())
report("old", old > 0.5, f"pre-patch arithmetic: sum err / sum ref {old:.4g} vs proxy_err {float(perr1):.4g} (broken, as on the 27B head)")
# new: the patch (a copy before quantize_exl3)
w = W0.clone(); w_orig = w.detach().cpu().clone()
wq2, perr2, t2 = Q.quantize_exl3(w, H_data(), qa(), True)
w = w_orig.to(wq2.device)
e, r = rowerr(w, wq2.float()); new = float(e.sum() / r.sum())
report("new", abs(new / float(perr2) - 1) < 0.05, f"patched arithmetic: sum err / sum ref {new:.4g} vs proxy_err {float(perr2):.4g} "
       f"(ratio {new / float(perr2):.4f}); every row err <= ref: {bool((e <= r).all())}")
same = torch.equal(wq1, wq2) and float(perr1) == float(perr2) and t1.keys() == t2.keys() and all(torch.equal(t1[x], t2[x]) for x in t1)
report("same", same, f"wq, proxy_err and the {len(t1)} stored tensors identical with and without the copy")
src = open(os.path.join(R, "quantize", "ourq.py")).read()
i_copy = src.find("w_orig = w.detach().cpu().clone() if a.save_wq else None")
i_q = src.find("quantize_exl3(w, Hk, qa, a.save_wq)")
i_sv = src.find("if a.save_wq:", i_q)
i_rest = src.find("w = w_orig.to(wq.device); del w_orig", i_sv)
i_err = src.find("E = w[:, c:c + 8192] - wq[:, c:c + 8192]", i_sv)
report("source", -1 < i_copy < i_q < i_sv < i_rest < i_err, f"ourq.py: copy at {i_copy}, quantize at {i_q}, save-wq block at {i_sv}, "
       f"restore at {i_rest}, row errors at {i_err} (offsets, in this order)")
print(f"ROWERR CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
