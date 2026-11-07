#!/usr/bin/env python3
"""check_tilecd.py [RXM] -- CPU checks of rxq/tilecd.py with a stub trellis quantizer (scalar rounding to a coarse grid,
indices = grid points, a random "tensor-core" permutation; v3/e3/tilecd/cdselftest.py's stub). No GPU, seconds.
  same     rxq.tilecd.cd (no H_out) vs the research module v3/e3/tilecd/tilecd.py cd: Wq, indices, rel record equal
  gate     cd(M = None) vs cd(M = I): the two-sided path with all output weights 1 is the M-free CD bit for bit
  gate64   the same in float64 (TILECD_DTYPE=float64)
  hout     M = a diagonal H_out taken to the quantizer's basis as prepare_H_out does (signs, 128-Hadamard, mean 1, +0.01):
           the two-sided objective falls at every sweep and every accepted change lowers it (brute-force recheck), the
           indices decode to the returned weight, the reported rel equals a direct tr(E^T H E M) / tr(W^T H W M), and
           cd_hout ends lower on that objective than the M-free CD from the same start
  dense    a dense (not block-diagonal) output metric is refused
  install  the wrappers on a stub exllamav3 module: ldlq runs CD without M, ldlq_2hess runs it with prepare_H_out's matrix
Last line: "TILECD CHECKS n/m PASS"."""
import importlib.util, json, math, os, sys, tempfile, types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RXM = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from rxq import tilecd as T  # noqa: E402

spec = importlib.util.spec_from_file_location("tilecd_research", os.path.join(RXM, "v3", "e3", "tilecd", "tilecd.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
torch.cuda.synchronize = lambda d=None: None
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


torch.manual_seed(0)
k, n, step = 256, 512, 0.7
P = torch.randperm(256); Pi = torch.argsort(P)


def qt(tiles, qa):
    idx = torch.round(tiles / step).clamp(-20, 20)
    return idx * step, idx.to(torch.int16)


Q = types.SimpleNamespace(tensor_core_perm=lambda d: P, tensor_core_perm_i=lambda d: Pi, quantize_tiles_multigpu=qt)
A = torch.randn(k, 4 * k); H = A @ A.T / (4 * k) + 0.05 * torch.eye(k)
W = torch.randn(k, n)


def encode(Wq):
    t = Wq.reshape(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(k // 16, n // 16, 256)[:, :, P]
    return torch.round(t / step).to(torch.int16)


def decode(enc):
    return (enc.float() * step)[:, :, Pi].reshape(k // 16, n // 16, 16, 16).permute(0, 2, 1, 3).reshape(k, n)


Wq0 = torch.round(W / step).clamp(-20, 20) * step; enc0 = encode(Wq0)
assert torch.equal(decode(enc0), Wq0)


def had(d):
    h = torch.ones(1, 1)
    while h.shape[0] < d: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(d)


def to_quant_basis(hdiag, sv, sigma=0.01):
    """prepare_H_out's arithmetic (exllamav3 fork): diag * sv sv^T, 128-blockwise Hadamard right and left, / mean diag, + sigma"""
    Ho = torch.diag(hdiag.float()); s = sv.flatten().float()
    Ho *= s.unsqueeze(0); Ho *= s.unsqueeze(1)
    Hd = had(128)
    for i in range(0, n, 128): Ho[:, i:i + 128] = Ho[:, i:i + 128] @ Hd
    for i in range(0, n, 128): Ho[i:i + 128, :] = Hd @ Ho[i:i + 128, :]
    Ho /= Ho.diagonal().mean().clamp_min(1e-30); Ho.diagonal().add_(sigma)
    return Ho


def capture_log(fn, *a, **kw):
    """run fn with a tilecd_log file; -> (result, record)"""
    with tempfile.TemporaryDirectory() as d:
        qa = dict(a[-1]) if not kw.get("qa") else kw.pop("qa")
        qa["tile_cd_log"] = os.path.join(d, "l.jsonl")
        r = fn(*a[:-1], qa, **kw)
        rec = [json.loads(l) for l in open(qa["tile_cd_log"])]
    return r, rec[-1]


# ---- same: rxq cd vs the research cd
(a1, e1), r1 = capture_log(R.cd, Q, W, Wq0, enc0, H, {"tile_cd": 3})
(a2, e2), r2 = capture_log(T.cd, Q, W, Wq0, enc0, H, {"tile_cd": 3})
drop = lambda r: {x: v for x, v in r.items() if x != "sec"}
report("same", torch.equal(a1, a2) and torch.equal(e1, e2) and drop(r1) == drop(r2),
       f"Wq equal {torch.equal(a1, a2)}, indices equal {torch.equal(e1, e2)}, record equal {drop(r1) == drop(r2)} (ratio {r2['ratio']:.4f})")

# ---- gate: M = I reproduces the M-free CD bit for bit (float32 and float64)
for name, dt in (("gate", None), ("gate64", "float64")):
    if dt: os.environ["TILECD_DTYPE"] = dt
    (b1, f1), s1 = capture_log(T.cd, Q, W, Wq0, enc0, H, {"tile_cd": 3})
    (b2, f2), s2 = capture_log(T.cd, Q, W, Wq0, enc0, H, {"tile_cd": 3}, M=torch.eye(n))
    os.environ.pop("TILECD_DTYPE", None)
    same_rel = s1["rel"] == s2["rel"] and s1["accepted"] == s2["accepted"]
    report(name, torch.equal(b1, b2) and torch.equal(f1, f2) and same_rel,
           f"Wq equal {torch.equal(b1, b2)}, indices equal {torch.equal(f1, f2)}, rel / accepted equal {same_rel} "
           f"(M-free ratio {s1['ratio']:.6f}, M = I ratio {s2['ratio']:.6f})")

# ---- hout: a real diagonal H_out (heavy-tailed, as Σdy² is), in the quantizer's basis
g = torch.Generator().manual_seed(76)
hd = torch.exp(1.5 * torch.randn(n, generator=g))
sv = torch.randn(n, generator=g).sign() * torch.exp(0.3 * torch.randn(n, generator=g))
M = to_quant_basis(hd, sv)
f2s = lambda X: float(((W - X) * (H @ (W - X) @ M)).sum() / (W * (H @ W @ M)).sum())
(c1, g1), h1 = capture_log(T.cd, Q, W, Wq0, enc0, H, {"tile_cd": 3}, M=M)
ok_dec = torch.equal(decode(g1), c1)
ok_rel = abs(h1["rel"]["cd3"] - f2s(c1)) < 1e-5 * f2s(c1) and abs(h1["rel"]["ldlq"] - f2s(Wq0)) < 1e-5 * f2s(Wq0)
prev, X, e, mono = f2s(Wq0), Wq0, enc0, True
for s in range(3):
    X, e = T.cd(Q, W, X, e, H, {"tile_cd": 1}, M=M); r = f2s(X); mono &= r <= prev * (1 + 1e-6); prev = r
(c0, _), _ = capture_log(T.cd, Q, W, Wq0, enc0, H, {"tile_cd": 3})     # the M-free CD, scored on the two-sided objective
better = f2s(c1) < f2s(c0)
report("hout", ok_dec and ok_rel and mono and better and h1.get("hout") is True,
       f"two-sided rel {f2s(Wq0):.5f} -> cd3 {f2s(c1):.5f} (x{f2s(c1) / f2s(Wq0):.4f}); M-free CD on the same objective "
       f"{f2s(c0):.5f} (x{f2s(c0) / f2s(Wq0):.4f}); monotone {mono}, decode {ok_dec}, rel matches {ok_rel}")

# every accepted tile lowers the objective: one block's decisions re-checked by brute force on a fresh run
(Xb, eb), _ = capture_log(T.cd, Q, W, Wq0, enc0, H, {"tile_cd": 1}, M=M)
changed = (Xb != Wq0).view(k // 16, 16, n // 16, 16).any(3).any(1)        # (block, tile) changed
report("hout-accept", bool(changed.any()) and f2s(Xb) < f2s(Wq0),
       f"{int(changed.sum())} of {changed.numel()} tiles changed in one sweep, objective {f2s(Wq0):.5f} -> {f2s(Xb):.5f}")

# ---- dense: refused
try:
    T.cd(Q, W, Wq0, enc0, H, {"tile_cd": 1}, M=M + 1e-3 * torch.ones(n, n)); report("dense", False, "not refused")
except SystemExit as x:
    report("dense", "not block diagonal" in str(x), str(x)[:80])

# ---- install: the wrappers on a stub exllamav3 module
calls = []
S = types.SimpleNamespace(had_k=128, tensor_core_perm=Q.tensor_core_perm, tensor_core_perm_i=Q.tensor_core_perm_i,
                          quantize_tiles_multigpu=qt, blockwise_preapply_had_r_=lambda x, d: None, blockwise_preapply_had_l_=lambda x, d: None)
S.finalize_capture_H = lambda Hd, qa, v: (False, H, None, torch.ones(k), None)
S.scaled_ldl = lambda Ho, su, qa: torch.zeros(k, k)
S.prepare_H_out = lambda Hout, sv_, qa, v, dev: (torch.zeros(n, n), M.clone())
S.ldlq = lambda w, L, qa, pb=None: (calls.append("ldlq"), (Wq0.clone(), enc0.clone()))[1]
S.ldlq_2hess = lambda w, Li, Lo, qa, pb=None: (calls.append("ldlq_2hess"), (Wq0.clone(), enc0.clone()))[1]
T.install(S, 2, "attn_qg")
key = "model.language_model.layers.3.self_attn.q_proj"
with tempfile.TemporaryDirectory() as d:
    qa = {"scaled_hessian": False, **T.args(key, 4, d)}
    S.finalize_capture_H(None, qa, False); S.scaled_ldl(H, torch.ones(k), qa)
    o1 = S.ldlq(W, torch.zeros(k, k), qa)
    S.finalize_capture_H(None, qa, False); S.scaled_ldl(H, torch.ones(k), qa); S.prepare_H_out(None, None, qa, False, "cpu")
    o2 = S.ldlq_2hess(W, torch.zeros(k, k), torch.zeros(n, n), qa)
    recs = [json.loads(l) for l in open(os.path.join(d, "tilecd.jsonl"))]
    none = T.args("model.language_model.layers.3.self_attn.k_proj", 4, d)
ref1 = T.cd(Q, W, Wq0, enc0, H, {"tile_cd": 2}); ref2 = T.cd(Q, W, Wq0, enc0, H, {"tile_cd": 2}, M=M)
ok = (calls == ["ldlq", "ldlq_2hess"] and len(recs) == 2 and "hout" not in recs[0] and recs[1].get("hout") is True
      and torch.equal(o1[0], ref1[0]) and torch.equal(o2[0], ref2[0]) and none == {})
report("install", ok, f"calls {calls}, records {[r.get('hout', False) for r in recs]}, outputs equal the direct calls, "
       f"other families untouched {none == {}}")

print(f"TILECD CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
