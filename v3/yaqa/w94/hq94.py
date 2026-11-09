#!/usr/bin/env python3
"""hq94.py -- CPU proxy of the 27B LM head's quantization for A6's settings (#125, RULES.md), colo.

The head is quantized as ourq does it (rows in sensitivity order, --head-perm; exllamav3's regularize: row RMS, random
row signs, 128-Hadamard over rows, input-channel RMS su over the whole tensor with random signs, input 128-Hadamard;
the global scale from exllamav3's search with ourq's drift; LDLQ last-to-first under P diag(su)(H + delta I) diag(su) P,
delta = 0.025 x mean diag (today's head donors); per-row closed-form scale refit), but only on BLOCKS (64 aligned
128-row blocks, a stratified sample, blocks94.json) and with the CPU bitshift trellis of v3/e3 (mul1x_i8, free start:
4-6% optimistic in absolute error, ratios between arms unaffected). Two-sided arms: w87's ldlq2 with the block's
H_out = P diag(F r^2) P / mean_all(F r^2) + sigma_out I, F = the softmax-Fisher diagonal fisher_qad_gen (headsens.npz)
in the same row order -- the objective of fork patch 0008's block-diagonal path (#115). One-sided = ldlq2 with H_out = I.

  hq94.py prep VARIANT OUT          raw | mr (mean row removed) | mr-r4 / mr-r8 (then the top-4 / -8 principal
                                    components of the rows, stored in bf16, removed): row RMS, su, signs, the scale-search
                                    tile sample and the sampled blocks' target rows -> OUT/prep-VARIANT.pt
  hq94.py scale VARIANT K OUT       exllamav3's global scale -> OUT/scale-VARIANT-K.json
  hq94.py quant VARIANT ARM K C OUT ARM one | two:SIGMA_OUT; C = chunk (8 blocks); per block the stored rows minus the
                                    target, D = Q - R (f32 [128, 5120]) -> OUT/d/VARIANT/ARM/K/b<block>.npy
  hq94.py real K WQROWS.npy OUT     the real donor's rows on the blocks (gpubox K<k>/wq, extracted): D = Wq - W
  hq94.py score OUT SET DUMPDIR SIM.npz TAG...   head-only KL per dump row of each TAG (VARIANT/ARM/K or real/K) on
                                    the blocks, by stratum -> OUT/kl/SET/<tag>.npz (kA, kB, kC; est = kA + wB kB + wC kC)
KL with only the rows V_S changed by dz (z' = z + dz on V_S): KL(p||q) = -sum_S p dz + log(1 + sum_S p (e^dz - 1)),
exact; p_v = exp(z_v - lse) with lse of the full bf16 head (headsim94.py). Strata scale additively (cross terms between
strata dropped: second order in the small tail probabilities). Env: CK, HESS (lm_head.input.safetensors), HS (headsens.npz),
PERM, BLOCKS (blocks94.json), OMP threads 1 (one job per core)."""
import json, math, os, sys, time
import numpy as np
import torch
from safetensors import safe_open

_h = os.path.dirname(os.path.abspath(__file__))
_s = os.path.join(_h, "..", "..", "e3", "settle")
sys.path[:0] = [os.path.join(_s, "w87"), os.path.join(_s, "w77"), os.path.join(_s, "w64"), _s, os.path.join(_s, "..")]
import settle87, settle77, settle, trellis, trellis2       # settle77 patches the drift (ourq's) and the codebooks
torch.set_grad_enabled(False)
CK = os.environ.get("CK", "/data/models/qwen38-27b-bf16")
HESS = os.environ.get("HESS", "/data/challenge/hessians/calib/lm_head.input.safetensors")
SIGMA, CB, RB, CH = 0.025, "mul1x_i8", 128, 16384
V, HID = 248320, 5120


def load_head():
    idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
    with safe_open(f"{CK}/{idx}", "pt") as f: W = f.get_tensor("lm_head.weight")
    return W[torch.from_numpy(np.load(os.environ["PERM"]))]          # bf16, sensitivity order


def blocks():
    B = json.load(open(os.environ["BLOCKS"])); return B


def rows_of(bl): return (torch.tensor(bl)[:, None] * RB + torch.arange(RB)).reshape(-1)


def prep(variant, out):
    t0 = time.time(); W = load_head(); B = blocks(); S = rows_of(B["all"])
    mr = variant.startswith("mr"); r_ = int(variant.split("-r")[1]) if "-r" in variant else 0
    g = torch.Generator().manual_seed(94)
    sv = (torch.randint(0, 2, (V,), generator=g) * 2 - 1).double()
    su_sign = (torch.randint(0, 2, (HID,), generator=g) * 2 - 1).double()
    m = torch.zeros(HID, dtype=torch.float64)
    if mr:
        for c in range(0, V, CH): m += W[c:c + CH].double().sum(0)
        m /= V
    Vb = torch.zeros(HID, 0, dtype=torch.float64)
    if r_:
        G = torch.zeros(HID, HID, dtype=torch.float64)
        for c in range(0, V, CH):
            X = W[c:c + CH].double() - m; G += X.T @ X
        ev, evec = torch.linalg.eigh(G); del G
        Vb = evec[:, -r_:].flip(1).to(torch.bfloat16).double()       # stored in bf16
        expl = float(ev[-r_:].sum() / ev.sum())
    def resid(c, n=CH):                                               # rows c..c+n of the quantized target R
        X = W[c:c + n].double() - m
        if r_: X = X - (X @ Vb).to(torch.bfloat16).double() @ Vb.T   # the factor A = X V stored in bf16 too
        return X
    r = torch.zeros(V, dtype=torch.float64); colsq = torch.zeros(HID, dtype=torch.float64)
    for c in range(0, V, CH):
        X = resid(c); rr = X.pow(2).mean(1).sqrt().clamp(min=1e-12); r[c:c + CH] = rr
        colsq += (X / rr[:, None]).pow(2).sum(0)
    su = (colsq / V).sqrt(); su[su < 1e-30] = 0.1; su = su * su_sign
    tn, tk = V // 16, HID // 16                                       # exllamav3's tile sample for the scale search
    ms = torch.zeros(tn, tk)
    def xf(c):
        X = resid(c) / r[c:c + CH, None]
        return settle.had_cols(settle.had_rows((X * sv[c:c + CH, None]).float()) / su.float())
    for c in range(0, V, CH):
        ms[c // 16:(c + CH) // 16] = xf(c).reshape(-1, 16, tk, 16).square().mean(dim=(1, 3))
    dl = max(tk, tn); ii = torch.arange(dl).repeat_interleave(3); ww = torch.arange(3).repeat(dl)
    kk = ii % tk; nn = (ii + ww) % tn
    msf = ms.flatten(); nx = min(max(8, dl * 3 // 16), (msf.numel() + 1) // 2)
    ex = torch.cat((torch.topk(msf, nx).indices, torch.topk(msf, nx, largest=False).indices))
    tr, tc = torch.cat((nn, ex // tk)), torch.cat((kk, ex % tk))
    rng = np.random.default_rng(1794)
    keep = np.sort(rng.choice(len(tr), 2304, replace=False)) if len(tr) > 2304 else np.arange(len(tr))
    tr, tc = tr[keep], tc[keep]; T = np.zeros((len(tr), 256), np.float32)
    for c in range(0, V, CH):
        sel = ((tr * 16 >= c) & (tr * 16 < c + CH)).nonzero().flatten()
        if len(sel) == 0: continue
        w4 = xf(c).reshape(-1, 16, tk, 16)
        T[sel.numpy()] = w4[tr[sel] - c // 16, :, tc[sel], :].reshape(-1, 256).numpy()
    R_S = torch.cat([resid(int(b) * RB, RB) for b in B["all"]]).float()
    W_S = W[S].float()
    o = {"variant": variant, "su": su, "r": r, "sv": sv, "m": m, "Vb": Vb, "T": T, "R_S": R_S, "W_S": W_S,
         "blocks": B["all"], "sec": time.time() - t0}
    if r_: o["explained"] = expl
    torch.save(o, f"{out}/prep-{variant}.pt.part"); os.replace(f"{out}/prep-{variant}.pt.part", f"{out}/prep-{variant}.pt")
    print(variant, f"{time.time() - t0:.0f}s", {k: o[k] for k in ("explained",) if k in o})


def load_prep(variant, out): return torch.load(f"{out}/prep-{variant}.pt", weights_only=False)


def scale(variant, K, out):
    f = f"{out}/scale-{variant}-{K}.json"
    if os.path.exists(f): return
    p = load_prep(variant, out); c = np.ascontiguousarray(trellis2.codebook(CB), dtype=np.float32)
    t0 = time.time(); s, sx = settle.search_scale(p["T"], c, float(K))
    json.dump({"s_exl": s, "search": sx, "drift": settle77.drift(float(K)), "sec": round(time.time() - t0, 1)}, open(f + ".part", "w"))
    os.replace(f + ".part", f)


def hessian():
    with safe_open(HESS, "pt") as f:
        k = int(f.metadata()["k"]); up = f.get_tensor("hessian_upper").double()
    H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k)
    H[iu[0], iu[1]] = up; return H + H.T - torch.diag(H.diagonal())


def quant(variant, arm, K, chunk, out):
    B = blocks(); bl = B["all"][int(chunk) * 8:(int(chunk) + 1) * 8]
    od = f"{out}/d/{variant}/{arm}/K{K}"; os.makedirs(od, exist_ok=True)
    todo = [b for b in bl if not os.path.exists(f"{od}/b{b}.npy")]
    if not todo: return
    p = load_prep(variant, out); s = json.load(open(f"{out}/scale-{variant}-{K}.json"))["s_exl"]
    H = hessian(); dl = SIGMA * H.diagonal().mean()
    Hd = H.clone(); dead = Hd.diagonal() == 0; Hd[dead, dead] = 1; Hd.diagonal().add_(dl)
    Hd *= p["su"][None, :]; Hd *= p["su"][:, None]
    Hd = settle.had_cols(Hd); Hd = settle.had_cols(Hd.T.contiguous()).T.contiguous()
    Lin = settle87.block_ldl(Hd); del Hd
    cb = np.ascontiguousarray(trellis2.codebook(CB), dtype=np.float32); vf = settle87.vit_fun(cb, float(K))
    if arm != "one":
        so = float(arm.split(":")[1]); F = torch.from_numpy(np.load(os.environ["HS"])["fisher_qad_gen"]).double()
        F = F[torch.from_numpy(np.load(os.environ["PERM"]))]; mall = float((F * p["r"] ** 2).mean())
    pos = {b: i for i, b in enumerate(p["blocks"])}
    for b in todo:
        t0 = time.time(); i = pos[b]; rows = slice(b * RB, (b + 1) * RB)
        R = p["R_S"][i * RB:(i + 1) * RB].double(); r = p["r"][rows][:, None]; sv = p["sv"][rows][:, None]
        Wt = settle.had_cols(settle.had_rows(R / r * sv) / p["su"])
        if arm == "one": Lout = torch.eye(RB, dtype=torch.float64)
        else:
            d = F[rows] * p["r"][rows] ** 2; hd = settle.had128()
            Ho = (hd * d[None, :]) @ hd / mall; Ho.diagonal().add_(so); Lout = settle87.block_ldl(Ho)
        Q = settle87.ldlq2(Wt * s, Lin, Lout, vf) / s
        q = settle.had_rows(settle.had_cols(Q) * p["su"]) * sv * r
        hq = q @ H
        c = ((hq * R).sum(1) + dl * (q * R).sum(1)) / ((hq * q).sum(1) + dl * (q * q).sum(1)).clamp(min=1e-300)
        D = (q * c[:, None] - R).float().numpy()
        np.save(f"{od}/b{b}.part.npy", D); os.replace(f"{od}/b{b}.part.npy", f"{od}/b{b}.npy")
        e = float(torch.einsum("ij,jk,ik->", torch.from_numpy(D).double(), H, torch.from_numpy(D).double()))
        ref = float(torch.einsum("ij,jk,ik->", R, H, R))
        with open(f"{od}/log.jsonl", "a") as fh:
            fh.write(json.dumps({"block": b, "rel": e / ref, "err": e, "ref": ref, "sec": round(time.time() - t0, 1)}) + "\n")
        print(variant, arm, K, b, f"rel {e / ref:.4e}", f"{time.time() - t0:.0f}s", flush=True)


def real(K, wqrows, out):
    p = load_prep("raw", out); Wq = torch.from_numpy(np.load(wqrows)).float()
    od = f"{out}/d/real/one/K{K}"; os.makedirs(od, exist_ok=True)
    for i, b in enumerate(p["blocks"]):
        np.save(f"{od}/b{b}.npy", (Wq[i * RB:(i + 1) * RB] - p["W_S"][i * RB:(i + 1) * RB]).numpy())


def score(out, set_, dump, sim, tags, sums=False):
    """sums=True (score2, added after the rule's G2 failed; report-only): per row and stratum the sums a = sum_S p dz
    and b = sum_S p (e^dz - 1) instead of KL, for the weighted joint estimate KL = -sum_s w_s a_s + log(1 + sum_s w_s b_s)
    (every sampled row stands for w_s rows; keeps the cross-stratum terms, e.g. a common logit shift, that the
    additive estimate charges)"""
    B = blocks(); od = f"{out}/{'sums' if sums else 'kl'}/{set_}"; os.makedirs(od, exist_ok=True)
    Hh, _, N = open(f"{dump}/336.meta").read().split(); N = int(N)
    X = torch.from_numpy(np.fromfile(f"{dump}/336.x", np.int16).reshape(N, int(Hh))).view(torch.bfloat16).float()
    lse = torch.from_numpy(np.load(sim)["lse"]).double(); assert len(lse) == N
    WS = load_prep("raw", out)["W_S"]                                  # the bf16 head's sampled rows (f32)
    pos = {b: i for i, b in enumerate(B["all"])}
    for tag in tags:
        f = f"{od}/{tag.replace('/', '_').replace(':', '')}.npz"
        if os.path.exists(f): continue
        t0 = time.time(); res = {}
        for st in ("A", "B", "C"):
            idx = rows_of([pos[b] for b in B[st]])                     # positions inside W_S / D
            D = torch.cat([torch.from_numpy(np.load(f"{out}/d/{tag}/b{b}.npy")) for b in B[st]])
            Wst = WS[idx]; kl = torch.zeros(N, dtype=torch.float64); sa = torch.zeros(N, dtype=torch.float64); sb = torch.zeros_like(sa)
            for c in range(0, N, 4096):
                x = X[c:c + 4096]
                z = (x @ Wst.T).double(); dz = (x @ D.T).double()
                p = torch.exp(z - lse[c:c + 4096, None])
                sa[c:c + 4096] = (p * dz).sum(1); sb[c:c + 4096] = (p * torch.expm1(dz)).sum(1)
                kl[c:c + 4096] = -sa[c:c + 4096] + torch.log1p(sb[c:c + 4096])
            res["k" + st] = kl.float().numpy()
            if sums: res["a" + st] = sa.numpy(); res["b" + st] = sb.numpy()
        res["wB"], res["wC"] = np.array(B["wB"]), np.array(B["wC"])
        np.savez(f + ".part.npz", **res); os.replace(f + ".part.npz", f)
        est = res["kA"] + B["wB"] * res["kB"] + B["wC"] * res["kC"]
        print(set_, tag, f"est {est.mean():.6f} (A {res['kA'].mean():.6f} B {res['kB'].mean():.6f} C {res['kC'].mean():.6f})",
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "prep": prep(*a)
    elif c == "scale": scale(*a)
    elif c == "quant": quant(*a)
    elif c == "real": real(*a)
    elif c == "score": score(a[0], a[1], a[2], a[3], a[4:])
    elif c == "score2": score(a[0], a[1], a[2], a[3], a[4:], sums=True)
