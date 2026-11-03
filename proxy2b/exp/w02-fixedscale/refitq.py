#!/usr/bin/env python3
"""refitq.py SRC DST [--rounds N] [--undamped] [--hout DIR] [--alpha FAM=A,..] [--shards] -- fixed-code scale refits
of an ourq.py --save-wq quant (w02, ranked.md B1 and C9). The trellis codes never change: every operation is
W_q' = diag(r) W_q diag(c) on SRC/wq (fsutil.py says why), written to DST/wq for evalq.py, with the multipliers in
DST/scales/<key>.safetensors (r per input row, c per output row) and DST/refit.jsonl per tensor.

  B1  --rounds N         alternate refit_scales' two closed-form steps N more times, starting from the stored quant
      --undamped         fit in the raw input Hessian (refit_scales fits in the damped one: sigma_reg x mean diagonal
                         added, 0.025, input groups 0.005 -- read per tensor from SRC/proxy.jsonl)
      --hout DIR         weight output column n by h_n = diag(H_out)_n (DIR/<key>.safetensors, "hout_diag" or the
                         diagonal of "hout"; normalised to mean 1): the input-side solve becomes
                         ((Q diag(h) Qᵀ) ∘ H) r = rowsum(Q ∘ (H W) ∘ h); the output-side step is per column, so h
                         cancels there. A missing file is an error.
  C9  --alpha FAM=A,..   after the refit: c_n *= (1 - e_n)^(-A/2), e_n = the row's relative error e_nᵀHe_n / w_nᵀHw_n
                         (raw H). A = 1 restores each row's H-norm (undoes the least-squares shrink: a fitted row has
                         |q|² = |w|²(1 - e)), 0 = no change, < 0 = more shrink. FAM: attn_in attn_o gdn_in gdn_out
                         gate_up down, or all.
  --shards               also write SRC's q-L*.safetensors with suh *= r, svh *= c (the container form)
Tensors no operation touches are hard-linked. --rounds 0 without --alpha is a no-op (DST/wq = SRC/wq).
Logged per tensor: one-sided error (raw H) and h-weighted error before / after, and SRC's proxy_err (= the one-sided
error of the stored quant; a check that the Hessian and damping were reconstructed right)."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsutil as U

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--model", default="/data/models/qwen35-08b")
ap.add_argument("--hess", default="/data/h08/calib")
ap.add_argument("--rounds", type=int, default=0)
ap.add_argument("--undamped", action="store_true")
ap.add_argument("--hout", default=None)
ap.add_argument("--hfloor", type=float, default=0.0, help="h += hfloor x mean(h) before normalising")
ap.add_argument("--alpha", default=None)
ap.add_argument("--layers", default=None, help="A-B: only these layers (others linked); tests")
ap.add_argument("--shards", action="store_true")
ap.add_argument("--dry", action="store_true", help="no files, log to stdout only (self-tests)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = torch.device(a.device); U.cap_vram()
ALPHA = {}
if a.alpha:
    for kv in a.alpha.split(","):
        f, v = kv.split("="); assert f == "all" or f in U.FAMS, f
        for g in (U.FAMS if f == "all" else (f,)): ALPHA[g] = float(v)

def refit(W, Q, H, h, rounds):
    """refit_scales' alternation (output scales, then input scales), with output-column weights h in the input solve.
    In place on Q; returns the accumulated (r, c)."""
    k, n = Q.shape
    HW = H @ W
    rt, ct = torch.ones(k, device=Q.device), torch.ones(n, device=Q.device)
    for _ in range(rounds):
        num = (Q * HW).sum(0); dn = (Q * (H @ Q)).sum(0)
        c = torch.where(dn > 1e-30, num / dn.clamp(min=1e-30), torch.ones_like(num))
        Q.mul_(c.view(1, n)); ct *= c
        Qh = Q if h is None else Q * h.view(1, n)
        A = (Qh @ Q.T) * H
        b = (Qh * HW).sum(1)
        A.diagonal().add_(1e-6 * A.diagonal().mean())
        r = torch.linalg.solve(A, b)
        r = torch.where(torch.isfinite(r) & (r > 0), r, torch.ones_like(r))
        Q.mul_(r.view(k, 1)); rt *= r
    return rt, ct

src_proxy, src_sigma = {}, {}
if os.path.exists(f"{a.src}/proxy.jsonl"):
    for l in open(f"{a.src}/proxy.jsonl"):
        d = json.loads(l); src_proxy[d["key"]] = d["proxy_err"]; src_sigma[d["key"]] = d.get("sigma_reg")
keys = U.wq_keys(a.src)
lo, hi = map(int, a.layers.split("-")) if a.layers else (0, 10**6)
if not a.dry: os.makedirs(f"{a.dst}/wq", exist_ok=True)
ck = U.Ckpt(a.model); log = open(f"{a.dst}/refit.jsonl", "w") if not a.dry else sys.stdout
Hc = {}
t0 = time.time()
for key in sorted(keys, key=U.group):
    act = lo <= U.layer(key) <= hi and (a.rounds > 0 or ALPHA.get(U.fam(key), 0.0) != 0.0)
    if not act:
        if not a.dry: U.link_wq(a.src, a.dst, key)
        continue
    gn = U.group(key)
    if gn not in Hc: Hc.clear(); Hc[gn] = U.hessian(f"{a.hess}/{gn}.safetensors", dev)
    H = Hc[gn]
    W = ck.get(key + ".weight").to(dev).float().T.contiguous()
    Q = U.load_wq(a.src, key, dev)
    h = None
    if a.hout:
        from safetensors import safe_open
        with safe_open(f"{a.hout}/{key}.safetensors", "pt") as f:         # missing file: error (ranked A8)
            h = (f.get_tensor("hout_diag") if "hout_diag" in f.keys() else f.get_tensor("hout").diagonal()).float().to(dev)
        assert h.shape[0] == Q.shape[1] and torch.isfinite(h).all() and (h >= 0).all(), key
        h = h + a.hfloor * h.mean(); h = h / h.mean()
    hw = h if h is not None else torch.ones(Q.shape[1], device=dev)
    e0, r0 = U.colerr(W, Q, H)
    rec = {"key": key, "fam": U.fam(key), "src_proxy": src_proxy.get(key), "e_before": float(e0.sum() / r0.sum()),
           "eh_before": float((hw.double() * e0).sum() / (hw.double() * r0).sum())}
    r, c = torch.ones(Q.shape[0], device=dev), torch.ones(Q.shape[1], device=dev)
    if a.rounds:
        sig = src_sigma.get(key) or (0.005 if gn.endswith(f"layers.{U.layer(key)}.input") else 0.025)   # q2b.sh's
        Hf = H if a.undamped else H + sig * H.diagonal().mean() * torch.eye(H.shape[0], device=dev)
        r, c = refit(W, Q, Hf, h, a.rounds); rec["sigma_fit"] = 0.0 if a.undamped else sig
    al = ALPHA.get(U.fam(key), 0.0)
    if al:
        e, rf = U.colerr(W, Q, H)
        m = (1 - (e / rf.clamp(min=1e-30)).clamp(max=0.5)).pow(-al / 2).float()
        Q.mul_(m.view(1, -1)); c *= m; rec.update(alpha=al, alpha_mult_mean=float(m.mean()), alpha_mult_max=float(m.max()))
    e1, r1 = U.colerr(W, Q, H)
    rec.update(e_after=float(e1.sum() / r1.sum()), eh_after=float((hw.double() * e1).sum() / (hw.double() * r1).sum()),
               r_rms=float((r.log().pow(2).mean()).sqrt()), c_rms=float((c.log().pow(2).mean()).sqrt()))
    log.write(json.dumps(rec) + "\n"); log.flush()
    if not a.dry: U.write_wq(a.dst, key, Q); U.write_scales(a.dst, key, r, c)
    del W, Q
args = {k: v for k, v in vars(a).items() if k not in ("dst", "dry", "device")}
if a.shards and not a.dry: U.edit_shards(a.src, a.dst, args)
if not a.dry:
    json.dump(args, open(f"{a.dst}/refitq.json", "w"), indent=1)
    assert len(U.wq_keys(a.dst)) == len(keys), "DST/wq incomplete"
    open(f"{a.dst}/.done", "w").close()
print(f"refitq done: {len(keys)} tensors, {time.time() - t0:.0f}s", file=sys.stderr)
