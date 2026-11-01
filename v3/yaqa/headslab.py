#!/usr/bin/env python3
"""headslab.py -- row-slab LM head: per-vocab-row bit allocation driven by the head's output-side Fisher diagonal.

  headslab.py perm HEADSENS.npz PERM.npy
      vocab rows sorted by training-pool sensitivity (qad pool, scored positions; env HS_TRAIN), most sensitive first.
      qad only, so the fit-pool slices that judge plans (calib-A head dump, golden-ml) stay out of sample.
      The head is then quantized in that order (ourq.py --head-perm), so a slab is a run of whole 128-row blocks
      (EXL3's output rotation mixes 128 rows; a slab cannot cut one).
  headslab.py plan HEADSENS.npz PERM.npy QDIR PLAN.json
      QDIR/K<k>/wq/lm_head.rowerr.npz: per-row error (w-q)ᵀH(w-q) at each K (permuted order). Row cost =
      Fisher(v) * err(v) (KLD ≈ ½ Σ_v p(1-p) Δlogit_v², cross terms dropped). 
      For budgets "uniform K4 / K5 / K6 bytes": uniform, S-slab (contiguous in sensitivity order, DP) and per-block
      (Lagrangian, the ceiling) assignments, a
      floor of max(HS_FLOOR = 3, round(budget K - 1.5)) (a K2 tail leaks mass into rare tokens: vtsim), costs under the training Fisher and the held-out verify Fisher.
      Curve mode (the solver's slab-head menu): HS_BUDGETS=3.5,3.625,.. -> for every budget a 3-slab plan "s<B>-3"
      plus the uniform plan "u<K>" of every donor K; HS_KS=3,3.5,.. restricts the donors (headsim keeps every
      donor on the card: ~2.5 GB each, so a long K range runs as two passes). Donor dirs may be fractional (K3.5).
  headslab.py build PLAN.json NAME PERM.npy QDIR DONOR.rad
      writes plan NAME's head (rows from each K's dequantized quant, un-permuted, bf16) over output.weight of DONOR
      (a reflink copy of the bf16 container), in place."""
import json, os, struct, sys, numpy as np

BLK, HID = 128, 5120
TRAIN, HELD = tuple(os.environ.get("HS_TRAIN", "qad").split(",")), ("verify",)
FLOOR = int(os.environ.get("HS_FLOOR", 3))

def fisher(npz, pools):
    z = np.load(npz); return sum(z[f"fisher_{p}_gen"] for p in pools)

def cmd_perm(hs, out):
    z = np.load(hs); F = fisher(hs, TRAIN); P = sum(z[f"psum_{p}_gen"] for p in TRAIN)
    perm = np.lexsort((-P, -F)).astype(np.int64)      # by Fisher, ties (zero-mass rows) by probability mass
    assert len(perm) % BLK == 0
    np.save(out, perm); print(f"{len(perm)} rows, {len(perm) // BLK} blocks; top-1 block holds "
                              f"{F[perm[:BLK]].sum() / F.sum():.1%} of the training Fisher")

def block_costs(hs, perm, qdir):
    Ks = sorted(float(d[1:]) for d in os.listdir(qdir) if d[0] == "K" and os.path.exists(f"{qdir}/{d}/wq/lm_head.rowerr.npz"))
    if os.environ.get("HS_KS"): Ks = [k for k in Ks if k in {float(x) for x in os.environ["HS_KS"].split(",")}]
    E = np.stack([np.load(f"{qdir}/K{k:g}/wq/lm_head.rowerr.npz")["err"] for k in Ks])      # [K, rows], permuted
    nb = E.shape[1] // BLK
    C = {}
    for name, pools in (("train", TRAIN), ("held", HELD)):
        F = fisher(hs, pools)[perm]
        C[name] = (E * F).reshape(len(Ks), nb, BLK).sum(2)                                   # [K, blocks]
    return Ks, C, nb

def lagrange(cost, Ks, kmin, budget_bits, nb):
    ok = np.array(Ks) >= kmin; Kv = np.array(Ks)
    lo, hi = 0.0, 1.0
    while True:                                       # grow hi until the budget holds
        a = np.where(ok[:, None], cost + hi * Kv[:, None], np.inf).argmin(0)
        if Kv[a].sum() <= budget_bits: break
        hi *= 4
    for _ in range(80):
        lam = (lo + hi) / 2
        a = np.where(ok[:, None], cost + lam * Kv[:, None], np.inf).argmin(0)
        lo, hi = (lam, hi) if Kv[a].sum() > budget_bits else (lo, lam)
    return np.where(ok[:, None], cost + hi * Kv[:, None], np.inf).argmin(0)

def slabs(cost, Ks, kmin, budget_bits, nb, S, step=4):
    """contiguous segments (cuts on multiples of `step` blocks), at most S, one K each; Lagrangian over the DP"""
    cuts = np.unique(np.r_[np.arange(0, nb, step), nb]); m = len(cuts)
    pc = np.concatenate([np.zeros((len(Ks), 1)), np.cumsum(cost, 1)], 1)
    Kv = np.array(Ks); ok = Kv >= kmin
    def solve(lam):
        INF = np.inf; dp = np.full((S + 1, m), INF); dp[0, 0] = 0; arg = {}
        for s in range(1, S + 1):
            for i in range(1, m):
                j = np.arange(i)
                seg = (pc[:, cuts[i]][:, None] - pc[:, cuts[j]]) + lam * Kv[:, None] * (cuts[i] - cuts[j])[None, :]
                seg[~ok] = INF
                tot = dp[s - 1, j][None, :] + seg
                k, jj = np.unravel_index(np.argmin(tot), tot.shape)
                dp[s, i] = tot[k, jj]; arg[(s, i)] = (jj, k)
        s = int(np.argmin(dp[:, m - 1])); i = m - 1; a = np.zeros(nb, int)
        while i > 0:
            j, k = arg[(s, i)]; a[cuts[j]:cuts[i]] = k; i, s = j, s - 1
        return a
    lo, hi = 0.0, 1.0
    while (Kv[solve(hi)]).sum() > budget_bits: hi *= 4
    for _ in range(40):
        lam = (lo + hi) / 2
        lo, hi = (lam, hi) if Kv[solve(lam)].sum() > budget_bits else (lo, lam)
    return solve(hi)

def cmd_plan(hs, permf, qdir, out):
    perm = np.load(permf); Ks, C, nb = block_costs(hs, perm, qdir); Kv = np.array(Ks)
    plans = {}
    def report(name, a, ref):
        r = {"K_blocks": [float(Kv[i]) for i in a], "avg_bits": float(Kv[a].mean())}
        for c in ("train", "held"):
            r[c] = float(C[c][a, np.arange(nb)].sum()); r[c + "_vs_uniform"] = r[c] / ref[c] - 1
        segs = []; i = 0
        while i < nb:
            j = i
            while j < nb and a[j] == a[i]: j += 1
            segs.append(f"{(j - i) * BLK}@K{Kv[a[i]]:g}"); i = j
        r["segments"] = segs if len(segs) <= 12 else f"{len(segs)} segments"
        plans[name] = r
        print(f"{name:10s} avg {r['avg_bits']:.3f} b  train {r['train_vs_uniform']:+.1%}  held-out {r['held_vs_uniform']:+.1%}  "
              f"{r['segments'] if isinstance(r['segments'], str) else ' '.join(r['segments'])}", flush=True)
    if os.environ.get("HS_BUDGETS"):                  # curve mode
        for k in Ks:
            u = np.full(nb, Ks.index(k)); report(f"u{k:g}", u, {c: C[c][u, np.arange(nb)].sum() for c in C})
        for B in (float(x) for x in os.environ["HS_BUDGETS"].split(",")):
            lo = max(k for k in Ks if k <= B) if any(k <= B for k in Ks) else Ks[0]
            u = np.full(nb, Ks.index(lo)); ref = {c: C[c][u, np.arange(nb)].sum() for c in C}
            kmin = max(FLOOR, round(B - 1.5 + 1e-9))
            if kmin not in Ks: kmin = min(k for k in Ks if k >= kmin)
            report(f"s{B:g}-3", slabs(C["train"], Ks, kmin, B * nb, nb, 3), ref)
        json.dump({"Ks": Ks, "plans": plans}, open(out, "w"), indent=1); return
    for B in (4, 5, 6):                               # where V2 mixes put the head (K4 at 2.5-3 bpw ... K6 at 4.5-5)
        if B not in Ks: continue
        u = np.full(nb, Ks.index(B))
        ref = {c: C[c][u, np.arange(nb)].sum() for c in C}
        report(f"u{B}", u, ref)
        kmin = max(FLOOR, round(B - 1.5 + 1e-9))
        for S in (3, 4):
            report(f"s{B}-{S}", slabs(C["train"], Ks, kmin, B * nb, nb, S), ref)
        report(f"b{B}", lagrange(C["train"], Ks, kmin, B * nb, nb), ref)
        report(f"b{B}-nofloor", lagrange(C["train"], Ks, min(Ks), B * nb, nb), ref)
    json.dump({"Ks": Ks, "plans": plans}, open(out, "w"), indent=1)

def entry(path, want="output.weight"):
    f = open(path, "rb"); h = f.read(4096)
    str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt = struct.unpack_from("<6Q", h, 16 + 5 * 8)
    f.seek(str_off); S = f.read(str_bytes)
    es = 3 * 8 + 4 + 4 + 8 * 6 + 4 * 3 + 4 + 8 + 8 + 8 + 16
    f.seek(dir_off); D = f.read(es * dir_cnt)
    for i in range(dir_cnt):
        e = D[i * es:(i + 1) * es]
        name = struct.unpack_from("<Q", e, 0)[0]
        if S[name:S.index(b"\0", name)].decode() != want: continue
        rank = struct.unpack_from("<I", e, 28)[0]; shape = struct.unpack_from("<6q", e, 32)[:rank]
        off, nbytes = struct.unpack_from("<QQ", e, es - 32)
        return off, nbytes, shape
    raise KeyError(want)

def cmd_build(planf, name, permf, qdir, donor):
    from safetensors import safe_open
    P = json.load(open(planf)); a = np.array(P["plans"][name]["K_blocks"]); perm = np.load(permf)
    off, nbytes, shape = entry(donor)
    V = len(perm); assert tuple(shape) == (V, HID) and nbytes == V * HID * 2, (shape, nbytes)
    rowK = np.repeat(a, BLK)                          # K of each permuted row
    out = np.memmap(donor, np.uint16, "r+", offset=off, shape=(V, HID))
    for k in sorted(set(rowK.tolist())):
        idx = np.nonzero(rowK == k)[0]
        with safe_open(f"{qdir}/K{k:g}/wq/lm_head.safetensors", "pt", device="cpu") as f:
            W = f.get_tensor("weight")                # (V, HID) f16, permuted rows
        import torch
        for c in range(0, len(idx), 16384):
            ii = idx[c:c + 16384]
            out[perm[ii]] = W[torch.from_numpy(ii)].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        del W
        print(f"  K{k:g}: {len(idx)} rows", flush=True)
    out.flush(); del out
    print(f"{name} written into {donor} (output.weight at {off}, {nbytes} bytes)")

if __name__ == "__main__":
    c, args = sys.argv[1], sys.argv[2:]
    {"perm": cmd_perm, "plan": cmd_plan, "build": cmd_build}[c](*args)
