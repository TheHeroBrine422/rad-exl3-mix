#!/usr/bin/env python3
"""heph_check.py LAYER EXPERT KIND OUT.jsonl -- HEPH re-encode of one DS4.1-F routed-expert matrix (w235), one jsonl line per spec.

Source = the checkpoint's MXFP4 codes: targets z = the E2M1 atoms read straight from the nibbles (no dequant round trip), scale =
the stored e8m0 byte of each 1x32 group (exact power of two). Sequence = 64 consecutive weights of one row along the input dim (two
32-groups), tail-biting two-pass (heph.c). Branch weight (HEPH_SPECS variant): std = diag of the expert's own H (paper eq. 4, z domain),
sw = diag(H) * s_g^2 (the output-error proxy the paper rejects), h1 = 1 (no calibration). Metrics in the ORIGINAL (unrotated) domain:
rel = tr(dW H dW^T) / tr(W H W^T) with the expert's FULL H (w211's h13 for gate_up, h2 for down), rel_diag (H -> diag H), rel_mse (H -> I).
Rows: the w211 seeded permutation slice (W211_SLICE, W235_ROWS_GU / W235_ROWS_DN rows; 0 = all rows), so the rows equal w211's.
Specs env HEPH_SPECS = comma list of variant:K:L[:seed] (default std:1:12,std:2:12,std:3:12). Lines already in OUT are skipped.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("W235_W211DIR", "/data/q27/w235/code/w211"))
import numpy as np
import torch
import dscommon as D
import heph
torch.set_grad_enabled(False); torch.set_num_threads(1)
T = 64


def load_codes(layer, e, kind):
    """exact atoms z [N, Kd] float32 (sign of zero kept) and e8m0 bytes E [N, Kd/32] uint8 of the stacked matrix."""
    zs, Es = [], []
    for n in (["w1", "w3"] if kind == "gate_up" else ["w2"]):
        k = f"layers.{layer}.ffn.experts.{e}.{n}"
        u = D._get(k + ".weight").view(torch.uint8).numpy()
        zs.append(np.stack([heph.FP4_TABLE[u & 0x0F], heph.FP4_TABLE[u >> 4]], axis=-1).reshape(u.shape[0], -1))
        Es.append(D._get(k + ".scale").view(torch.uint8).numpy())
    return np.concatenate(zs, 0), np.concatenate(Es, 0)


def main():
    layer, e, kind, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    sl = int(os.environ.get("W211_SLICE", "0"))
    nrows = int(os.environ.get("W235_ROWS_GU" if kind == "gate_up" else "W235_ROWS_DN", "0"))
    specs = [s.split(":") for s in os.environ.get("HEPH_SPECS", "std:1:12,std:2:12,std:3:12").split(",")]
    specs = [(v, float(k), int(L), int(sd[0]) if sd else 1) for v, k, L, *sd in specs]
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l); done.add((r["variant"], r["K"], r["L"], r["seed"], r["rows"], r["slice"]))
    todo = [s for s in specs if (s[0], s[1], s[2], s[3], nrows, sl) not in done]
    if not todo: return
    t0 = time.time()
    z_all, E_all = load_codes(layer, e, kind); N, Kd = z_all.shape
    ec = heph.exponent_cost(E_all)
    H0, stats = D.load_expert_h(layer, e, kind)
    hdiag = H0.diagonal().clone()
    if nrows:
        g = torch.Generator().manual_seed(7 + layer + 100003 * e)
        idx = torch.randperm(N, generator=g)[sl * nrows:(sl + 1) * nrows].sort()[0]; assert len(idx) == nrows
        idx = idx.numpy()
    else: idx = np.arange(N)
    z = z_all[idx]; E = E_all[idx]; R = len(idx)
    sg = np.ldexp(np.float32(1.0), E.astype(np.int32) - 127).astype(np.float32)           # [R, Kd/32]
    sg_el = np.repeat(sg, 32, axis=1).astype(np.float64)
    W = torch.from_numpy(z.astype(np.float64) * sg_el)                                    # exact source weights, float64
    hn = (hdiag / hdiag.mean()).numpy().astype(np.float32)
    assert torch.equal(W, D.load_expert_w(layer, e, kind)[idx]), "codes x e8m0 != w211's dequantizer"
    den_full = float(torch.einsum("ij,jk,ik->", W, H0, W)); den_diag = float((W ** 2 @ hdiag).sum()); den_mse = float((W ** 2).sum())
    t_setup = time.time() - t0
    for variant, K, L, seed in todo:
        t1 = time.time()
        if variant == "h1": wt = np.ones((R, Kd), dtype=np.float32)
        else:
            wt = np.broadcast_to(hn, (R, Kd)).astype(np.float32)
            if variant == "sw": wt = (wt * (sg_el ** 2)).astype(np.float32); wt /= wt.mean()
        cw, _, _, bad, cb, kk = heph.encode(z.reshape(R * Kd // T, T), wt.reshape(R * Kd // T, T), K, L, T, seed)
        cw = cw.reshape(R, Kd)
        t_enc = time.time() - t1
        What = torch.from_numpy(cw.astype(np.float64) * sg_el)
        Dm = W - What
        rel = float(torch.einsum("ij,jk,ik->", Dm, H0, Dm)) / den_full
        rel_diag = float((Dm ** 2 @ hdiag).sum()) / den_diag; rel_mse = float((Dm ** 2).sum()) / den_mse
        ong = bool(np.isin(np.abs(cw), heph.MAGS).all())
        rec = {"layer": layer, "e": e, "kind": kind, "variant": variant, "K": K, "L": L, "seed": seed, "T": T, "rows": nrows, "slice": sl,
               "R": R, "Kd": Kd, "rel": rel, "rel_diag": rel_diag, "rel_mse": rel_mse,
               "frac_exact": float((cw == z).mean()), "on_grid": ong, "bad_seq": int(bad), "enc_sec": round(t_enc, 2),
               "sec_per_row": t_enc / R, "setup_sec": round(t_setup, 1), "exp": ec, "n_routed": stats["n_routed"], "kept_rows": stats["n"]}
        print(json.dumps(rec), flush=True)
        with open(out, "a") as fh: fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__": main()
