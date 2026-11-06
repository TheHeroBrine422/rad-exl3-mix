#!/usr/bin/env python3
"""blockx.py DUMPDIR MAP IDX[,IDX...] OUT.jsonl -- review idea 7: A4 with the k worst 128-input blocks kept at A8 (CPU).

Same operands and quantizer as v3/e1b/aqproxy.py (imported: rotation, quantizer, weight rows <= 4096), on the E1c
dumps (512 rows per linear = 8 prefill steps x 64 rows, calib-A, K6 model). For A4 at group g (64, 128; clip 1) a
per-tensor bitmask marks k in {0, 1, 2, 4} rotated 128-blocks whose activations are quantized at A8 (same group)
instead. The mask is chosen on one half of the rows (steps 0-3 or 4-7) and scored on the other (2-fold; sums of
both folds are reported, so k = 0 equals aqproxy's all-row rel_out).
Selectors (all from the selection half only):
  amax  -- mean over rows of the rotated block's absmax^2 (what sets the kernel's g64/g128 scale; the idea's rule)
  inmax -- the same in input space (x * suh, before the rotation)
  err   -- each block's own cut of the output error energy if it alone went to A8 (cross terms included)
Linearity: e_out = U(d4) W^T - sum_{b in mask} U_b(d4_b - d8_b) W_b^T, U the block-local inverse rotation, d = xr - q.
One JSON line per (tensor, group, selector, k) plus the A8 references (all blocks A8 at g, and A8-g128).
"""
import json, os, sys
import numpy as np

AQ = os.environ.get("AQ_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../e1b"))
sys.path.insert(0, AQ)
import aqproxy as aq   # noqa: E402
import mkmask          # noqa: E402

KS = (1, 2, 4)
GROUPS = (64, 128)
SELS = ("amax", "inmax", "err")


def unrot(d, s):   # block-local inverse rotation (aqproxy.unrotate), any column range of whole blocks
    return aq.unrotate(d, s)


def one(D, rows, lab, i):
    fam, layer = lab[i]; _, n, k, mats = rows[i]
    K, bf, m2 = map(int, open(f"{D}/{i}.meta").read().split())
    raw = np.fromfile(f"{D}/{i}.x", dtype=np.uint16).reshape(-1, K)
    x = ((raw.astype(np.uint32) << 16).view(np.float32) if bf else raw.view(np.float16).astype(np.float32))
    suhs = [np.fromfile(f"{D}/{i}.suh", dtype=np.float16).astype(np.float32)]
    if m2 == 2: suhs.append(np.fromfile(f"{D}/{i}.suh1", dtype=np.float16).astype(np.float32))
    p = f"{aq.P}{layer}."
    cands = ([p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight"] if fam == "gdn_in" else
             [p + f"self_attn.{t}_proj.weight" for t in ("q", "k", "v")] if fam == "attn_in" else [])
    Ws = aq.weight(aq.hf_names(fam, layer, n, k, aq.shapes_for(cands)))
    assert len(Ws) == len(suhs), (i, fam, len(Ws), len(suhs))
    R, B = x.shape[0], K // 128
    folds = [(np.arange(0, R // 2), np.arange(R // 2, R)), (np.arange(R // 2, R), np.arange(0, R // 2))]
    ref = sum(float(((x @ W.T).astype(np.float64) ** 2).sum()) for W in Ws)
    acc = {}            # (g, sel, k) -> output error energy summed over halves and folds
    chosen = {}         # (g, sel, k) -> blocks per (half, fold)
    a8 = {}
    share1 = {}         # g -> share of A4 output error removed by the single best block (err selector, in-sample)
    for h, (s, W) in enumerate(zip(suhs, Ws)):
        xr = aq.rotate(x, s)
        xin = x * s
        amax_r = (np.abs(xr).reshape(R, B, 128).max(2) ** 2)           # [R, B]
        amax_i = (np.abs(xin).reshape(R, B, 128).max(2) ** 2)
        Wt = W.T.copy()                                                  # [K, N]
        q8g128 = aq.quant(xr, 8, 128, 1.0)
        e = unrot(xr - q8g128, s)
        a8["a8g128"] = a8.get("a8g128", 0.0) + float(((e @ Wt).astype(np.float64) ** 2).sum())
        for g in GROUPS:
            d4 = xr - aq.quant(xr, 4, g, 1.0)
            d8 = xr - aq.quant(xr, 8, g, 1.0)
            e8 = unrot(d8, s)
            a8[f"a8g{g}"] = a8.get(f"a8g{g}", 0.0) + float(((e8 @ Wt).astype(np.float64) ** 2).sum())
            A = unrot(d4, s) @ Wt                                        # [R, N] A4 output error
            dd = unrot(d4 - d8, s)                                       # [R, K] per-block fix (block-local)

            def Db(b, r):  # block b's output-error fix on rows r
                return dd[r, b * 128:(b + 1) * 128] @ Wt[b * 128:(b + 1) * 128]

            for fi, (S, E) in enumerate(folds):
                AE = A[E].astype(np.float64)
                base = float((AE ** 2).sum())
                acc[(g, "-", 0)] = acc.get((g, "-", 0), 0.0) + base
                AS = A[S]
                s2 = float((AS.astype(np.float64) ** 2).sum())
                errsc = np.array([s2 - float(((AS - Db(b, S)).astype(np.float64) ** 2).sum()) for b in range(B)])
                if fi == 0:   # in-sample (all rows) cut of fold 0's best block
                    tot = float((A.astype(np.float64) ** 2).sum())
                    b1 = int(np.argmax(errsc))
                    share1[g] = share1.get(g, 0.0) + tot - float(((A - Db(b1, slice(None))).astype(np.float64) ** 2).sum())
                score = {"amax": amax_r[S].mean(0), "inmax": amax_i[S].mean(0), "err": errsc}
                for sel in SELS:
                    order = np.argsort(-score[sel], kind="stable")
                    for kk in KS:
                        blk = sorted(int(b) for b in order[:kk])
                        eo = AE.copy()
                        for b in blk: eo -= Db(b, E)
                        acc[(g, sel, kk)] = acc.get((g, sel, kk), 0.0) + float((eo ** 2).sum())
                        chosen.setdefault((g, sel, kk), []).append(blk)
    out = []
    head = {"i": i, "fam": fam, "layer": layer, "N": n, "K": K, "B": B, "rows": R, "halves": len(suhs)}
    for g in GROUPS:
        out.append({**head, "g": g, "sel": "-", "k": 0, "rel_out": acc[(g, "-", 0)] / ref,
                    "a8_rel_out": a8[f"a8g{g}"] / ref, "a8g128_rel_out": a8["a8g128"] / ref,
                    "best1_share_insample": share1[g] / acc[(g, "-", 0)]})
        for sel in SELS:
            for kk in KS:
                out.append({**head, "g": g, "sel": sel, "k": kk, "rel_out": acc[(g, sel, kk)] / ref,
                            "blocks": chosen[(g, sel, kk)]})
    return out


def main():
    D, mp, idxs, outp = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(",")], sys.argv[4]
    rows = sorted(tuple(map(int, l.split())) for l in open(mp) if l.strip())
    lab = mkmask.label(rows)
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["i"] for l in open(outp) if l.strip()}
    for i in idxs:
        if i in done: continue
        lines = one(D, rows, lab, i)
        with open(outp, "a") as f: f.write("".join(json.dumps(o) + "\n" for o in lines))
        print(i, lab[i], "done", flush=True)


if __name__ == "__main__":
    main()
