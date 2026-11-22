#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""agg218.py RUNS [--json OUT] -- #218 readout over /data/w218/runs:
  judge   every runs/judge/<variant>.json (and runs/scale/scale.json) vs rel: KL ratio, paired per document, 95 % bootstrap CI
          (2,000 resamples of documents, seed 0), documents better / worse
  tune    per round, every runs/rR/tune/<variant>.json vs cur
  grad    per round: train KL, the gradient's SNR from samp.npz (per sampled entry: |G| / its standard error from the spread of
          the per-step gradients, steps weighted by their rows), signal share = sum G^2 / (sum G^2 + sum se^2)
  enc     per round and eta: tiles whose values changed (share), the record decode check, mean proxy, encode seconds"""
import argparse, glob, json, os

import numpy as np


def paired(a, b, reps=2000):
    sa, sb, n = (np.asarray(a["sums"]), np.asarray(b["sums"]), np.asarray(a["rows"], np.float64))
    assert a["docs"] == b["docs"] and a["rows"] == b["rows"], "not the same documents / rows"
    r = sb.sum() / sa.sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(reps):
        i = rng.integers(0, len(n), len(n)); bs.append(sb[i].sum() / sa[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    da, db = sa / n, sb / n
    return {"kl_a": sa.sum() / n.sum(), "kl_b": sb.sum() / n.sum(), "ratio": r, "ci": [lo, hi], "better": int((db < da).sum()),
            "worse": int((db > da).sum()), "docs": len(n), "rows": int(n.sum())}


def fmt(name, p):
    return (f"{name:24s} KL {p['kl_b']:.6f} vs {p['kl_a']:.6f}  ratio {p['ratio']:.4f} [{p['ci'][0]:.4f}, {p['ci'][1]:.4f}]  "
            f"better/worse {p['better']}/{p['worse']} of {p['docs']} ({p['rows']} rows)")


def snr(dirs):
    """over the steps of every chunk dir (the same sampled entries: seed 218 in every chunk)"""
    zs = [np.load(os.path.join(d, "samp.npz")) for d in dirs]
    rows = np.concatenate([z["rows"] for z in zs]).astype(np.float64); N = rows.sum()
    for z in zs[1:]:
        assert all(np.array_equal(z[k], zs[0][k]) for k in zs[0].files if k.endswith(":idx")), "chunks sampled different entries"
    out = {}
    allm, allv = [], []
    for k in zs[0].files:
        if k == "rows" or k.endswith(":idx"): continue
        # z[k][s] = step s's gradient sum / its chunk's rows; as a share of G over all chunks: x_s = z_s N_chunk / N. Steps differ
        # in rows, so the standard error of G = sum_s x_s is the ratio estimator's: g_s = x_s N / rows_s (step s's per-row
        # gradient), se^2 = S / (S - 1) sum_s (rows_s / N)^2 (g_s - G)^2
        x = np.concatenate([z[k].astype(np.float64) * (z["rows"].sum() / N) for z in zs]); S = x.shape[0]; m = x.sum(0)
        gs = x * (N / rows[:, None]); w = (rows / N)[:, None]
        se = np.sqrt(S / (S - 1) * ((w * (gs - m)) ** 2).sum(0))
        s = np.abs(m) / np.maximum(se, 1e-30)
        out[k] = {"median_snr": float(np.median(s)), "frac_snr_gt2": float((s > 2).mean()),
                  "signal_share": float((m ** 2).sum() / max((m ** 2).sum() + (se ** 2).sum(), 1e-30))}
        allm.append(m); allv.append(se)
    m = np.concatenate(allm); se = np.concatenate(allv); s = np.abs(m) / np.maximum(se, 1e-30)
    out["_all"] = {"steps": int(len(rows)), "median_snr": float(np.median(s)), "frac_snr_gt2": float((s > 2).mean()),
                   "signal_share": float((m ** 2).sum() / ((m ** 2).sum() + (se ** 2).sum()))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--json", default=None)
    a = ap.parse_args(); R = a.runs; res = {"judge": {}, "tune": {}, "grad": {}, "enc": {}}
    jd = os.path.join(R, "judge")
    if os.path.exists(os.path.join(jd, "rel.json")):
        rel = json.load(open(os.path.join(jd, "rel.json")))
        print("== judge (qad218-heldout) vs rel")
        files = sorted(glob.glob(os.path.join(jd, "*.json"))) + glob.glob(os.path.join(R, "scale*", "scale.json"))
        for f in files:
            v = json.load(open(f)); name = v["name"] if "scale" not in f else os.path.basename(os.path.dirname(f))
            if name == "rel": continue
            p = paired(rel, v); res["judge"][name] = p; print(fmt(name, p))
    for rd in sorted(glob.glob(os.path.join(R, "r[0-9]"))):
        r = os.path.basename(rd)
        gd = sorted(d for d in glob.glob(os.path.join(rd, "g[0-9]")) if os.path.exists(os.path.join(d, "grad.json")))
        if gd:
            gs = [json.load(open(os.path.join(d, "grad.json"))) for d in gd]
            g = {"rows": sum(x["rows"] for x in gs), "steps": sum(x["steps"] for x in gs), "sec": sum(x["sec"] for x in gs),
                 "train_kl": sum(x["train_kl"] * x["rows"] for x in gs) / sum(x["rows"] for x in gs), "chunks": len(gs)}
            s = snr(gd)
            res["grad"][r] = {**g, "snr": s}
            for d, x in zip(gd, gs):
                res["grad"][r].setdefault("by_chunk", {})[os.path.basename(d)] = snr([d])["_all"]
            print(f"== {r} grad ({len(gd)} chunks): {g['rows']} rows, {g['steps']} steps, train KL {g['train_kl']:.6f}, {g['sec']:.0f} s; SNR all: "
                  + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s["_all"].items()}))
            for k, v in s.items():
                if k != "_all": print(f"   {k.split('layers.')[1]:32s} median SNR {v['median_snr']:.3f}  >2: {v['frac_snr_gt2']:.3f}  signal share {v['signal_share']:.3f}")
        for sf in sorted(glob.glob(os.path.join(rd, "enc", "eta-*", "stats.json"))):
            st = json.load(open(sf)); e = st["eta"]
            px = np.mean([v["proxy"] for v in st["tensors"].values()])
            res["enc"].setdefault(r, {})[str(e)] = {"tiles_values_changed": st["tiles_values_changed"], "tiles": st["tiles"],
                                                     "decode_all_equal": st["decode_all_equal"], "mean_proxy": px, "sec": st["sec_encode"],
                                                     "per_tensor": {k: v["tiles_values_changed"] / v["tiles"] for k, v in st["tensors"].items()}}
            print(f"   {r} eta {e:<6g} tiles changed {st['tiles_values_changed']:>8d} / {st['tiles']} ({100 * st['tiles_values_changed'] / st['tiles']:.2f} %), "
                  f"decode ok {st['decode_all_equal']}, mean proxy {px:.4e}, encode {st['sec_encode']:.0f} s")
        td = os.path.join(rd, "tune")
        if os.path.exists(os.path.join(td, "cur.json")):
            cur = json.load(open(os.path.join(td, "cur.json")))
            for f in sorted(glob.glob(os.path.join(td, "*.json"))):
                v = json.load(open(f))
                if v["name"] == "cur": continue
                p = paired(cur, v); res["tune"].setdefault(r, {})[v["name"]] = p; print("   tune " + fmt(v["name"], p))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
