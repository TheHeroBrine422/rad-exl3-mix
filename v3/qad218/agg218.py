#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""agg218.py RUNS [--json OUT] -- #218 readout over /data/w218/runs:
  judge   every runs/judge/<variant>.json (and runs/scale/scale.json) vs rel: KL ratio, paired per document, 95 % bootstrap CI
          (2,000 resamples of documents, seed 0), documents better / worse
  tune    per round, every runs/rR/tune/<variant>.json vs cur
  grad    per round: train KL, the gradient's SNR from samp.npz (per sampled entry: |mean of the per-step gradients| / (their
          std / sqrt(steps)); the per-step gradient = the step's sum scaled to the round's rows), energy share of the mean
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


def snr(d):
    z = np.load(os.path.join(d, "samp.npz")); rows = z["rows"].astype(np.float64); N = rows.sum()
    out = {}
    allm, allv = [], []
    for k in z.files:
        if k == "rows" or k.endswith(":idx"): continue
        x = z[k] * (N / rows[:, None])                       # per-step gradient of the mean KL (steps x samples)
        S = x.shape[0]; m = x.mean(0); se = x.std(0, ddof=1) / np.sqrt(S)
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
        if os.path.exists(os.path.join(rd, "grad", "grad.json")):
            g = json.load(open(os.path.join(rd, "grad", "grad.json")))
            s = snr(os.path.join(rd, "grad"))
            res["grad"][r] = {"rows": g["rows"], "steps": g["steps"], "train_kl": g["train_kl"], "sec": g["sec"], "snr": s}
            print(f"== {r} grad: {g['rows']} rows, {g['steps']} steps, train KL {g['train_kl']:.6f}, {g['sec']:.0f} s; SNR all: "
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
