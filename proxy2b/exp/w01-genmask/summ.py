#!/usr/bin/env python3
"""summ.py GOLDEN DIR CTLDIR -- w01-genmask (ranked.md A1): per K, the gen-masked mean KL of the masked-loss arms
(DIR/k{K}gm{,_s1,_s2}.f32), the unmasked-loss control (CTLDIR/k{K}eB{,_s1,_s2}.f32, yaqaFact7's 1K Sketch-B diagonal
on the same windows, seed and labels) and base (CTLDIR/k{K}base{,_s1,_s2}.f32); 3-seed means, per-seed SD, the
masked / unmasked ratio, seed-matched comparisons, and the draft pass rule of README.md. Missing control files are
looked up in DIR (run.sh quantizes them there when yaqaFact7 has not)."""
import json, os, sys
import numpy as np
gd, D, C = sys.argv[1:4]
gen = np.fromfile(f"{gd}/gen.u8", np.uint8).astype(bool)
def kl(tag):
    for d in (D, C):
        if os.path.exists(f"{d}/{tag}.f32"):
            a = np.fromfile(f"{d}/{tag}.f32", np.float32).reshape(-1, 4); assert len(a) == len(gen); return float(a[gen, 0].mean())
    return None
SFX = ["", "_s1", "_s2"]; out = {}
for K in (3, 2):
    arm = {n: [kl(f"k{K}{t}{s}") for s in SFX] for n, t in (("gm", "gm"), ("eB", "eB"), ("base", "base"))}
    if any(v is None for v in arm["gm"] + arm["eB"]): print(f"K{K}: incomplete {arm}"); continue
    m = {n: float(np.mean(v)) for n, v in arm.items() if None not in v}
    sd = {n: float(np.std(v, ddof=1) / np.mean(v)) for n, v in arm.items() if None not in v}
    r = m["gm"] / m["eB"]; seeds = [g < e for g, e in zip(arm["gm"], arm["eB"])]
    out[K] = dict(kl=arm, mean=m, seed_sd_rel=sd, ratio=r, seeds_masked_better=seeds,
                  gain_vs_base={n: (m[n] / m["base"] - 1) if "base" in m else None for n in ("gm", "eB")})
    print(f"K{K}: masked {m['gm']:.5f} (seeds {', '.join(f'{x:.5f}' for x in arm['gm'])}, SD {sd['gm']:.2%}) | unmasked "
          f"{m['eB']:.5f} (SD {sd['eB']:.2%}) | ratio {r:.4f} | masked better on seeds {seeds}"
          + (f" | vs base: masked {m['gm'] / m['base'] - 1:+.2%}, unmasked {m['eB'] / m['base'] - 1:+.2%}" if "base" in m else ""))
if 3 in out and 2 in out:
    r3, r2, s3 = out[3]["ratio"], out[2]["ratio"], all(out[3]["seeds_masked_better"])
    if r3 <= 0.975 and s3 and r2 <= 1.00: v = "PASS: masked loss for the 27B diagonal collection (YAQA_GENMASK=1)"
    elif r3 >= 1.00 or r2 > 1.02: v = "FAIL: keep the all-position loss"
    else: v = "INCONCLUSIVE: keep the all-position loss (status quo)"
    out["verdict"] = v; print(v)
for t in ("cmask/hdiag-snap1024", "../c18/hdiag-snap1024"):
    f = f"{D}/{t}/halves.json"
    if os.path.exists(f): h = json.load(open(f)); print(f"halves {t}: median {h['median']:.4f} p5 {h['p5']:.4f} min {h['min']:.4f}")
json.dump(out, open(f"{D}/summary.json", "w"), indent=1)
