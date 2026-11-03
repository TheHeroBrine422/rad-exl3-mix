#!/usr/bin/env python3
"""selftest.py [GOLDEN SCOREDIR] -- CPU checks of race.py / aggsim.py.
1. synthetic slice with known structure (category offsets, one noisy category): race.py runs end to end; its stage-1
   P(win) is calibrated; stratified se < plain; Neyman < 1; wrong rejections at alpha 0.01 <= 2%.
2. real per-row files (if given, e.g. the 0.8B arms): race.py's delta-method se of log(B/A) vs a document bootstrap.
3. aggsim's prefix-deviation simulation has variance se^2 (1-f)/f."""
import json, math, os, subprocess, sys, tempfile
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ok = True
def check(name, cond, msg):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name}: {msg}", flush=True)

# ---- 1. synthetic
rng = np.random.default_rng(1)
with tempfile.TemporaryDirectory() as T:
    g = f"{T}/gold"; s = f"{T}/sc"; os.makedirs(g); os.makedirs(s)
    D = 160; cats = ["code", "web", "chat", "multilingual"]; src = {"code": "stackv2-edu", "web": "fineweb-edu",
                                                                     "chat": "wildchat", "multilingual": "fineweb2-hq"}
    c = rng.integers(0, 4, D); L = rng.integers(200, 2000, D); docs, r0 = [], 0
    for i in range(D):
        docs.append({"source": src[cats[c[i]]] + "/x", "tokens": int(L[i]), "score_from": 0, "row0": int(r0)}); r0 += int(L[i]) - 1
    R = r0; json.dump({"rows": int(R), "docs": docs}, open(f"{g}/kld.json", "w"))
    np.ones(R, np.uint8).tofile(f"{g}/gen.u8")
    doc = np.repeat(np.arange(D), L - 1)
    lvl = rng.gamma(2, 0.5, D) * np.array([1.0, 0.7, 0.8, 1.6])[c]          # per-document KL level of the base
    noise_sd = np.array([0.03, 0.03, 0.03, 0.12])[c]                        # multilingual reacts noisily
    def write(tag, ratio, catk):                                             # catk: sd of per-category shifts
        off = catk * rng.standard_normal(4)                                  # this candidate hurts some categories more
        per = ratio * (1 + off[c]) * (1 + noise_sd * rng.standard_normal(D))
        kl = lvl[doc] * per[doc] * rng.gamma(4, 0.25, R)                     # row noise
        np.stack([kl, np.ones(R), np.ones(R), np.ones(R)], 1).astype(np.float32).tofile(f"{s}/{tag}.f32")
    write("k3a", 1.0, 0.0)
    for j, r in enumerate(np.exp(rng.normal(0.0, 0.03, 60))): write(f"k3b{j:02d}", r, 0.04)
    out = f"{T}/res.jsonl"
    p = subprocess.run([sys.executable, f"{H}/race.py", g, s, "--orders", "8", "--out", out, "--label", "synthetic"],
                       capture_output=True, text=True)
    print(p.stdout[-2500:], p.stderr[-1500:])
    res = json.loads(open(out).readline())
    cal = res["g3"]["ratio"]["calibration_stage1"]
    dev = [abs(rate - (lo + min(hi, 1) ) / 2) for lo, hi, n, rate in cal if n >= 100 and rate is not None and 0.05 <= lo < 0.8]
    check("calibration", dev and max(dev) < 0.12, f"|observed - bin centre| in the middle bins: {np.round(dev, 3).tolist()}")
    w = res["g3"]["ratio"]["by_alpha"]["0.01"]["wrong_per_accept"]
    check("wrong rejections", w <= 0.02, f"alpha 0.01: {100 * w:.2f}% of accepts")
    print(f"  (additive strata: strat / plain {res['g6']['se_ratio_strat']:.3f}; category effects here are multiplicative)")
    check("stratified ratio se", res["g6"]["se_ratio_strat_ratio"] < 0.97, f"strat-ratio / plain {res['g6']['se_ratio_strat_ratio']:.3f} (category shifts built in)")
    wc = res["g3"]["cat"]["by_alpha"]["0.01"]["wrong_per_accept"]
    print(f"  (variant cat, not in the pass rule: alpha 0.01 {100 * wc:.2f}% wrong rejections -- overconfident at 10% of the docs)")
    # the V2 stream path: results order, best-so-far incumbents, frontier predictions
    import shutil
    rs = f"{T}/rs"; os.makedirs(rs); res2 = f"{T}/res2.jsonl"; truth = {}
    with open(f"{T}/results.jsonl", "w") as fr, open(f"{T}/frontier-s1.txt", "w") as ff:
        for j in range(60):
            tag = f"{['a', 'a', 'f2'][j % 3]}-knap-s{j}-ts0-sc4.00-e4-3.00"
            src_ = f"{s}/k3b{j:02d}.f32"; shutil.copy(src_, f"{rs}/{tag}.f32")
            kl = float(np.fromfile(src_, np.float32).reshape(-1, 4)[:, 0].astype(float).mean())
            fr.write(json.dumps({"tag": tag, "golden": "gold", "kld": kl}) + "\n"); truth[tag] = kl
            ff.write(f"knap-s{j}-ts0-sc4.00-e4-3.00 bpw 3.000  pred KLD {kl * math.exp(rng.normal(0, 0.02)):.6f} +-5%\n")
    p = subprocess.run([sys.executable, f"{H}/race.py", g, rs, "--results", f"{T}/results.jsonl", "--pred", f"{T}/frontier-*.txt",
                        "--orders", "4", "--out", res2, "--label", "stream-synthetic"], capture_output=True, text=True)
    print(p.stdout[-1200:], p.stderr[-800:])
    r2 = json.loads(open(res2).readline()) if os.path.exists(res2) else {}
    check("stream path", r2.get("pairs") == 59 and "prior" in r2.get("g3", {}) and "predictions for 59 of 59" in p.stdout,
          f"pairs {r2.get('pairs')} (expect 59: one golden, one target), prior variant run")
    check("neyman", res["g6"]["se_ratio_neyman"] < 0.97, f"neyman / plain {res['g6']['se_ratio_neyman']:.3f} (one noisy category)")

# ---- 2. delta-method se vs document bootstrap on real files
if len(sys.argv) > 2:
    gd, sd = sys.argv[1:3]
    m = json.load(open(f"{gd}/kld.json")); R = m["rows"]; Dn = len(m["docs"])
    gen = np.fromfile(f"{gd}/gen.u8", np.uint8).astype(bool); doc = np.zeros(R, np.int64)
    for i, d in enumerate(m["docs"]): doc[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]] = i
    def ds(t): return np.bincount(doc[gen], np.fromfile(f"{sd}/{t}.f32", np.float32).reshape(R, 4)[gen, 0].astype(float), Dn)
    prs = [("k3base", "k3base_s1"), ("k3base", "k3bdiag"), ("k2base", "k2bdiag"), ("k2base", "k2base_s2"),
           ("k3ldlq", "k3hout"), ("k2ldlq", "k2yb8192"), ("k3base", "k3cub"), ("k2base", "k2eB")]
    rr = []; bt = np.random.default_rng(0).integers(0, Dn, (4000, Dn))
    for x, y in prs:
        if not (os.path.exists(f"{sd}/{x}.f32") and os.path.exists(f"{sd}/{y}.f32")): continue
        A, B = ds(x), ds(y); e = B - B.sum() / A.sum() * A
        se_d = math.sqrt(Dn / (Dn - 1) * (e ** 2).sum()) / B.sum()
        se_b = np.log(B[bt].sum(1) / A[bt].sum(1)).std()
        rr.append(se_d / se_b); print(f"  {x} -> {y}: delta {se_d:.5f}  bootstrap {se_b:.5f}")
    check("delta vs bootstrap", rr and 0.9 < min(rr) and max(rr) < 1.1, f"ratios {np.round(rr, 3).tolist()}")

# ---- 3. aggsim's prefix simulation
rng = np.random.default_rng(2); st = np.array([0.1, 0.2, 0.35, 0.5, 0.7]); n = 200000; se = 0.02
z = rng.standard_normal((n, len(st)))
cum = np.cumsum(np.c_[z[:, :1] * np.sqrt(st[0]), z[:, 1:] * np.sqrt(np.diff(st))], 1)
bridge = cum - st * (cum[:, -1:] + rng.standard_normal((n, 1)) * np.sqrt(1 - st[-1]))
emp = (se * bridge / st).std(0); th = se * np.sqrt((1 - st) / st)
check("aggsim bridge", np.abs(emp / th - 1).max() < 0.01, f"sd / theory {np.round(emp / th, 4).tolist()}")
print("SELFTEST", "PASS" if ok else "FAIL")
