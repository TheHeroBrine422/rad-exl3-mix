#!/usr/bin/env python3
"""summ.py -- the w07 experiments' arm pick and verdict, from evalq.py --out score files (gen-masked mean KL; paired
per-document bootstrap as pcmp.py). Score files: DIR/<tag>.f32 (search slice), DIR/<tag>.fit.f32 (fit slice); a base
tag is read from DIR, else from Y (/data/yb). Tags: k<K><arm>[_s<seed>], base = k<K>base[_s<seed>].
  summ.py pick DIR K ARM[,ARM..] [BASEDIR]        the arm with the lowest fit-slice KL at seed 0 (table on stderr)
  summ.py verdict DIR GOLDEN Y K ARM RULE BAR [CTRL,CTRL..]
      per seed 0 / 1 / 2: KL(arm) / KL(base), paired z; "every arm seed below every base seed"; then
      RULE seed: PASS if every per-seed ratio <= BAR and every arm seed is below every base seed (ranked.md C1)
      RULE mean: PASS if the mean-KL ratio (3 seeds each) <= BAR and every arm seed is below every base seed (C13)
      RULE ctrl: CTRL = control tags (the same tensors requantized at other seeds on the old Hessians, against base seed
                 0): their ratios' SD s; PASS if the mean per-seed ratio <= min(BAR, 1 - 3 s / sqrt(3)) and every per-seed
                 ratio < 1 (C6)
      FAIL if the mean-KL ratio >= 1.00 (no gain at all); otherwise INCONCLUSIVE (status quo: not adopted)."""
import json, os, sys
import numpy as np
def f32(d, y, tag):
    p = f"{d}/{tag}" if os.path.exists(f"{d}/{tag}") else f"{y}/{tag}"
    return np.fromfile(p, np.float32).reshape(-1, 4) if os.path.exists(p) else None
def kl(A): g = A[:, 2] > 0; return float(A[g, 0].mean())
def paired(gd, A, B, N=2000):                     # pcmp.py's arithmetic: diff B - A, per-document bootstrap SE
    m = json.load(open(f"{gd}/kld.json")); gen = A[:, 2] > 0; sa, sb, n = [], [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); gg = gen[s]
        sa.append(A[s, 0][gg].sum()); sb.append(B[s, 0][gg].sum()); n.append(gg.sum())
    sa, sb, n = map(np.asarray, (sa, sb, n)); idx = np.random.default_rng(0).integers(0, len(n), (N, len(n)))
    se = ((sb[idx].sum(1) - sa[idx].sum(1)) / n[idx].sum(1)).std(); diff = sb.sum() / n.sum() - sa.sum() / n.sum()
    return float(diff / se)
sfx = lambda s: "" if s == 0 else f"_s{s}"
cmd = sys.argv[1]
if cmd == "pick":
    d, K, arms = sys.argv[2], sys.argv[3], sys.argv[4].split(","); y = sys.argv[5] if len(sys.argv) > 5 else "/data/yb"
    b = f32(d, y, f"k{K}base.fit.f32"); rows = []
    for a in arms:
        A = f32(d, y, f"k{K}{a}.fit.f32")
        if A is not None: rows.append((kl(A), a))
    if b is None or len(rows) != len(arms): sys.exit(f"pick K{K}: missing fit scores ({len(rows)} of {len(arms)} arms, base {b is not None})")
    for v, a in sorted(rows): print(f"  K{K} fit {a:6s} {v:.5f}  ratio to base {v / kl(b):.4f}", file=sys.stderr)
    print(min(rows)[1])
elif cmd == "verdict":
    d, gd, y, K, arm, rule, bar = sys.argv[2:9]; bar = float(bar); ctrl = sys.argv[9].split(",") if len(sys.argv) > 9 else []
    out = {"K": K, "arm": arm, "rule": rule, "bar": bar, "seeds": []}
    A = [f32(d, y, f"k{K}{arm}{sfx(s)}.f32") for s in range(3)]; B = [f32(d, y, f"k{K}base{sfx(s)}.f32") for s in range(3)]
    assert all(x is not None for x in A + B), f"K{K} {arm}: missing search scores"
    ka, kb = [kl(x) for x in A], [kl(x) for x in B]
    for s in range(3): out["seeds"].append({"seed": s, "kl": ka[s], "base": kb[s], "ratio": ka[s] / kb[s], "z": paired(gd, B[s], A[s])})
    r = [x["ratio"] for x in out["seeds"]]; mr = float(np.mean(ka) / np.mean(kb)); ebe = max(ka) < min(kb)
    out.update(mean_ratio=mr, mean_seed_ratio=float(np.mean(r)), every_beats_every=ebe)
    if rule == "seed": ok = max(r) <= bar and ebe
    elif rule == "mean": ok = mr <= bar and ebe
    else:
        cr = [kl(f32(d, y, f"k{K}{c}.f32")) / kb[0] for c in ctrl]; sd = float(np.std(cr, ddof=1))
        out.update(ctrl_ratios=cr, ctrl_sd=sd, bar_eff=min(bar, 1 - 3 * sd / 3 ** 0.5))
        ok = np.mean(r) <= out["bar_eff"] and max(r) < 1
    out["verdict"] = "PASS" if ok else "FAIL" if mr >= 1.0 else "INCONCLUSIVE"
    for x in out["seeds"]: print(f"K{K} {arm} seed {x['seed']}: {x['kl']:.5f} vs base {x['base']:.5f}  ratio {x['ratio']:.4f}  z {x['z']:+.1f}")
    print(f"K{K} {arm}: mean-KL ratio {mr:.4f}, per-seed ratios mean {np.mean(r):.4f} max {max(r):.4f}, every-beats-every {ebe}"
          + (f", control ratios {['%.4f' % c for c in out['ctrl_ratios']]} sd {out['ctrl_sd']:.4f} -> bar {out['bar_eff']:.4f}" if ctrl else "")
          + f"  =>  {out['verdict']} (rule {rule}, bar {bar})")
    with open(f"{d}/summary.jsonl", "a") as fo: fo.write(json.dumps(out) + "\n")
