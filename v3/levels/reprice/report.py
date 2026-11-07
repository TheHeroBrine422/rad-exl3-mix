#!/usr/bin/env python3
"""report.py DATADIR -- w58 tables for reprice.md from run5's out/*.jsonl (both boxes' outputs copied into DATADIR).
Every KLD here is a kfit2-model prediction (no measurement). Fast variant rule (chosen here, before the results): the
lowest predicted KLD among the arm's lambda-grid solutions whose GEMM time is within 2% of the fastest one at that
size (the all-A8 end), i.e. "int8 wherever it costs least, at ~full A8 speed"."""
import collections, glob, json, math, os, re, sys

D = sys.argv[1]
R = {}
for p in glob.glob(f"{D}/*.jsonl"):
    n = os.path.basename(p)[:-6]
    m = re.fullmatch(r"(\w+?)-(acc|f)-([\d.]+)(?:-l([\de.+-]+))?", n)
    if not m: continue
    ls = [json.loads(l) for l in open(p) if l.strip()]
    if ls: R[(m[1], m[2], float(m[3]), float(m[4]) if m[4] else 0.0)] = ls[0]
ARMS = sorted({k[0] for k in R})
SIZES = sorted({k[2] for k in R})
V2B = {2.0: 0.1644, 2.5: 0.0666, 3.0: 0.0306, 3.5: 0.0171, 4.0: 0.00828, 4.5: 0.00480, 5.0: 0.00256, 6.0: 0.000914,
       7.0: 0.00045}                                          # V2 board KLDs (projections-2026-11-06.md)
PROJ_MENU = {2.0: -21, 3.0: -10, 4.0: -10, 5.0: -6}           # 11-04/05 paper: knap4 V3 menu vs V2 menu (model)
PROJ_SWJ = {2.0: -26, 3.0: -16}                               # 11-06 re-base: + swj (K2.5 gain, levels <= K3.5)
PROJ_FREEZE = {2.0: 0.120, 2.5: 0.0487, 3.0: 0.0243, 3.5: 0.0136, 4.0: 0.00656, 4.5: 0.00380, 5.0: 0.00226,
               6.0: 0.000806, 7.0: 0.000397}                  # projections-2026-11-06 "V3 at freeze", expected column

def gemm_x(r): return r["t_f16_ms"] / r["t_lin_ms"]
def fast(arm, s):
    pts = [r for (a, v, z, _), r in R.items() if a == arm and v == "f" and z == s]
    if not pts: return None
    tmin = min(r["t_lin_ms"] for r in pts)
    return min((r for r in pts if r["t_lin_ms"] <= 1.02 * tmin), key=lambda r: r["kld"])
def acc(arm, s): return R.get((arm, "acc", s, 0.0))
def lv(c):
    m = re.fullmatch(r"(i4|i8|m1)-K([\d.]+)", c)
    return (m[1], float(m[2])) if m else None
def shares(r, mode=None):
    """param share per option (trunk + head), optionally only tensors in a mode"""
    if "picks" not in r: return r["param_share"]
    return r["param_share"] if mode is None else None

out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
def pct(a, b): return f"{100 * (a / b - 1):+.1f}%"

# ---- 1. per size: accurate and fast picks, main arm
P("## T1 main arm (swj all, no CD): accurate and fast per size\n")
P("| size | acc KLD | emb | head | acc: i8 / m1 / i4 / turbo param share | acc K>=6: i8 vs m1 | fast KLD | vs acc | GEMM x | A8 FLOP share | fast K>=6: i8 / m1 (A8 / f16) | fast lambda |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
def cbshare(r):
    c = collections.Counter()
    for k, v in r["param_share"].items():
        x = lv(k); c[x[0] if x else ("head" if k.startswith("ours-") else "turbo/bf16")] += v
    return c
def hi_split(r):
    c = collections.Counter()
    for k, v in r["param_share"].items():
        x = lv(k)
        if x and x[1] >= 6 and x[0] in ("i8", "m1"): c[f"{x[0]}-K{x[1]:g}"] += v
    return " ".join(f"{k} {100 * v:.1f}" for k, v in sorted(c.items(), key=lambda kv: (kv[0][:2], float(kv[0][4:])))) or "-"
for s in SIZES:
    a, f = acc("main", s), fast("main", s)
    if not a: continue
    cb = cbshare(a)
    fs = "| - | - | - | - | - | - |"
    if f:
        fl = f["flop_share"].get("a8", 0)
        fs = f"| {f['kld']:.5g} | {pct(f['kld'], a['kld'])} | {gemm_x(f):.3f} | {fl:.3f} | {hi_split(f)} | {f['lambda']:g} |"
    P(f"| {s:g} | {a['kld']:.5g} | {a['emb']} | {a['picks']['output.weight'][0]} | {100 * cb['i8']:.1f} / {100 * cb['m1']:.1f} / {100 * cb['i4']:.1f} / "
      f"{100 * cb['turbo/bf16']:.2f} | {hi_split(a)} " + fs)

# ---- 1b. high-K detail for the fast variant: which codebook at K>=6 runs on A8 vs f16
P("\n## T1b fast variant at 5.0-7.0, main arm: per K>=6 level, param share on A8 / f16 (all lambdas of the grid)\n")
for s in [x for x in SIZES if x >= 5]:
    for (a_, v, z, lam), r in sorted(R.items()):
        if a_ != "main" or v != "f" or z != s or "picks" not in r: continue
        c = collections.Counter(); tot = 0
        for w, (opt, mode) in r["picks"].items():
            x = lv(opt)
            if x and x[1] >= 6: c[(opt, mode[:3])] += 1
        P(f"- {s:g} l{lam:g}: KLD {r['kld']:.5g} (a {r['kld_a']:.2g}) GEMM x{gemm_x(r):.3f}; tensors " +
          ", ".join(f"{o}@{m} {n}" for (o, m), n in sorted(c.items(), key=lambda kv: -kv[1])[:12]))

# ---- 2. KLD vs projections
P("\n## T2 predicted KLD vs V2 and the projections (accurate; fast in brackets)\n")
P("| size | main acc (fast) | V2-menu model | main / V2 menu | 11-04/06 projection (menu; +swj) | old knap4 V3 acc | main / old | board-anchored = V2 board x ratio (fast) | projections-11-06 V3-at-freeze exp. |")
P("|---|---|---|---|---|---|---|---|---|")
for s in SIZES:
    a, f, v2, o = acc("main", s), fast("main", s), acc("v2menu", s), acc("old", s)
    if not a: continue
    rv = a["kld"] / v2["kld"] if v2 else None
    pj = (f"{PROJ_MENU[s]}%" if s in PROJ_MENU else "-") + ("; " + f"{PROJ_SWJ[s]}%" if s in PROJ_SWJ else "")
    ba = f"{V2B[s] * rv:.4g}" + (f" ({V2B[s] * f['kld'] / v2['kld']:.4g})" if f else "") if (rv and s in V2B) else "-"
    P(f"| {s:g} | {a['kld']:.5g} ({f['kld']:.5g}) | {v2['kld']:.5g} | {pct(a['kld'], v2['kld'])} | {pj} | "
      f"{o['kld']:.5g} | {pct(a['kld'], o['kld'])} | {ba} | {PROJ_FREEZE.get(s, '-')} |" if v2 and o and f else f"| {s:g} | {a['kld']:.5g} | ... |")

# ---- 3. arms side by side (accurate / fast KLD)
P("\n## T3 all arms: accurate KLD, fast KLD (GEMM x), vs main\n")
P("| arm | " + " | ".join(f"{s:g}" for s in SIZES) + " |")
P("|---|" + "---|" * len(SIZES))
for arm in ARMS:
    cells = []
    for s in SIZES:
        a, f, ma, mf = acc(arm, s), fast(arm, s), acc("main", s), fast("main", s)
        c = []
        if a: c.append(f"{a['kld']:.4g}" + (f" ({pct(a['kld'], ma['kld'])})" if ma and arm != "main" else ""))
        if f: c.append(f"f {f['kld']:.4g}" + (f" ({pct(f['kld'], mf['kld'])})" if mf and arm != "main" else "") + f" x{gemm_x(f):.2f}")
        cells.append("<br>".join(c) or "")
    P(f"| {arm} | " + " | ".join(cells) + " |")

# ---- 4. level usage (main arm): per level, sizes where it is picked (acc / fast), max param share
P("\n## T4 level usage, main arm (accurate and fast at every size; fast = all lambdas)\n")
use = collections.defaultdict(lambda: {"acc": {}, "fast": {}, "fastall": collections.Counter()})
NT = collections.defaultdict(lambda: collections.Counter())
for (arm, v, s, lam), r in R.items():
    if arm != "main": continue
    for k, sh in r["param_share"].items():
        if v == "acc": use[k]["acc"][s] = sh
        else:
            use[k]["fastall"][s] = max(use[k]["fastall"][s], sh)
    if "picks" in r:
        for w, (opt, mode) in r["picks"].items(): NT[(v, s, lam)][opt] += 1
for s in SIZES:
    f = fast("main", s)
    if f:
        for k, sh in f["param_share"].items(): use[k]["fast"][s] = sh
def key(k):
    x = lv(k)
    return (0, {"i4": 0, "i8": 1, "m1": 2}[x[0]], x[1]) if x else (1, 0, k)
ladder = ([("i4", k) for k in [1] + [1 + i / 8 for i in range(1, 17)]] + [("i8", 1 + i / 8) for i in range(1, 33)] +
          [("i8", 5 + i / 4) for i in range(1, 9)] + [("m1", 6 + i / 4) for i in range(0, 9)])
P("| level | accurate: sizes (max share) | fast variant: sizes (max share) | any fast lambda: sizes |")
P("|---|---|---|---|")
never = []
for cb, k in ladder:
    c = f"{cb}-K{k:g}"; u = use.get(c)
    fmt = lambda d: (",".join(f"{s:g}" for s in sorted(d)) + f" ({100 * max(d.values()):.1f}%)") if d else "-"
    if not u or not (u["acc"] or u["fastall"]): never.append(c); P(f"| {c} | - | - | - |"); continue
    P(f"| {c} | {fmt(u['acc'])} | {fmt(u['fast'])} | {fmt(dict(u['fastall']))} |")
others = sorted(k for k in use if not lv(k))
P("\nother options picked (head, turbo, bf16): " + "; ".join(f"{k}: acc {','.join(f'{s:g}' for s in sorted(use[k]['acc']))} / fast {','.join(f'{s:g}' for s in sorted(use[k]['fast']))}" for k in others))
P("\nnever picked by the main arm at 2.0-7.0 (any variant, any lambda): " + ", ".join(never))

# ---- 5. tensor counts at the K6-7 levels (main, accurate and fast)
P("\n## T5 tensors per level at K>=6 (main arm)\n")
for s in [x for x in SIZES if x >= 5]:
    a, f = acc("main", s), fast("main", s)
    for nm, r in (("acc", a), ("fast", f)):
        if not r or "picks" not in r: continue
        c = collections.Counter((o, m) for w, (o, m) in r["picks"].items() if lv(o) and lv(o)[1] >= 6)
        P(f"- {s:g} {nm}: " + ", ".join(f"{o}@{m} {n}" for (o, m), n in sorted(c.items(), key=lambda kv: key(kv[0][0]))))
print("\n".join(out))
