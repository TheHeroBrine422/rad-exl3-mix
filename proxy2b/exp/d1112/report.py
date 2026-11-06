#!/usr/bin/env python3
"""report.py O -- d1112 (README.md): reads O/result.txt ("pair A -> B" lines of pcmp.py, eval and drift lines) and
applies the README's rules for item 1 (stack x X, K2 / K3), item 2 (K4 / K5) and item 3 (fractional-K drift).
Markdown to stdout. Missing pairs are shown as missing; a verdict is "interim" until its decision seeds are all in."""
import math, re, sys

O = sys.argv[1]
P, KL, DRIFT = {}, {}, {}
for l in open(f"{O}/result.txt"):
    m = re.search(r"pair (\S+) -> (\S+): \"ratio\": ([0-9.]+), \"diff\": [0-9.e+-]+, \"se\": [0-9.e+-]+, \"z\": ([0-9.e+-]+)", l)
    if m: P[(m[1], m[2])] = (float(m[3]), float(m[4])); continue
    m = re.search(r"^\S+ (\S+) drift: \"drift\": ([0-9.]+)", l)
    if m: DRIFT[m[1]] = float(m[2]); continue
    m = re.search(r"^\S+ (\S+): \"kld\": ([0-9.e+-]+)", l)
    if m: KL[m[1]] = float(m[2])

sfx = lambda s: "" if s == 0 else f"_s{s}"

def stat(a, b, seeds):
    """per-seed (seed, ratio, z) of pair a+sfx -> b+sfx, n, geo, mean log, SE over seeds"""
    rs = [(s, *P[(a + sfx(s), b + sfx(s))]) for s in seeds if (a + sfx(s), b + sfx(s)) in P]
    if not rs: return None
    lg = [math.log(r) for _, r, _ in rs]; n = len(lg); m = sum(lg) / n
    se = (sum((x - m) ** 2 for x in lg) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else float("nan")
    return dict(rs=rs, n=n, geo=math.exp(m), mlog=m, se=se, lt1=all(r < 1 for _, r, _ in rs), gt1=all(r > 1 for _, r, _ in rs))

def fmt(st):
    if st is None: return "— | — | — | —"
    return (" ".join(f"s{s} {r:.4f} ({z:+.1f})" for s, r, z in st["rs"]) +
            f" | {st['n']} | **{st['geo']:.4f}** | {st['mlog']:+.4f} ± {st['se']:.4f}")

def stack_rows(ks, seeds):
    print("| K | golden | per seed: ratio sx / st (z) | n | geo | mean log ± SE |\n|---|---|---|---|---|---|")
    for k in ks:
        for g, lab in (("", "search"), ("vf-", "verify"), ("xc-", "xcorp2"), ("ag-", "agent")):
            print(f"| K{k} | {lab} | {fmt(stat(f'{g}k{k}st', f'{g}k{k}sx', seeds))} |")
    print()

def stages(ks, seeds):
    print("Stage readouts (search; not deciding) and the reproduction check (d1104's k{K}edyswjasym_r0 -> today's st):\n")
    print("| K | pair | per seed | n | geo | mean log ± SE |\n|---|---|---|---|---|---|")
    for k in ks:
        for a, b, lab in ((f"k{k}ste", f"k{k}sxe", "X on the trunk quant (sxe / ste)"),
                          (f"k{k}ste", f"k{k}st", "stack over its trunk quant (st / ste)")):
            print(f"| K{k} | {lab} | {fmt(stat(a, b, seeds))} |")
        # swj stage and the reproduction check carry their suffix inside / a prefix: built per seed
        for lab, f in (("X after swj (sxe_swj / ste_swj)", lambda s: (f"k{k}ste{sfx(s)}_swj", f"k{k}sxe{sfx(s)}_swj")),
                       ("repro: st / d1104 stack", lambda s: (f"d1104:k{k}edyswjasym_r0{sfx(s)}", f"k{k}st{sfx(s)}"))):
            rs = [(s, *P[f(s)]) for s in seeds if f(s) in P]
            if not rs: print(f"| K{k} | {lab} | — | — | — | — |"); continue
            lg = [math.log(r) for _, r, _ in rs]; m = sum(lg) / len(lg)
            print(f"| K{k} | {lab} | " + " ".join(f"s{s} {r:.4f} ({z:+.1f})" for s, r, z in rs) + f" | {len(rs)} | {math.exp(m):.4f} | {m:+.4f} |")
    print()

def vkeep(k, seeds):
    s, v = stat(f"k{k}st", f"k{k}sx", seeds), stat(f"vf-k{k}st", f"vf-k{k}sx", seeds)
    if not s or not v: return None, None
    if s["mlog"] >= 0: return False, None
    return v["mlog"] <= 2 / 3 * s["mlog"], v["mlog"] / s["mlog"]

def crossing(a, b):
    return a is not None and b is not None and ((a["geo"] <= 0.99) != (b["geo"] <= 0.99) or (a["geo"] > 0.995) != (b["geo"] > 0.995))

# ---------------------------------------------------------------- item 1
print("# d1112 report\n")
print("## Item 1 · stack × X (hi-X0.5), K3 / K2\n")
D1, ALL1 = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6]
print("Decision seeds 0–4:\n"); stack_rows([3, 2], D1)
S = {k: stat(f"k{k}st", f"k{k}sx", D1) for k in (3, 2)}
if any(S[k] is None for k in (3, 2)): print("**Item 1: no data yet.**\n")
else:
    n_ok = all(S[k]["n"] == 5 for k in (3, 2))
    sp = {k: S[k]["geo"] <= 0.99 and S[k]["lt1"] for k in (3, 2)}
    vk = {k: vkeep(k, D1) for k in (3, 2)}
    dont = any(S[k]["geo"] > 0.995 for k in (3, 2))
    if all(sp.values()) and all(vk[k][0] for k in (3, 2)): v = "STACKS (adopt-compatible)"
    elif dont: v = "DOESN'T STACK"
    elif all(sp.values()): v = "SEARCH-ONLY GAIN (verify keeps < 2/3)"
    else: v = "PARTIAL (the 27B pairs decide)"
    R = {k: stat(f"k{k}st", f"k{k}sx", ALL1) for k in (3, 2)}
    frag = any(crossing(S[k], R[k]) for k in (3, 2) if R[k] and R[k]["n"] > S[k]["n"])
    print(f"**Item 1 verdict{'' if n_ok else ' (INTERIM: n < 5 at some K)'}: {v}{' — FRAGILE (seeds 5–6 cross a threshold)' if frag else ''}.**  ")
    for k in (3, 2):
        kv, frac = vk[k]
        print(f"K{k}: search geo {S[k]['geo']:.4f} (n {S[k]['n']}), every seed < 1: {S[k]['lt1']}; verify keeps "
              f"{'—' if frac is None else f'{frac:.0%}'} of the search gain ({'ok' if kv else 'no'}).  ")
    for k in (3, 2):
        x, a = stat(f"xc-k{k}st", f"xc-k{k}sx", D1), stat(f"ag-k{k}st", f"ag-k{k}sx", D1)
        if x and x["geo"] > 0.98: print(f"FLAG K{k}: xcorp2 geo {x['geo']:.4f} > 0.98 (X loses its xcorp point on the stack).  ")
        if a and a["geo"] > 1.02: print(f"FLAG K{k}: agent geo {a['geo']:.4f} > 1.02.  ")
    print()
print("All seeds (0–6, robustness):\n"); stack_rows([3, 2], ALL1)
stages([3, 2], ALL1)

# ---------------------------------------------------------------- item 2
print("## Item 2 · transfer, K4 / K5\n")
D2, ALL2 = [0, 1, 2], [0, 1, 2, 3, 4]
print("Decision seeds 0–2:\n"); stack_rows([4, 5], D2)
for k in (4, 5):
    s = stat(f"k{k}st", f"k{k}sx", D2)
    if s is None: print(f"K{k}: no data yet.  "); continue
    kv, frac = vkeep(k, D2)
    if s["geo"] <= 0.99 and s["lt1"]: v = "X AT THIS K" if kv else "SEARCH-ONLY GAIN"
    elif 0.99 < s["geo"] < 1.01: v = "HARMLESS"
    elif s["geo"] >= 1.01 and s["gt1"]: v = "KEEP X OFF"
    else: v = "INCONCLUSIVE"
    r = stat(f"k{k}st", f"k{k}sx", ALL2); frag = r and r["n"] > s["n"] and crossing(s, r)
    print(f"**K{k}{'' if s['n'] == 3 else ' (INTERIM)'}: {v}** — geo {s['geo']:.4f} (n {s['n']}), verify keeps "
          f"{'—' if frac is None else f'{frac:.0%}'}{'; FRAGILE (seeds 3–4)' if frag else ''}.  ")
print("\nAll seeds (0–4, robustness):\n"); stack_rows([4, 5], ALL2)
stages([4, 5], ALL2)

# ---------------------------------------------------------------- item 3
print("## Item 3 · fractional-K drift: interpolated (default) vs exact lookup (OURQ_DRIFT_EXACT=1)\n")
EXP = {"2.25": 1.01349, "2.75": 1.0065, "1.75": 1.02646, "1.25": 1.05726}
print("| K | per seed: ratio fi / fx (z) | n | geo | mean log ± SE | drift i / x seen | class |\n|---|---|---|---|---|---|---|")
conf = rev = 0; full = True; bad = []
for k in ("2.25", "2.75", "1.75", "1.25"):
    s = stat(f"f{k}x", f"f{k}i", [0, 1, 2])
    di = sorted({v for t, v in DRIFT.items() if t.startswith(f"f{k}i")}); dx = sorted({v for t, v in DRIFT.items() if t.startswith(f"f{k}x")})
    if any(abs(v - EXP[k]) > 1e-4 for v in di) or any(abs(v - 1.0) > 1e-9 for v in dx): bad.append(k)
    c = "—"
    if s:
        c = "confirm" if s["geo"] <= 0.99 and s["lt1"] else "revert" if s["geo"] >= 1.01 and s["gt1"] else "none"
        if k != "1.25": conf += c == "confirm"; rev += c == "revert"; full &= s["n"] == 3
    else: full &= k == "1.25"
    print(f"| {k}{'' if k != '1.25' else ' (reported)'} | {fmt(s)} | {di} / {dx} | {c} |")
v = "(a) INTERPOLATION CONFIRMED" if conf >= 2 else "(b) REVERT TO THE EXACT LOOKUP (lead's call)" if rev >= 2 else "(c) NO MEASURABLE EFFECT: keep the default"
print(f"\n**Item 3{'' if full else ' (INTERIM)'}: {v}**{' — DRIFT CHECK FAILED at K ' + ', '.join(bad) + ' (arms invalid)' if bad else ''}.\n")

# ---------------------------------------------------------------- mean KLDs
print("## Mean KLD per arm (search, gen-masked)\n")
for k in (3, 2, 4, 5):
    row = [f"{a}{sfx(s)} {KL[f'k{k}{a}{sfx(s)}']:.5f}" for s in ALL1 for a in ("st", "sx") if f"k{k}{a}{sfx(s)}" in KL]
    if row: print(f"- K{k}: " + ", ".join(row))
