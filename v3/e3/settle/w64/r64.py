#!/usr/bin/env python3
"""r64.py gate|pick|report|done O -- g64.sh's readouts (../w64.md items 1 and 3), from O/result.txt's lines
("TAG: \"kld\": ..." and "pair A -> B: \"ratio\": r, \"diff\": d, \"se\": s, \"z\": z").
  gate    item 1's gate: pairs d1112:k3st -> u3 and d1112:k3ste_swj -> p3 within +-1% (exit 1 otherwise)
  pick    item 3's alpha* per (stack, K) by C9's rule (#85) from the seed-0 fit screen -> JSON on stdout
          {"u3": "attn_o=a,gdn_out=b,down=c" | "none", ...}
  report  markdown: gate, item 1 cells / verdicts, item 3 screen / alpha* / confirm / verdicts
  done    exit 0 when every arm of items 1 and 3 is scored"""
import json, math, os, re, sys

cmd, O = sys.argv[1], sys.argv[2]
P, KL = {}, {}
for l in open(f"{O}/result.txt") if os.path.exists(f"{O}/result.txt") else []:
    m = re.search(r"pair (\S+) -> (\S+): \"ratio\": ([0-9.]+), \"diff\": ([0-9.e-]+), \"se\": ([0-9.e-]+), \"z\": ([0-9.e-]*)", l)
    if m:
        P[(m.group(1), m.group(2))] = {"r": float(m.group(3)), "d": float(m.group(4)), "se": float(m.group(5)),
                                       "z": float(m.group(6)) if m.group(6) else 0.0}
        continue
    m = re.search(r"^\S+ (\S+): \"kld\": ([0-9.e-]+)", l)
    if m: KL[m.group(1)] = float(m.group(2))
sfx = lambda s: "" if s == 0 else f"_s{s}"
geo = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
FAMS = ("attn_o", "gdn_out", "down"); ALPHAS = (-1, 1, 2, 3)
CELLS = (("3", "2.5"), ("3", "3.5"), ("2", "2.5"))


def gate():
    r = {b: P.get((a, b), {}).get("r") for a, b in (("d1112:k3st", "u3"), ("d1112:k3ste_swj", "p3"))}
    ok = all(v is not None and abs(v - 1) <= 0.01 for v in r.values())
    return ok, f"GATE {'PASS' if ok else 'FAIL'}: d1112 -> fresh (u3 asym, p3 plain): {r}"


def pick():
    out = {}
    for st in "up":
        for k in ("3", "2"):
            b = f"fit-{st}{k}"; al = {}
            for fam in FAMS:
                pts = [(0.0, 0.0, 0.0)] + [(a, P[(b, f"{b}-{fam}{a}")]["d"], P[(b, f"{b}-{fam}{a}")]["z"])
                                           for a in ALPHAS if (b, f"{b}-{fam}{a}") in P]
                if len(pts) < 5: al[fam] = None; continue
                # least-squares quadratic through (alpha, diff)
                n = len(pts); sx = [sum(p[0] ** j for p in pts) for j in range(5)]
                sy = [sum(p[1] * p[0] ** j for p in pts) for j in range(3)]
                A = [[sx[i + j] for j in range(3)] for i in range(3)]
                c = solve3(A, sy)                               # y = c0 + c1 a + c2 a^2
                best = min(pts, key=lambda p: p[1])
                if c[2] > 0: v = max(-1.0, min(3.0, -c[1] / (2 * c[2])))
                else: v = best[0]
                v = round(v * 20) / 20
                if not (best[0] != 0 and best[2] <= -2): v = 0.0
                al[fam] = v
            if any(v is None for v in al.values()): out[f"{st}{k}"] = "missing"
            elif all(v == 0 for v in al.values()): out[f"{st}{k}"] = "none"
            else: out[f"{st}{k}"] = ",".join(f"{f}={al[f]:g}" for f in FAMS)
    return out


def solve3(A, b):
    import copy
    A = copy.deepcopy(A); b = list(b); n = 3
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(A[r][i])); A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]
        for r in range(i + 1, n):
            f = A[r][i] / A[i][i]
            for c in range(i, n): A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)): x[i] = (b[i] - sum(A[i][c] * x[c] for c in range(i + 1, n))) / A[i][i]
    return x


def item1():
    L = ["## Item 1 · #111 writers asym under knapsack mixes (search; verify in brackets)", "",
         "| cell (Kw, Ks) | seed | r_x = xs/xc | z | r_m = xm/xc | z | verify r_x | verify r_m |", "|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    for kw, ks in CELLS:
        rx, rm, vx, vm = [], [], [], []
        for s in (0, 1, 2):
            x = sfx(s); c = f"xc{kw}-{ks}{x}"
            px = P.get((c, f"xs{kw}-{ks}{x}")); pm = P.get((c, f"xm{kw}-{ks}{x}"))
            qx = P.get((f"vf-{c}", f"vf-xs{kw}-{ks}{x}")); qm = P.get((f"vf-{c}", f"vf-xm{kw}-{ks}{x}"))
            if px: rx.append(px["r"])
            if pm: rm.append(pm["r"])
            if qx: vx.append(qx["r"])
            if qm: vm.append(qm["r"])
            f = lambda p, k: f"{p[k]:.4f}" if p and k == "r" else (f"{p[k]:+.1f}" if p else "–")
            L.append(f"| ({kw}, {ks}) | {s} | {f(px, 'r')} | {f(px, 'z')} | {f(pm, 'r')} | {f(pm, 'z')} | {f(qx, 'r')} | {f(qm, 'r')} |")
        gx, gm = geo(rx), geo(rm)
        kap = math.log(gx) / math.log(gm) if len(rx) == 3 and len(rm) == 3 and gm < 0.99 else float("nan")
        L.append(f"| ({kw}, {ks}) | **geo** | **{gx:.4f}** | | **{gm:.4f}** | | {geo(vx):.4f} | {geo(vm):.4f} |")
        L.append(f"| | κ = ln geo r_x / ln geo r_m | **{kap:.2f}** | | | | | |")
        if len(rx) == 3 and len(rm) == 3:
            verdicts[(kw, ks)] = ("survives" if gx <= 0.99 and all(r < 1 for r in rx) and kap >= 0.5 else
                                  "fails" if gx >= 1.0 else "partial")
    if len(verdicts) == 3:
        v = set(verdicts.values())
        rule = "FAILS" if "fails" in v else "SURVIVES" if v == {"survives"} else "PARTIAL"
        L += ["", f"**Rule (w64.md item 1): {rule}** (per cell: {', '.join(f'({a}, {b}) {c}' for (a, b), c in verdicts.items())})"]
    else: L += ["", f"(rule pending: {len(verdicts)} of 3 cells complete)"]
    return L


def item3():
    L = ["## Item 3 · #112 C9 α on the writers (st = after asym, pl = plain)", "", "### Screen (seed 0, fit): ratio (z) vs α = 0", "",
         "| stack K | family | α −1 | α 1 | α 2 | α 3 |", "|---|---|---|---|---|---|"]
    for st in "up":
        for k in ("3", "2"):
            b = f"fit-{st}{k}"
            for fam in FAMS:
                c = [P.get((b, f"{b}-{fam}{a}")) for a in ALPHAS]
                L.append(f"| {'st' if st == 'u' else 'pl'} K{k} | {fam} | " + " | ".join(f"{p['r']:.4f} ({p['z']:+.1f})" if p else "–" for p in c) + " |")
    try: al = pick()
    except KeyError: al = {}
    L += ["", f"α* (C9's pick rule): {json.dumps(al)}", "", "### Confirm (search; verify)", "",
          "| stack K | α* | s0 | s1 | s2 | geo | verify geo | verify keeps |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for st in "up":
        for k in ("3", "2"):
            rs, vs = [], []
            for s in (0, 1, 2):
                x = sfx(s); p = P.get((f"{st}{k}{x}", f"{st}{k}{x}c9")); q = P.get((f"vf-{st}{k}{x}", f"vf-{st}{k}{x}c9"))
                rs.append(p["r"] if p else None); vs.append(q["r"] if q else None)
            ok = all(v is not None for v in rs + vs)
            keep = (sum(math.log(v) for v in vs) / sum(math.log(v) for v in rs)) if ok and geo(rs) != 1 else float("nan")
            L.append(f"| {'st' if st == 'u' else 'pl'} K{k} | {al.get(st + k, '–')} | " + " | ".join(f"{v:.4f}" if v else "–" for v in rs)
                     + f" | {geo([v for v in rs if v]):.4f} | {geo([v for v in vs if v]):.4f} | {keep:.2f} |")
            res[(st, k)] = (rs, vs, keep, al.get(st + k))
    for st, name in (("u", "st (adoption)"), ("p", "pl (reported)")):
        r = [res.get((st, k)) for k in ("3", "2")]
        if any(x is None for x in r) or any(x[3] is None for x in r): continue
        if all(x[3] == "none" for x in r): L.append(f"\n**{name}: FAIL** (α* = 0 for every family at both K)"); continue
        if any(None in x[0] or None in x[1] for x in r if x[3] != "none"): L.append(f"\n{name}: pending"); continue
        g = {k: (geo(x[0]) if x[3] != "none" else 1.0) for k, x in zip(("3", "2"), r)}
        pas = {k: x[3] != "none" and g[k] <= 0.99 and all(v < 1 for v in x[0]) and x[2] >= 2 / 3 for k, x in zip(("3", "2"), r)}
        v = "PASS → 27B fixed-code pair" if all(pas.values()) else "FAIL" if all(g[k] > 0.995 for k in g) else "PARTIAL"
        L.append(f"\n**{name}: {v}** (geo K3 {g['3']:.4f}, K2 {g['2']:.4f}; per-K pass {pas})")
    return L


if cmd == "gate":
    ok, msg = gate(); print(msg); sys.exit(0 if ok else 1)
if cmd == "pick":
    a = pick(); print(json.dumps(a)); sys.exit(0 if "missing" not in a.values() else 1)
if cmd == "report":
    L = ["# w64 4070 readout (g64.sh)", "", gate()[1], ""] + item1() + [""] + item3()
    L += ["", "## KLD per arm", "", "| tag | KLD |", "|---|---|"] + [f"| {t} | {v:.5f} |" for t, v in sorted(KL.items())]
    print("\n".join(L)); sys.exit(0)
if cmd == "done":
    need = [f"{a}{kw}-{ks}{sfx(s)}" for kw, ks in CELLS for a in ("xc", "xs", "xm") for s in (0, 1, 2)]
    need += [f"vf-{t}" for t in need]
    try: al = json.load(open(f"{O}/alpha.json"))
    except (OSError, ValueError): sys.exit(1)
    for st in "up":
        for k in ("3", "2"):
            if al.get(st + k) != "none": need += [f"{v}{st}{k}{sfx(s)}c9" for s in (0, 1, 2) for v in ("", "vf-")]
    miss = [t for t in need if not os.path.exists(f"{O}/{t}.f32")]
    print(f"{len(need) - len(miss)} / {len(need)} arms scored"); sys.exit(0 if not miss else 1)
