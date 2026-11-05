#!/usr/bin/env python3
"""r90knap.py ratios | run WORKDIR [PAR] | report WORKDIR -- #90 rule 2: sub-1 options priced in knap4 (README "#90").

ratios   family curves from #90 (data/e3/sub1/sub1-cpu*.jsonl): rel(k) / rel(K1) per tensor, k = 0.125 .. 0.875; per
         family the geometric mean over its tensors (sub1ratios.json, the rule's price) and the worst tensor's ratio
         (sub1ratios-max.json, a conservative what-if). Written to data/e3/sub1/.
run      (cpu1; 58 single-thread knap4 solves, ~15 min at 16 at a time) copies knap4.py into WORKDIR, applies
         knap4-sub1.patch (KN_SUB1, KN_BK; see the patch), and solves quality-only (lambda 0, f16 path: at lambda 0
         knap4 never picks an int8 mode) at 1.5 / 1.75 / 2.0 / 2.25 / 2.5 bpw (the rule's sizes; 1.25 / 1.3 / 1.4 too,
         reported only) for
           menu v3   the V3 option table as k4v3test.sh (KN_V3 KN_FRAC KN_EMBT=2 KN_HEADSLAB KN_LEVELS KN_DAMP KN_SPEED)
           menu v3e  the same + K1.125 .. 1.875 at 1/8 steps (the i4 ladder's levels between K1 and K2, interpolated
                     log-linearly: #90's K1.25 / 1.5 sit within 1.5% of that line)
         x arm base (no sub-1), sub1 (KN_SUB1=sub1ratios.json), sub1max (KN_SUB1=sub1ratios-max.json). Each solve
         writes its .mix (KN_WRITE, check off) so the per-tensor picks can be read back. DP bucket 256 KiB (KN_BK):
         knap4's default 1 MiB rounds a 1024 x 5120 tensor's K0.75 to 0 buckets and the solve to ~2% of the exact
         optimum (adding options then raised the predicted KLD by up to 0.7%); the rule's menu (v3e) base / sub1 is
         also solved at 64 KiB to show the 256 KiB numbers have converged.
report   collects WORKDIR into data/e3/sub1/r90knap.jsonl (one line per menu / arm / target, with the sub-1 picks) and
         prints the table (data/e3/sub1/r90knap.md). Verdict per size: sub1 vs base on menu v3e (the V3 ladder has
         1/8 steps below K3, so the fair base has them too) at 256 KiB; menu v3 shown for comparison with
         knap4v3-2026-11-01 (its 1 MiB base reproduces that file's 2.0 bpw KLD 0.14796 exactly)."""
import collections, glob, json, math, os, re, shutil, subprocess, sys, time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
D = f"{ROOT}/data/e3/sub1"
SUB = [k / 8 for k in range(1, 8)]
TARGETS = [1.5, 1.75, 2.0, 2.25, 2.5]
EXTRA = [1.25, 1.3, 1.4]                                                              # reported only
BKS = [1 << 18, 1 << 16]                                                              # main, convergence check
LADDER = "2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.5,6"                  # knap4's KN_V3 default
MENUS = {"v3": LADDER, "v3e": ",".join(f"{1 + k / 8:g}" for k in range(1, 8)) + "," + LADDER}
ARMS = {"base": None, "sub1": "sub1ratios.json", "sub1max": "sub1ratios-max.json"}


def ratios():
    R = {}
    for p in sorted(glob.glob(f"{D}/sub1-cpu*.jsonl")):
        for l in open(p):
            o = json.loads(l)
            if o["method"] == "ldlq_cubic0.02_i8": R[(o["layer"], o["w"], float(o["bits"]))] = o["rel"]
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for (l, w, k), r in R.items():
        if k < 1 and (l, w, 1.0) in R: fam[w][k].append(r / R[(l, w, 1.0)])
    gm = {w: {f"{k:g}": math.exp(sum(map(math.log, v)) / len(v)) for k, v in sorted(d.items())} for w, d in fam.items()}
    mx = {w: {f"{k:g}": max(v) for k, v in sorted(d.items())} for w, d in fam.items()}
    n = {w: len(d[0.5]) for w, d in fam.items()}
    src = "data/e3/sub1/sub1-cpu*.jsonl (#90, i8c, 128 rows)"
    json.dump({"ratios": gm, "n": n, "how": "geometric mean of rel(k)/rel(K1) over the family's tensors", "src": src},
              open(f"{D}/sub1ratios.json", "w"), indent=1)
    json.dump({"ratios": mx, "n": n, "how": "worst tensor's rel(k)/rel(K1) in the family", "src": src},
              open(f"{D}/sub1ratios-max.json", "w"), indent=1)
    for w in sorted(gm): print(f"{w:12s} n={n[w]} " + " ".join(f"K{k} {v:.2f}" for k, v in gm[w].items()))


def run(wd, par):
    nf = f"{wd}/v3/nf"
    if not os.path.exists(f"{nf}/.patched"):        # a private copy of the tree knap4 reads, then the patch
        for d in ("v3/nf", "v3/e1b", "v3/e3/data", "v3/bench/data", "data/v3"):
            shutil.copytree(f"{ROOT}/{d}", f"{wd}/{d}", dirs_exist_ok=True)
        subprocess.run(["patch", "-p1", "-i", f"{ROOT}/v3/e3/sub1/knap4-sub1.patch"], cwd=wd, check=True)
        open(f"{nf}/.patched", "w").close()
    for a in ARMS.values():
        if a: shutil.copy(f"{D}/{a}", f"{wd}/{a}")
    os.makedirs(f"{wd}/out", exist_ok=True)
    M, A, F, P = "../../data/v3/kfit2-model-final.json", "../e1b/data/aq.jsonl", "../e1b/data/aqfactors.json", "../e1b/data/e0-ident.map"
    jobs = []
    spec = [("v3e", arm, tg, BKS[1]) for arm in ("base", "sub1") for tg in TARGETS] + \
           [(menu, arm, tg, BKS[0]) for menu in MENUS for arm in ARMS for tg in EXTRA + TARGETS]     # slowest first
    for menu, arm, tg, bk in spec:
        lad, rf = MENUS[menu], ARMS[arm]
        n = f"{menu}-{arm}-{tg:g}-b{bk}"; o = f"{wd}/out/{n}"
        if os.path.exists(f"{o}.jsonl") and os.path.getsize(f"{o}.jsonl"): continue
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KN_V3="1", KN_FRAC="1", KN_EMBT="2",
                   KN_HEADSLAB="1", KN_LEVELS="../../data/v3/levels", KN_DAMP="1", KN_MODES="f16",
                   KN_SPEED="../bench/data/speedcal-stock.jsonl", KN_LADDER=lad, KN_WRITE=f"{o}.d", KN_WRITE_CHECK="0",
                   KN_BK=str(bk))
        if rf: env["KN_SUB1"] = f"{wd}/{rf}"
        cmd = ["nice", "-n", "10", "python3", "knap4.py", M, A, F, P, f"{o}.part", f"{tg}", "0"]
        jobs.append((n, cmd, env, o))
    print(f"{len(jobs)} solves, {par} at a time", flush=True)
    running = []
    while jobs or running:
        while jobs and len(running) < par:
            n, cmd, env, o = jobs.pop(0)
            if os.path.exists(f"{o}.part"): os.remove(f"{o}.part")
            running.append((n, o, subprocess.Popen(cmd, cwd=nf, env=env, stdout=open(f"{o}.log", "w"), stderr=subprocess.STDOUT)))
        for x in list(running):
            if x[2].poll() is not None:
                running.remove(x); n, o, p = x
                if p.returncode == 0 and os.path.exists(f"{o}.part"): os.replace(f"{o}.part", f"{o}.jsonl")
                print(f"{n}: rc {p.returncode}", flush=True)
        if running: time.sleep(2)
    print("R90KNAP DONE", flush=True)


def report(wd):
    md = json.load(open(f"{ROOT}/data/v3/kfit2-model-final.json"))
    NP = {}                                           # tensor -> parameter count (bytes of K1 -> K2 step x 8)
    for t in md["tensors"]:
        o = t["opts"]
        if "ours-K1" in o and "ours-K2" in o: NP[t["w"]] = (o["ours-K2"][0] - o["ours-K1"][0]) * 8
    trunk = sum(v for w, v in NP.items() if w != "output.weight")
    rows = []
    for f in sorted(glob.glob(f"{wd}/out/*.jsonl")):
        menu, arm, tg, bk = os.path.basename(f)[:-6].split("-")
        r = json.loads(open(f).readline())
        mix = glob.glob(f"{f[:-6]}.d/*.mix")
        picks = {}
        if mix:
            for l in open(mix[0]):
                w, p = l.split()
                m = re.search(r"ours-K([\d.]+)\.rad$", p)
                picks[w] = float(m.group(1)) if m else os.path.basename(p)
        s1 = {w: k for w, k in picks.items() if isinstance(k, float) and k < 1}
        k1 = {w: k for w, k in picks.items() if isinstance(k, float) and 1 <= k < 2}
        fam = collections.defaultdict(lambda: [0, 0.0])
        for w, k in s1.items():
            fam[w.split(".")[-2]][0] += 1; fam[w.split(".")[-2]][1] += NP.get(w, 0) / trunk
        rows.append(dict(menu=menu, arm=arm, target=float(tg), bk=int(bk[1:]), bpw=r["bpw"], kld=r["kld"], emb=r["emb"],
                         sub1_share=sum(NP.get(w, 0) for w in s1) / trunk, k1x_share=sum(NP.get(w, 0) for w in k1) / trunk,
                         sub1_n=len(s1), sub1_fam={k: [v[0], round(v[1], 4)] for k, v in sorted(fam.items())},
                         sub1_k=dict(sorted(collections.Counter(f"{k:g}" for k in s1.values()).items())),
                         sub1_picks={w: k for w, k in sorted(s1.items(), key=lambda x: (int(x[0].split(".")[1]), x[0]))},
                         param_share=r["param_share"]))
    with open(f"{D}/r90knap.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    I = {(r["menu"], r["arm"], r["target"], r["bk"]): r for r in rows}
    B = BKS[0]
    out = ["# #90 rule 2: knap4 with sub-1 options (quality only, lambda 0)", "",
           "Predicted KLD (kfit2 model) per size; change vs the same menu without sub-1 (base). Rule: >= 3% lower "
           f"with the geomean family ratios (arm sub1) on menu v3e, at {B >> 10} KiB buckets. 1.25 / 1.3 / 1.4: reported only.",
           "", "| menu | bpw | base KLD | sub1 KLD | change | sub1max KLD | change | trunk params at K<1 (sub1) | "
           "at K1-1.875 base -> sub1 | emb base / sub1 | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for menu in MENUS:
        for tg in EXTRA + TARGETS:
            b, s, x = (I.get((menu, a, tg, B)) for a in ("base", "sub1", "sub1max"))
            if not b or not s: out.append(f"| {menu} | {tg:g} | incomplete |"); continue
            ch = s["kld"] / b["kld"] - 1
            xs = f"{x['kld']:.5f} | {x['kld'] / b['kld'] - 1:+.1%}" if x else "- | -"
            v = ("PASS" if ch <= -0.03 else "below 3%") if menu == "v3e" and tg in TARGETS else "(reported)"
            out.append(f"| {menu} | {tg:g} | {b['kld']:.5f} | {s['kld']:.5f} | {ch:+.2%} | {xs} | {s['sub1_share']:.1%} "
                       f"({s['sub1_n']} tensors) | {b['k1x_share']:.1%} -> {s['k1x_share']:.1%} | {b['emb']} / {s['emb']} | {v} |")
    out += ["", "## Menu granularity: K1.125-1.875 levels (v3e) vs knap4's default V3 ladder (v3), both without sub-1", ""]
    for tg in EXTRA + TARGETS:
        a, b = I.get(("v3", "base", tg, B)), I.get(("v3e", "base", tg, B))
        if a and b: out.append(f"- {tg:g} bpw: {a['kld']:.5f} -> {b['kld']:.5f} ({b['kld'] / a['kld'] - 1:+.1%}); "
                               f"trunk params at K1-1.875 {b['k1x_share']:.0%}")
    out += ["", "## Solver resolution (DP bucket): base / sub1 KLD and the sub-1 change", ""]
    for menu, bks in (("v3e", sorted({r["bk"] for r in rows if r["menu"] == "v3e"})),
                      ("v3", sorted({r["bk"] for r in rows if r["menu"] == "v3"}))):
        for tg in TARGETS:
            cells = []
            for bk in bks:
                b, s = I.get((menu, "base", tg, bk)), I.get((menu, "sub1", tg, bk))
                if b and s: cells.append(f"{bk >> 10} KiB {b['kld']:.5f} / {s['kld']:.5f} ({s['kld'] / b['kld'] - 1:+.2%})")
            if cells: out.append(f"- {menu} {tg:g} bpw: " + "; ".join(cells))
    out += ["", f"## Which tensors go sub-1 (menu v3e, arm sub1, {B >> 10} KiB): per family tensors (share of trunk params), K used", ""]
    for tg in EXTRA + TARGETS:
        s = I.get(("v3e", "sub1", tg, B))
        if not s: continue
        fams = ", ".join(f"{k} {v[0]} ({v[1]:.1%})" for k, v in sorted(s["sub1_fam"].items(), key=lambda x: -x[1][1]))
        ks = ", ".join(f"K{k}: {n}" for k, n in s["sub1_k"].items())
        out.append(f"- **{tg:g} bpw** ({s['sub1_n']} tensors, {s['sub1_share']:.1%}): {fams or 'none'}; {ks or '-'}")
        lay = collections.defaultdict(list)
        for w, k in s["sub1_picks"].items(): lay[w.split(".")[-2]].append(f"{w.split('.')[1]}:{k:g}")
        for f_, v in sorted(lay.items()): out.append(f"  - {f_}: " + " ".join(v))
    open(f"{D}/r90knap.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "ratios": ratios()
    elif c == "run": run(os.path.abspath(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    elif c == "report": report(os.path.abspath(sys.argv[2]))
    else: print(__doc__); sys.exit(1)
