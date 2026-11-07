#!/usr/bin/env python3
"""r90knapx.py run WORKDIR [PAR] | report WORKDIR -- w46 item 3: #90's rule 2 (knap4 with sub-1 options, quality only)
re-priced with the settle.py sub-1 curves. A wrapper around v3/e3/sub1/r90knap.py (unchanged): same knap4 copy and
patch, same settings, but data dir data/e3/settle/reval, menu v3e only (the rule's menu), 256 KiB buckets only, and
the arms
  base   no sub-1
  sub1   KN_SUB1 = sub1ratios-settle.json (settle.py, int4grid, 2 samples; revalreport.py --sub1json)
  sub1max  KN_SUB1 = sub1ratios-old.json (a copy of data/e3/sub1/sub1ratios.json: the old trellis2 i8c curves),
         so the old pricing is re-solved in the same run (r90knap's report labels this column "sub1max")
Output: data/e3/settle/reval/r90knap.jsonl / r90knap.md (via r90knap.report)."""
import os, subprocess, sys, time, shutil
_h = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_h, "../../sub1"))
import r90knap as R

R.D = os.path.normpath(os.path.join(R.ROOT, "data/e3/settle/reval"))
R.MENUS = {"v3e": R.MENUS["v3e"]}
R.ARMS = {"base": None, "sub1": "sub1ratios-settle.json", "sub1max": "sub1ratios-old.json"}
R.BKS = [1 << 18]


def run(wd, par):
    nf = f"{wd}/v3/nf"
    if not os.path.exists(f"{nf}/.patched"):
        for d in ("v3/nf", "v3/e1b", "v3/e3/data", "v3/bench/data", "data/v3"):
            shutil.copytree(f"{R.ROOT}/{d}", f"{wd}/{d}", dirs_exist_ok=True)
        if "KN_SUB1" not in open(f"{nf}/knap4.py").read():  # merged into knap4.py since 5b617d9 (11-05)
            subprocess.run(["patch", "-p1", "-i", f"{R.ROOT}/v3/e3/sub1/knap4-sub1.patch"], cwd=wd, check=True)
        open(f"{nf}/.patched", "w").close()
    for a in R.ARMS.values():
        if a: shutil.copy(f"{R.D}/{a}", f"{wd}/{a}")
    os.makedirs(f"{wd}/out", exist_ok=True)
    M, A, F, P = "../../data/v3/kfit2-model-final.json", "../e1b/data/aq.jsonl", "../e1b/data/aqfactors.json", "../e1b/data/e0-ident.map"
    jobs = []
    for arm, rf in R.ARMS.items():
        for tg in R.EXTRA + R.TARGETS:
            n = f"v3e-{arm}-{tg:g}-b{R.BKS[0]}"; o = f"{wd}/out/{n}"
            if os.path.exists(f"{o}.jsonl") and os.path.getsize(f"{o}.jsonl"): continue
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KN_V3="1", KN_FRAC="1", KN_EMBT="2",
                       KN_HEADSLAB="1", KN_LEVELS="../../data/v3/levels", KN_DAMP="1", KN_MODES="f16",
                       KN_SPEED="../bench/data/speedcal-stock.jsonl", KN_LADDER=R.MENUS["v3e"], KN_WRITE=f"{o}.d",
                       KN_WRITE_CHECK="0", KN_BK=str(R.BKS[0]))
            if rf: env["KN_SUB1"] = f"{wd}/{rf}"
            jobs.append((n, ["nice", "-n", "10", "python3", "knap4.py", M, A, F, P, f"{o}.part", f"{tg}", "0"], env, o))
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
    print("R90KNAPX DONE", flush=True)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "run": run(os.path.abspath(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    elif c == "report": R.report(os.path.abspath(sys.argv[2]))
    else: print(__doc__); sys.exit(1)
