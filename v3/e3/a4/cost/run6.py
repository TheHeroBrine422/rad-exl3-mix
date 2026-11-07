#!/usr/bin/env python3
"""run6.py JOBS MAXPAR -- w72 copy of w58's run5.py running knap5a.py (v3/e3/a4/cost); otherwise unchanged. knap5 solves (restartable).
JOBS: one job a line, tab-separated: name, env (space-separated K=V), targets, lambdas. Each job writes
out/<name>.jsonl (via .part, renamed when complete; finished jobs are skipped on rerun), log logs/<name>.log.
A restarted runner adopts solves still running from a previous runner (pgrep on their .part path): it waits for them and
renames the .part if it holds a complete line. logs/MAXPAR (if present) overrides MAXPAR, re-read every round.
Runs from the tree's v3/nf (knap5's relative data paths). Prints 'FAIL <name> rc=<rc>' per failure and
'W72 QUEUE DONE' at the end. touch logs/STOP: no new starts."""
import json, os, subprocess, sys, time
W = os.path.dirname(os.path.abspath(__file__)); NF = f"{W}/tree/v3/nf"
M, A, F, P = "../../data/v3/kfit2-model-final.json", "../e1b/data/aq.jsonl", "../e1b/data/aqfactors.json", "../e1b/data/e0-ident.map"
jobs = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
maxp = int(sys.argv[2]); run = {}; q = [j for j in jobs if not os.path.exists(f"{W}/out/{j[0]}.jsonl")]
def live(n): return subprocess.run(["pgrep", "-f", f"out/{n}.part "], capture_output=True).returncode == 0
adopt = {j[0] for j in q if live(j[0])}; q = [j for j in q if j[0] not in adopt]
if adopt: print(f"{time.strftime('%H:%M:%S')} adopted {len(adopt)} running solves", flush=True)
print(f"{time.strftime('%H:%M:%S')} {len(q)} of {len(jobs)} jobs to run, {maxp} at a time", flush=True)
while q or run or adopt:
    try: maxp = int(open(f"{W}/logs/MAXPAR").read())
    except (OSError, ValueError): pass
    for n in [n for n in adopt if not live(n)]:
        adopt.discard(n); p_ = f"{W}/out/{n}.part"
        try: ok = bool(json.loads(open(p_).readline()))
        except Exception: ok = False
        if ok: os.rename(p_, f"{W}/out/{n}.jsonl"); print(f"{time.strftime('%H:%M:%S')} ok {n} (adopted)", flush=True)
        else: print(f"{time.strftime('%H:%M:%S')} FAIL {n} (adopted, no output)", flush=True)
    for n, (p, f) in list(run.items()):
        rc = p.poll()
        if rc is None: continue
        f.close(); del run[n]
        if rc == 0: os.rename(f"{W}/out/{n}.part", f"{W}/out/{n}.jsonl"); print(f"{time.strftime('%H:%M:%S')} ok {n}", flush=True)
        else: print(f"{time.strftime('%H:%M:%S')} FAIL {n} rc={rc}", flush=True)
    while q and len(run) + len(adopt) < maxp and not os.path.exists(f"{W}/logs/STOP"):
        n, ev, tg, lm = q.pop(0)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", KN_BK=str(1 << 18))
        env.update(kv.split("=", 1) for kv in ev.split())
        if os.path.exists(f"{W}/out/{n}.part"): os.remove(f"{W}/out/{n}.part")
        f = open(f"{W}/logs/{n}.log", "w")
        run[n] = (subprocess.Popen(["nice", "-n", "19", "python3", "knap5a.py", M, A, F, P, f"{W}/out/{n}.part", tg, lm],
                                   cwd=NF, env=env, stdout=f, stderr=subprocess.STDOUT), f)
    if os.path.exists(f"{W}/logs/STOP") and not run and not adopt: break
    time.sleep(5)
print(f"{time.strftime('%H:%M:%S')} W72 QUEUE DONE", flush=True)
