#!/usr/bin/env python3
"""r94bq.py JOBS [--max N] [--maxdown D] [--anoncap GB] -- w45's #94-rerun queue runner: w43's settleq.py with the
codebook as the job's first field (run from its directory, /data/work/w45, launched as
`python3 /data/work/w45/r94bq.py ...` so the watchdog's pattern matches).

JOBS lines "CB LAYER TENSOR K SEED ROWS" (r94bplan.py), run in order: settlex.py gain LAYER TENSOR K OUT ROWS with
SETTLE_CB = CB, SETTLE_M (default 0.9,1.0,1.1), E3_SEED = E3_ROWSEED = SEED, one thread, nice 10.
OUT = out/CB/sSEED/LAYER-TENSOR-KK.jsonl; a job is skipped when OUT
exists (settle.py writes it whole at the end) or another process already runs it. A job starts only while fewer than
--max of ours run (--maxdown for ffn_down) and the cgroup's anon memory, counting our running jobs at their estimated
peak (ffn_down 13 GiB, else 1.6 or 2.5 for ffn_gate_up; v1 measured 2026-11-06 + proxy v2's extra H copy), stays below --anoncap GiB with the new job; an
ffn_down job also needs room left for the free slots to fill with small jobs (keeps the CPUs busy when memory binds). A job that cannot start
does not block later ones. Failed jobs (rc != 0) are logged "FAIL" and not retried in this run (a rerun retries).
Touch logs/STOP: no new starts; exits after the running jobs ("R94B QUEUE STOPPED"). End: "R94B QUEUE DONE".
Log: logs/queue.log."""
import argparse, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("jobs"); ap.add_argument("--max", type=int, default=16); ap.add_argument("--maxdown", type=int, default=1)
ap.add_argument("--anoncap", type=float, default=110)
a = ap.parse_args()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("logs", exist_ok=True)
GB = 1 << 30
EST = {"ffn_down": 13, "ffn_gate_up": 2.5}; EST0 = 1.6
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "E3_CKPT": os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16"),
       "E3_HESS": os.environ.get("E3_HESS", "/data/challenge/hessians/calib")}


def log(m):
    with open("logs/queue.log", "a") as f: f.write(time.strftime("%H:%M:%S", time.gmtime()) + " " + m + "\n")


def anon():
    for l in open("/sys/fs/cgroup/memory.stat"):
        if l.startswith("anon "): return int(l.split()[1]) / GB
    return 0.0


def rss(pid):
    try: return int(open(f"/proc/{pid}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / GB
    except OSError: return 0.0


def ours_all():
    """(all, ffn_down) settlex.py jobs running from this directory, ours or a previous runner's"""
    n = d = 0; here = os.getcwd()
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            c = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
            if b"settlex.py" in c and os.readlink(f"/proc/{p}/cwd") == here: n += 1; d += b"ffn_down" in c
        except OSError: pass
    return n, d


def running_elsewhere(out):
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            if out.encode() in open(f"/proc/{p}/cmdline", "rb").read(): return True
        except OSError: pass
    return False


jobs = []
for line in open(a.jobs):
    f = line.split()
    if len(f) == 6:
        cb, layer, w, K, seed, rows = f
        jobs.append((cb, "gain", layer, w, K, seed, rows, f"out/{cb}/s{seed}/{layer}-{w}-K{K}.jsonl"))
pending = [j for j in jobs if not os.path.exists(j[7])]
run = {}                    # pid -> (job, Popen, est GiB, t0)
ok = fail = 0
log(f"start: {len(jobs)} jobs, {len(pending)} to do; max {a.max}, maxdown {a.maxdown}, anoncap {a.anoncap} GiB")
while pending or run:
    for pid, (j, p, e, t0) in list(run.items()):
        rc = p.poll()
        if rc is None: continue
        del run[pid]
        if rc == 0 and os.path.exists(j[7]): ok += 1; log(f"done {' '.join(j[:7])} {time.time() - t0:.0f}s")
        else: fail += 1; log(f"FAIL rc={rc} {' '.join(j[:7])} (logs/{os.path.basename(j[7])[:-6]}-{j[0]}-s{j[5]}.log)")
    if os.path.exists("logs/STOP"):
        pending = []
        if not run: log(f"R94B QUEUE STOPPED ok={ok} fail={fail}"); sys.exit(0)
    other = None                                          # memory beside our jobs: read once a pass (it jitters)
    for j in list(pending):
        n_all, d_all = ours_all()
        if n_all >= a.max: break
        if os.path.exists(j[7]): pending.remove(j); continue
        isd = j[3] == "ffn_down"
        if isd and d_all >= a.maxdown: continue
        e = EST.get(j[3], EST0)
        mine_est = sum(v[2] for v in run.values())
        if other is None: other = max(anon() - sum(rss(pid) for pid in run), 0)
        resv = max(0, a.max - n_all - 1) * EST0 if isd else 0
        if other + mine_est + e + resv > a.anoncap:
            if isd: continue                                  # a small job may still fit
            break
        if running_elsewhere(j[7]): pending.remove(j); continue
        os.makedirs(os.path.dirname(j[7]), exist_ok=True)
        lg = open(f"logs/{os.path.basename(j[7])[:-6]}-{j[0]}-s{j[5]}.log", "w")
        p = subprocess.Popen(["nice", "-n", "10", "python3", "-W", "ignore", "settlex.py", j[1], j[2], j[3], j[4], j[7], j[6]],
                             env={**ENV, "E3_SEED": j[5], "E3_ROWSEED": j[5], "SETTLE_CB": j[0],
                                  "SETTLE_M": os.environ.get("SETTLE_M", "0.9,1.0,1.1")}, stdout=lg, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        run[p.pid] = (j, p, e, time.time()); pending.remove(j)
        log(f"start {' '.join(j[:7])} (ours {len(run)}, anon {anon():.0f} GiB)")
        time.sleep(2)
    time.sleep(20)
log(f"R94B QUEUE DONE ok={ok} fail={fail}")
