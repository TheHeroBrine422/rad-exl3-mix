#!/usr/bin/env python3
"""q177.py JOBS [--max N] [--maxdown D] [--anoncap GB] -- w139's copy of w71's q71.py (#113) for #177 on Thor: settle177.py jobs
(the hash codebooks), job lines with a 7th field CB (W71_CB). Run as `python3 /data/work/w139/l177/q177.py ...` from anywhere (it chdirs to its own directory, the
watchdog's pattern matches the command line). The job file is re-read whenever it changes (append lines to queue
more work).

JOBS lines "LAYER TENSOR K SEED GROUP ARMS CB", run in file order: settle177.py LAYER TENSOR K OUT 128 with
W71_ARMS = ARMS, W71_CB = CB, E3_SEED = E3_ROWSEED = SEED, SETTLE_ORDER = rev, one thread, nice 10.
OUT = out/sSEED/LAYER-TENSOR-KK-GROUP.jsonl; a job is skipped when OUT exists (written whole at the end) or another
process already runs it (a killed job resumes at its next arm from OUT.arms). A job starts only while fewer than --max
of ours run (--maxdown for ffn_down) and the cgroup's anon memory, counting our running jobs at their estimated peak
(ffn_down 13 GiB, ffn_gate_up 2.5, else 1.6; a non-ffn_down job older than 4 min at 1.2x its measured RSS), stays
below --anoncap GiB with the new job. A job blocked by memory blocks the later ones in that pass; an ffn_down job
blocked by --maxdown does not. Failed jobs (rc != 0) are logged "FAIL" and not retried in this run (a rerun retries).
Touch logs/STOP: no new starts, exits after the running jobs ("Q177 QUEUE STOPPED"). End: "Q177 QUEUE DONE" (every
listed job done or failed and the file unchanged for 10 min). Log: logs/queue.log; status: logs/status (one line,
rewritten each pass)."""
import argparse, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("jobs"); ap.add_argument("--max", type=int, default=24); ap.add_argument("--maxdown", type=int, default=5)
ap.add_argument("--anoncap", type=float, default=100)
a = ap.parse_args()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("logs", exist_ok=True)
GB = 1 << 30
EST = {"ffn_down": 13, "ffn_gate_up": 2.5}; EST0 = 1.6
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "E3_CKPT": os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16"),
       "E3_HESS": os.environ.get("E3_HESS", "/data/challenge/hessians/calib"),
       "SETTLE_CACHE": os.path.abspath("cache"), "W71_SCALES": os.path.abspath("scales"),
       "W71_LIB": os.path.abspath("code/settle/w71/libbeamL.so"), "SETTLE_ORDER": "rev"}
SCRIPT = os.path.abspath("settle177.py")


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
    """(all, ffn_down) settle71.py jobs running, ours or a previous runner's"""
    n = d = 0
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            c = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
            if any(x.endswith(b"settle177.py") for x in c) and b"python3" in c[0]: n += 1; d += b"ffn_down" in c
        except OSError: pass
    return n, d


def running_elsewhere(out):
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            c = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
            if out.encode() in c and b"python3" in c[0]: return True
        except OSError: pass
    return False


seen, pending = set(), []
def load():
    n = 0
    for line in open(a.jobs):
        f = line.split()
        if len(f) != 7 or line.startswith("#"): continue
        layer, w, K, seed, grp, arms, cb = f
        j = (layer, w, K, seed, grp, arms, f"out/s{seed}/{layer}-{w}-K{K}-{grp}.jsonl", cb)
        if j[6] in seen: continue
        seen.add(j[6])
        if not os.path.exists(j[6]): pending.append(j); n += 1
    return n


mt = os.path.getmtime(a.jobs); load(); t_change = time.time()
run = {}
ok = fail = 0
log(f"start: {len(seen)} jobs, {len(pending)} to do; max {a.max}, maxdown {a.maxdown}, anoncap {a.anoncap} GiB")
while True:
    if os.path.getmtime(a.jobs) != mt:
        mt = os.path.getmtime(a.jobs); t_change = time.time(); n = load()
        log(f"job file changed: {n} new jobs, {len(pending)} pending")
    for pid, (j, p, e, t0) in list(run.items()):
        rc = p.poll()
        if rc is None: continue
        del run[pid]
        tag = f"{j[0]}-{j[1]}-K{j[2]}-s{j[3]}-{j[4]}"
        if rc == 0 and os.path.exists(j[6]): ok += 1; log(f"done {tag} {time.time() - t0:.0f}s")
        else: fail += 1; log(f"FAIL rc={rc} {tag} (logs/{tag}.log)")
    with open("logs/status.part", "w") as f:
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} ok={ok} fail={fail} running={len(run)} pending={len(pending)} of {len(seen)}\n")
    os.replace("logs/status.part", "logs/status")
    if os.path.exists("logs/STOP"):
        pending = []
        if not run: log(f"Q177 QUEUE STOPPED ok={ok} fail={fail}"); sys.exit(0)
    if not pending and not run and time.time() - t_change > 600: break
    other = None
    for j in list(pending):
        n_all, d_all = ours_all()
        if n_all >= a.max: break
        if os.path.exists(j[6]): pending.remove(j); continue
        isd = j[1] == "ffn_down"
        if isd and d_all >= a.maxdown: continue
        e = EST.get(j[1], EST0)
        mine_est = sum(v[2] if (v[0][1] == "ffn_down" or time.time() - v[3] < 240) else min(v[2], max(1.2 * rss(pid), 0.4))
                       for pid, v in run.items())
        if other is None: other = max(anon() - sum(rss(pid) for pid in run), 0)
        if other + mine_est + e > a.anoncap: break
        if running_elsewhere(j[6]): pending.remove(j); continue
        os.makedirs(os.path.dirname(j[6]), exist_ok=True)
        tag = f"{j[0]}-{j[1]}-K{j[2]}-s{j[3]}-{j[4]}"
        lg = open(f"logs/{tag}.log", "a")
        env = {**ENV, "E3_SEED": j[3], "E3_ROWSEED": j[3], "W71_ARMS": j[5], "W71_CB": j[7]}
        p = subprocess.Popen(["nice", "-n", "19", "python3", "-W", "ignore", SCRIPT, j[0], j[1], j[2], j[6], "128"],
                             env=env, stdout=lg, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        run[p.pid] = (j, p, e, time.time()); pending.remove(j)
        log(f"start {tag} {j[5]} (ours {len(run)}, anon {anon():.0f} GiB)")
        time.sleep(2)
    time.sleep(20)
log(f"Q177 QUEUE DONE ok={ok} fail={fail}")
