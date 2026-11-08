#!/usr/bin/env python3
"""q87.py JOBS [--max N] [--anoncap GB] [--mycap GB] [--maxbig N --big GB] -- w77 queue runner (via w79's), copied for
w87 (../w87.md). Run as `python3 BOXDIR/q87.py JOBS ...` (it chdirs to its own directory). w87 addition: --mycap, a
cap on the sum of the EST_GB of our running jobs (colo: <= 8 GB of ours in total; the cgroup-wide --anoncap check
stays). The
job file is re-read whenever it changes (append lines to queue more work).

JOBS lines "ID|DEPS|EST_GB|DONE|CMD": DEPS = comma list of files that must exist first ('-' none), EST_GB the job's
peak memory estimate, DONE the file whose existence means the job is done (jobs write it last), CMD a shell command
run from this directory with one thread, nice 19 (w79). Jobs start in file order while fewer than --max of ours run and the
cgroup's anon memory + the estimates of our jobs started in the last 4 min + EST_GB stays below --anoncap; a job whose
DEPS are missing, or that does not fit in memory, or a big one (EST_GB >= --big) while --maxbig big jobs run, is skipped
for now (later jobs may start; file order is priority). Jobs already running when the runner starts (a previous
runner's children: a process whose command line is exactly `bash -c CMD`, or has DONE as a whole argument, as
bash execs the last command of CMD) are adopted: counted as running, not
restarted; done when their DONE appears, requeued if they vanish without it. Failed jobs (rc != 0 or DONE missing) are logged "FAIL"
and not retried in this run (a rerun retries). Touch logs/STOP: no new starts, exit after the running jobs
("W87 QUEUE STOPPED"). End: "W87 QUEUE DONE" (every job done or failed and the file unchanged for 10 min).
Log: logs/queue.log; status: logs/status (one line, rewritten each pass); per-job logs logs/job-ID.log."""
import argparse, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("jobs"); ap.add_argument("--max", type=int, default=12); ap.add_argument("--anoncap", type=float, default=100)
ap.add_argument("--mycap", type=float, default=1e9)
ap.add_argument("--maxbig", type=int, default=0); ap.add_argument("--big", type=float, default=10)
a = ap.parse_args()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("logs", exist_ok=True)
GB = 1 << 30
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def log(m):
    with open("logs/queue.log", "a") as f: f.write(time.strftime("%H:%M:%S", time.gmtime()) + " " + m + "\n")


def anon():
    for l in open("/sys/fs/cgroup/memory.stat"):
        if l.startswith("anon "): return int(l.split()[1]) / GB
    return 0.0


seen, pending = set(), []
def load():
    n = 0
    for line in open(a.jobs):
        if line.startswith("#") or line.count("|") < 4: continue
        jid, deps, est, done, cmd = line.rstrip("\n").split("|", 4)
        if jid in seen: continue
        seen.add(jid)
        if not os.path.exists(done):
            pending.append((jid, [] if deps == "-" else deps.split(","), float(est), done, cmd)); n += 1
    return n


def cmdlines():
    c = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try: c.append(open(f"/proc/{d}/cmdline", "rb").read())
            except OSError: pass
    return c


def alive(j, cl):
    return (b"bash\0-c\0" + j[4].encode() + b"\0" in cl) or any(b"\0" + j[3].encode() + b"\0" in c for c in cl)


def adopt():
    """move pending jobs that a previous runner left running into ext"""
    cl = cmdlines()
    for j in list(pending):
        if alive(j, cl):
            pending.remove(j); ext[j[0]] = j; log(f"adopted running {j[0]}")


mt = os.path.getmtime(a.jobs); load(); t_change = time.time()
run = {}; ext = {}; ok = fail = 0
adopt()
log(f"start: {len(seen)} jobs, {len(pending)} to do; max {a.max}, anoncap {a.anoncap} GiB, mycap {a.mycap} GiB")
while True:
    if os.path.getmtime(a.jobs) != mt:
        mt = os.path.getmtime(a.jobs); t_change = time.time(); n = load()
        log(f"job file changed: {n} new jobs, {len(pending)} pending")
    if ext:
        cl = cmdlines()
        for jid, j in list(ext.items()):
            if os.path.exists(j[3]): del ext[jid]; ok += 1; log(f"done {jid} (adopted)")
            elif not alive(j, cl):
                del ext[jid]; pending.insert(0, j); log(f"adopted {jid} gone without DONE: requeued")
    for pid, (j, p, t0) in list(run.items()):
        rc = p.poll()
        if rc is None: continue
        del run[pid]
        if rc == 0 and os.path.exists(j[3]): ok += 1; log(f"done {j[0]} {time.time() - t0:.0f}s")
        else: fail += 1; log(f"FAIL rc={rc} {j[0]} (logs/job-{j[0]}.log)")
    with open("logs/status.part", "w") as f:
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} ok={ok} fail={fail} running={len(run) + len(ext)} pending={len(pending)} of {len(seen)}\n")
    os.replace("logs/status.part", "logs/status")
    if os.path.exists("logs/STOP"):
        pending = []
        if not run and not ext: log(f"W87 QUEUE STOPPED ok={ok} fail={fail}"); sys.exit(0)
    if not pending and not run and not ext and time.time() - t_change > 600: break
    for j in list(pending):
        if len(run) + len(ext) >= a.max: break
        if os.path.exists(j[3]): pending.remove(j); continue
        if not all(os.path.exists(d) for d in j[1]): continue
        if a.maxbig and j[2] >= a.big and sum(x[2] >= a.big for x in [v[0] for v in run.values()] + list(ext.values())) >= a.maxbig: continue
        if sum(v[0][2] for v in run.values()) + sum(x[2] for x in ext.values()) + j[2] > a.mycap: continue
        recent = sum(v[0][2] for v in run.values() if time.time() - v[2] < 240)
        if anon() + recent + j[2] > a.anoncap: continue
        lg = open(f"logs/job-{j[0]}.log", "a")
        p = subprocess.Popen(["nice", "-n", "19", "bash", "-c", j[4]], env=ENV, stdout=lg, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        run[p.pid] = (j, p, time.time()); pending.remove(j)
        log(f"start {j[0]} (ours {len(run)}, anon {anon():.0f} GiB)")
        time.sleep(2)
    time.sleep(20)
log(f"W87 QUEUE DONE ok={ok} fail={fail}")
