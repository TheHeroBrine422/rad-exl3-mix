#!/usr/bin/env python3
"""q64.py JOBS [--max N] [--maxdown D] [--anoncap GB] -- w64's queue runner on a CPU box (../w64.md): w46's
revalq.py running settle64.py (SETTLE_ORDER=rev, exllamav3's LDLQ order) instead of reval.py; otherwise unchanged.
The job file is re-read whenever it changes (append lines to queue more work). Run from its directory
(/data/work/w64), launched as `python3 /data/work/w64/q64.py ...` so the watchdog's pattern matches.

JOBS lines "ITEM MODE CB LAYER TENSOR K SEED M" (plan64.py), run in file order: settle64.py MODE LAYER TENSOR K OUT 128
with SETTLE_CB = CB, SETTLE_M = M (gain), E3_SEED = E3_ROWSEED = SEED, one thread, nice 10.
OUT = out/MODE/CB/sSEED/LAYER-TENSOR-KK.jsonl; a job is skipped when OUT exists (written whole at the end) or another
process already runs it. A job starts only while fewer than --max of ours run (--maxdown for ffn_down) and the
cgroup's anon memory, counting our running jobs at their estimated peak (ffn_down 13 GiB, ffn_gate_up 2.5, else 1.6;
a non-ffn_down job older than 4 min at 1.2x its measured RSS, at most its estimate),
stays below --anoncap GiB with the new job. A job blocked by memory blocks the later ones in that pass (ffn_down,
the long pole, gets the memory small jobs free; revised 1:25 PM CST: before, small jobs took every freed GiB);
an ffn_down job blocked by --maxdown does not. Failed jobs (rc != 0) are logged "FAIL" and not
retried in this run (a rerun retries). Touch logs/STOP: no new starts; exits after the running jobs
("W64 QUEUE STOPPED"). End: "W64 QUEUE DONE" (once every listed job is done or failed and the file is unchanged
for 10 min). Log: logs/queue.log."""
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
    """(all, ffn_down) settle64.py jobs running from this directory, ours or a previous runner's"""
    n = d = 0; here = os.getcwd()
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            c = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
            if b"settle64.py" in c and os.readlink(f"/proc/{p}/cwd") == here: n += 1; d += b"ffn_down" in c
        except OSError: pass
    return n, d


def running_elsewhere(out):
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            if out.encode() in open(f"/proc/{p}/cmdline", "rb").read(): return True
        except OSError: pass
    return False


seen, pending = set(), []
def load():
    """append the job file's new lines (not seen before, output not yet there) to pending; returns how many"""
    n = 0
    for line in open(a.jobs):
        f = line.split()
        if len(f) != 8: continue
        item, mode, cb, layer, w, K, seed, M = f
        j = (item, mode, cb, layer, w, K, seed, M, f"out/{mode}/{cb}/s{seed}/{layer}-{w}-K{K}.jsonl")
        if j[8] in seen: continue
        seen.add(j[8])
        if not os.path.exists(j[8]): pending.append(j); n += 1
    return n


mt = os.path.getmtime(a.jobs); load(); t_change = time.time()
run = {}                    # pid -> (job, Popen, est GiB, t0)
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
        if rc == 0 and os.path.exists(j[8]): ok += 1; log(f"done {' '.join(j[:8])} {time.time() - t0:.0f}s")
        else: fail += 1; log(f"FAIL rc={rc} {' '.join(j[:8])} (logs/{j[1]}-{j[2]}-s{j[6]}-{j[3]}-{j[4]}-K{j[5]}.log)")
    if os.path.exists("logs/STOP"):
        pending = []
        if not run: log(f"W64 QUEUE STOPPED ok={ok} fail={fail}"); sys.exit(0)
    if not pending and not run and time.time() - t_change > 600: break
    other = None                                          # memory beside our jobs: read once a pass (it jitters)
    for j in list(pending):
        n_all, d_all = ours_all()
        if n_all >= a.max: break
        if os.path.exists(j[8]): pending.remove(j); continue
        isd = j[4] == "ffn_down"
        if isd and d_all >= a.maxdown: continue
        e = EST.get(j[4], EST0)
        # our running jobs at their estimated peak; a non-ffn_down job past its first 4 min (prep and scale search
        # done, prep cached) at 1.5x its measured RSS instead (gate / up measured 1.1 GiB against the 2.5 estimate)
        mine_est = sum(v[2] if (v[0][4] == "ffn_down" or time.time() - v[3] < 240) else min(v[2], max(1.2 * rss(pid), 0.4))
                       for pid, v in run.items())
        if other is None: other = max(anon() - sum(rss(pid) for pid in run), 0)
        if other + mine_est + e > a.anoncap:
            # an ffn_down job (the long pole) blocked by memory alone: start nothing else this pass, so the memory
            # small jobs free accumulates until it fits (small jobs had taken every freed GiB: ffn_down starved)
            break
        if running_elsewhere(j[8]): pending.remove(j); continue
        os.makedirs(os.path.dirname(j[8]), exist_ok=True)
        lg = open(f"logs/{j[1]}-{j[2]}-s{j[6]}-{j[3]}-{j[4]}-K{j[5]}.log", "w")
        env = {**ENV, "E3_SEED": j[6], "E3_ROWSEED": j[6], "SETTLE_CB": j[2], "SETTLE_ORDER": "rev"}
        if j[7] != "-": env["SETTLE_M"] = j[7]
        p = subprocess.Popen(["nice", "-n", "10", "python3", "-W", "ignore", "settle64.py", j[1], j[3], j[4], j[5], j[8], "128"],
                             env=env, stdout=lg, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        run[p.pid] = (j, p, e, time.time()); pending.remove(j)
        log(f"start {' '.join(j[:8])} (ours {len(run)}, anon {anon():.0f} GiB)")
        time.sleep(2)
    time.sleep(20)
log(f"W64 QUEUE DONE ok={ok} fail={fail}")
