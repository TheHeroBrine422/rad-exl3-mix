#!/usr/bin/env python3
"""q82.py JOBS [--max N] [--memcap GB] -- w82's restartable queue runner on a CPU box (../w82.md). Run from anywhere as
`python3 /data/work/w82/q82.py JOBS ...` (it chdirs to its own directory).

JOBS lines "MODE CB LAYER TENSOR K SAMPLE" (plan82.py), in file order: settle82.py MODE LAYER TENSOR K OUT 128 with
SETTLE_CB = CB, E3_SEED = E3_ROWSEED = SAMPLE, SETTLE_ORDER=rev, W82_ORDER=order82.json, one thread, nice 19.
OUT = out/MODE/CB/sSAMPLE/LAYER-TENSOR-KK.jsonl, written whole at the end: a job is skipped when OUT exists or
another process runs it, so a rerun resumes. Memory: a job starts only while the RSS of our running jobs (each young
job, < 4 min, counted at max(RSS, its estimate)) plus the new job's estimate stays below --memcap GiB.
Touch logs/STOP: no new starts, exits after the running jobs. Log logs/queue.log; end "W82 QUEUE DONE ok= fail=" ."""
import argparse, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("jobs"); ap.add_argument("--max", type=int, default=6); ap.add_argument("--memcap", type=float, default=7.5)
a = ap.parse_args()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("logs", exist_ok=True)
EST = {"ffn_gate": 2.6, "ffn_up": 2.6, "ssm_qkv": 1.8, "attn_qg": 2.0, "ssm_z": 1.4}
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "E3_CKPT": os.environ.get("E3_CKPT", "/data/models/qwen38-27b-bf16"),
       "E3_HESS": os.environ.get("E3_HESS", "/data/challenge/hessians/calib"),
       "SETTLE_ORDER": "rev", "W82_ORDER": os.path.abspath("order82.json"), "SETTLE_CACHE": "cache"}


def log(m):
    with open("logs/queue.log", "a") as f: f.write(time.strftime("%H:%M:%S", time.gmtime()) + " " + m + "\n")


def rss(pid):
    try: return int(open(f"/proc/{pid}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except OSError: return 0.0


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
    if len(f) != 6: continue
    mode, cb, layer, w, K, s = f
    jobs.append((mode, cb, layer, w, K, s, f"out/{mode}/{cb}/s{s}/{layer}-{w}-K{K}.jsonl"))
pending = [j for j in jobs if not os.path.exists(j[6])]
run = {}; ok = fail = 0
log(f"start: {len(jobs)} jobs, {len(pending)} to do; max {a.max}, memcap {a.memcap} GiB")
while pending or run:
    for pid, (j, p, t0) in list(run.items()):
        rc = p.poll()
        if rc is None: continue
        del run[pid]
        if rc == 0 and os.path.exists(j[6]): ok += 1; log(f"done {' '.join(j[:6])} {time.time() - t0:.0f}s")
        else: fail += 1; log(f"FAIL rc={rc} {' '.join(j[:6])}")
    if os.path.exists("logs/STOP"): pending = []
    while pending and len(run) < a.max:
        j = pending[0]
        if os.path.exists(j[6]) or running_elsewhere(j[6]): pending.pop(0); continue
        used = sum(max(rss(pid), EST.get(v[0][3], 1.2)) if time.time() - v[2] < 240 else rss(pid) for pid, v in run.items())
        if used + EST.get(j[3], 1.2) > a.memcap: break
        pending.pop(0)
        os.makedirs(os.path.dirname(j[6]), exist_ok=True)
        lg = open(f"logs/{j[0]}-{j[1]}-s{j[5]}-{j[2]}-{j[3]}-K{j[4]}.log", "w")
        env = {**ENV, "E3_SEED": j[5], "E3_ROWSEED": j[5], "SETTLE_CB": j[1]}
        p = subprocess.Popen(["nice", "-n", "19", "python3", "-W", "ignore", "settle82.py", j[0], j[2], j[3], j[4], j[6], "128"],
                             env=env, stdout=lg, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        run[p.pid] = (j, p, time.time())
        log(f"start {' '.join(j[:6])} (ours {len(run)}, est used {used:.1f} GiB)")
        time.sleep(3)
    time.sleep(15)
log(f"W82 QUEUE {'STOPPED' if os.path.exists('logs/STOP') else 'DONE'} ok={ok} fail={fail}")
