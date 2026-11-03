#!/usr/bin/env python3
"""a1sched.py "CTRL_LAYERS" "MORE_LAYERS" NCPU [MARGIN_GB] -- #81 job scheduler for one machine (replaces a1ctl.sh's
fixed-width xargs stages, which left most CPUs idle while the memory-heavy down jobs ran).

Jobs (each `a1test.py` single-threaded, skipped when its output exists or it is already running):
  1. #81-02 control arms ugperm / uperm at K2.5 / K3 on CTRL_LAYERS (predictors already on disk);
  2. #81-03 for MORE_LAYERS at K2.5, then K2, then K3, then K4 and K1.5 on every layer (K-dependence): stage-1 parts (uq, 34 per layer and K, ~3 GB each) and, once a
     layer's 34 parts exist, its down arms D / ug / u (~16 GB each).
Admission every few seconds: running a1test processes < NCPU and the cgroup's anon memory + the job's estimate below
the cgroup limit - MARGIN. Down jobs first whenever eligible and they fit (they are the long pole), uq parts fill the
rest. Restart-safe; stop: touch /data/work/a1/STOP (running jobs finish). Log: /data/logs/a1sched.log."""
import glob, json, os, subprocess, sys, time
CTRL = [int(x) for x in sys.argv[1].split()]; MORE = [int(x) for x in sys.argv[2].split()]
NCPU = int(sys.argv[3]); MARGIN = float(sys.argv[4]) * 2**30 if len(sys.argv) > 4 else 10 * 2**30
A = "/data/work/a1"; W = "/data/work/e3t"; MEM = {"uq": 3.0 * 2**30, "down": 17 * 2**30}
ENV = dict(os.environ, E3_CKPT="/data/models/qwen38-27b-bf16", E3_HESS="/data/challenge/hessians/calib",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def done_rows():
    s = set()
    if os.path.exists(f"{A}/a1.jsonl"):
        for l in open(f"{A}/a1.jsonl"):
            try: r = json.loads(l); s.add((r["layer"], float(r["K"]), r["arm"]))
            except Exception: pass
    return s
def running():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if l.startswith("python3 -W ignore a1test.py")]
def anon():
    for l in open("/sys/fs/cgroup/memory.stat"):
        if l.startswith("anon "): return int(l.split()[1])
def limit(): return int(open("/sys/fs/cgroup/memory.max").read())
def gu(l, K, r): return f"{A}/gu/L{l}-K{K:g}/gu-{r:05d}.npy"
def jobs():
    d = done_rows()
    for l in CTRL:
        for K in (2.5, 3.0):
            for arm in ("ugperm", "uperm"):
                if (l, K, arm) not in d: yield "down", f"down {l} {K:g} {arm}"
    for K, LS in ((2.5, MORE), (2.0, MORE), (3.0, MORE), (4.0, CTRL + MORE), (1.5, CTRL + MORE)):
        for l in LS:
            parts = [r for r in range(0, 34816, 1024) if not os.path.exists(gu(l, K, r))]
            if parts:
                for r in parts: yield "uq", f"uq {l} {K:g} {r} {r + 1024}"
            else:
                for arm in ("D", "ug", "u"):
                    if (l, K, arm) not in d: yield "down", f"down {l} {K:g} {arm}"
def launch(kind, spec):
    a = spec.split(); fac = "1.0" if kind == "uq" else "1.0,1.15"
    tail = [f"{A}/gu"] + ([f"{A}/a1.jsonl"] if kind == "down" else [])
    lg = f"{A}/logs/{kind}-{'-'.join(a[1:4])}.log"
    subprocess.Popen(["nice", "-n", "10", "python3", "-W", "ignore", "a1test.py", *a, *tail],
                     cwd=W, env=dict(ENV, E3_FACTORS=fac), stdout=open(lg, "a"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
os.makedirs(f"{A}/logs", exist_ok=True)
log(f"start: ctrl {CTRL} more {MORE} ncpu {NCPU} margin {MARGIN / 2**30:.0f} GB limit {limit() / 2**30:.0f} GB")
pending_mem = []                     # (time, bytes): estimates of jobs launched in the last 90 s (not yet resident)
while not os.path.exists(f"{A}/STOP"):
    run = running(); now = time.time()
    pending_mem = [(t, b) for t, b in pending_mem if now - t < 90]
    free = limit() - MARGIN - anon() - sum(b for _, b in pending_mem)
    todo = [j for j in jobs() if not any(f"a1test.py {j[1]} " in r for r in run)]
    if not todo and not run: log("A1SCHED DONE"); break
    todo.sort(key=lambda j: j[0] != "down")          # down first when it fits
    started = 0
    for kind, spec in todo:
        if len(run) + started >= NCPU: break
        if MEM[kind] > free: continue
        launch(kind, spec); started += 1; free -= MEM[kind]; pending_mem.append((now, MEM[kind]))
        log(f"launch {spec} (running {len(run) + started}, free {free / 2**30:.0f} GB)")
    time.sleep(20 if started else 30)
