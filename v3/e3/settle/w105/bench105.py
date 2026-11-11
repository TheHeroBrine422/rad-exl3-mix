#!/usr/bin/env python3
"""bench105.py ARM --level 2.5-i8 --layers 8-12 --mode proc|persist -n N [--seed S] [--no-prefetch] [--warm]
-- w105 throughput arms on gpubox card 0 (task local/workers/instr3/w105.md, INDEX #143).

One arm = one level's jobs of the given layers (rxq.ladder.level_jobs, group split, specs exactly as w92's mkjobs92.py
writes them; gpubox inputs) run by rxrun.pool on card 0 under /root/card0.lock (+ ram.lock shared) for the whole
arm, N at a time, mode proc (one `python -m rxq job` process a job, w92's layer92.sh way) or persist (rxrun.worker).
Cold start by default: the jobs' input bytes (group Hessians, H_out files, weight ranges) are dropped from the page
cache (posix_fadvise DONTNEED) before the arm, as in a real level whose 256 jobs read ~100 GB of inputs on a 60 GB
host. While the arm runs, card 0's VRAM use and busy % are sampled (sysfs, 0.5 s).
Outputs: $ROOT/bench/ARM/<job>/ (job dirs), ARM/jobs.jsonl (pool records), ARM/summary.json; one line in
$ROOT/result.txt. --ref ARM0: the arm's t- / c9- / diag- files compared with ARM0's (rxrun.cmp: tensor bytes + metadata).
Rerun: a finished arm (summary.json) is skipped."""
import argparse, json, os, subprocess, sys, threading, time

from rxq import config as C
from rxrun import cmp, pool

ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("--level", default="2.5-i8"); ap.add_argument("--layers", default="8-12")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mode", choices=("proc", "persist"), required=True)
ap.add_argument("-n", type=int, default=1); ap.add_argument("--no-prefetch", action="store_true")
ap.add_argument("--warm", action="store_true", help="do not evict the inputs first")
ap.add_argument("--vram-gib", type=float, default=28.0); ap.add_argument("--ref")
ap.add_argument("--root", default=os.environ.get("ROOT", "/data/w105"))
a = ap.parse_args()
R = a.root; A = f"{R}/bench/{a.arm}"
if os.path.exists(f"{A}/summary.json"): print(f"{a.arm}: done"); sys.exit(0)
MODEL = "/data/workspace/models/qwen38-27b-bf16"; HESS = "/root/challenge/hessians/calib"; HOUT = "/data/swap27/hout-bn"
CODE = os.environ["CODE"]
jd = f"{R}/jobs/{a.level}-s{a.seed}-L{a.layers}"
if not os.path.exists(f"{jd}/jobs.jsonl"):
    subprocess.run([sys.executable, f"{CODE}/v3/e3/settle/w92/mkjobs92.py", a.level, str(a.seed), MODEL, HESS, HOUT, jd + ".part",
                    "--layers", a.layers], check=True)
    os.replace(jd + ".part", jd)
    # mkjobs92 writes the spec "config" as the absolute path of the tree it ran from: fine, the tree is $CODE
names = [json.loads(l)["name"] for l in open(f"{jd}/jobs.jsonl")]
cfg = C.load(json.load(open(f"{jd}/{names[0]}.json"))["config"])
jobs = [pool.make_job(n, f"{jd}/{n}.json", f"{A}/{n}", cfg=cfg) for n in names]
os.makedirs(A, exist_ok=True)
if not a.warm:
    for j in jobs: pool.evict(pool.input_ranges(j["_spec"]))

SYS = "/sys/class/drm/card0/device"                     # 0000:03:00.0 = card 0
samples = []; stop = threading.Event()


def sampler():
    while not stop.is_set():
        try: samples.append((time.time(), int(open(f"{SYS}/mem_info_vram_used").read()), int(open(f"{SYS}/gpu_busy_percent").read())))
        except OSError: pass
        stop.wait(0.5)


dev = {"name": "card0", "device": 0, "lock": "/root/card0.lock", "ram_lock": "/root/ram.lock", "workers": a.n,
       "vram_gib": a.vram_gib, "nice": 5, "env": {}}
first = []


def on_done(rec):
    print(f"{time.strftime('%H:%M:%S')} {rec['name']} {rec['wall']:.1f}s (concurrent {rec['concurrent']})", flush=True)


th = threading.Thread(target=sampler, daemon=True)
t_req = time.time()
# the sampler starts with the first job: run_jobs blocks on the lock first, so start it from a wrapper
orig_session = pool._session


def session(*x, **k):
    first.append(time.time()); th.start()
    return orig_session(*x, **k)


pool._session = session
res = pool.run_jobs(jobs, dev, A, mode=a.mode, prefetch=not a.no_prefetch, on_done=on_done, tag=a.arm)
stop.set(); th.join()
recs = [json.loads(l) for l in open(f"{A}/jobs.jsonl") if json.loads(l)["rc"] == 0]
t0 = min(r["t0"] for r in recs); t1 = max(r["t1"] for r in recs)
win = [s for s in samples if t0 <= s[0] <= t1]
quant = 0.0; stages = {}
for n in names:
    for l in open(f"{A}/{n}/log.jsonl"):
        r = json.loads(l)["rxq"]; quant += r["wall"]
        for k, v in (r["stages"] or {}).items(): stages[k] = stages.get(k, 0.0) + v
s = {"arm": a.arm, "level": a.level, "layers": a.layers, "seed": a.seed, "mode": a.mode, "n": a.n, "prefetch": not a.no_prefetch,
     "cold": not a.warm, "jobs": len(recs), "lock_wait_s": round(first[0] - t_req, 1), "wall_s": round(t1 - t0, 1),
     "jobs_per_h": round(len(recs) / (t1 - t0) * 3600, 1), "sum_job_wall_s": round(sum(r["wall"] for r in recs), 1),
     "sum_tensor_wall_s": round(quant, 1), "stages_s": {k: round(v, 1) for k, v in sorted(stages.items())},
     "vram_peak_gib": round(max(x[1] for x in win) / 2 ** 30, 2) if win else None,
     "gpu_busy_mean": round(sum(x[2] for x in win) / len(win), 1) if win else None,
     "gpu_idle_frac": round(sum(1 for x in win if x[2] < 5) / len(win), 3) if win else None,
     "t0_utc": time.strftime("%H:%M:%S", time.gmtime(t0))}
if a.ref:
    nf, diff, _ = cmp.compare(f"{R}/bench/{a.ref}", A, quiet=True)
    s["vs_ref"] = {"ref": a.ref, "files": nf, "differ": len(diff), "first": diff[:3]}
json.dump(s, open(f"{A}/summary.json.part", "w"), indent=1); os.replace(f"{A}/summary.json.part", f"{A}/summary.json")
json.dump(samples, open(f"{A}/samples.json", "w"))
line = (f"{time.strftime('%H:%M:%S', time.gmtime())} {a.arm}: {a.mode} N={a.n} {a.level} L{a.layers}: {len(recs)} jobs in "
        f"{s['wall_s']:.0f} s = {s['jobs_per_h']:.0f} jobs/h; tensor walls {s['sum_tensor_wall_s']:.0f} s; VRAM peak "
        f"{s['vram_peak_gib']} GiB; GPU busy {s['gpu_busy_mean']}% (idle {s['gpu_idle_frac']}); lock wait {s['lock_wait_s']:.0f} s"
        + (f"; vs {a.ref}: {s['vs_ref']['differ']} of {s['vs_ref']['files']} files differ" if a.ref else ""))
print(line); open(f"{R}/result.txt", "a").write(line + "\n")
