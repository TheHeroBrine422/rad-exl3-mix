# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""N rxq quant jobs at once on one card (w105).

run_jobs(jobs, dev, logdir, ...): the jobs not yet done (job.json + every tensor file present) run on the device in
sessions (jobs: a list, or a JobQueue that several devices' run_jobs threads share; a device takes only the jobs whose
modeled VRAM <= its max_job_vram): the card lock is taken (exclusive), then ram.lock (shared), as every gpubox GPU job does; up to
dev["workers"] jobs run at once (mode "persist": rxrun.worker processes, reused; mode "proc": one
`python -m rxq job SPEC -o OUT --device D` process a job, w92's way); the session ends when the jobs are done, after
`chunk` jobs (the locks are released so a waiting job can take the card, then taken again), or at the hold file
(no new job starts; the running ones finish). Per-card pause (user 11-12): the file HOLD-<dev name> next to the
hold file (e.g. ROOT/HOLD-c3) pauses only that card: its running jobs finish, it releases the card and ram locks (so
other work can take the card) and waits, taking no jobs; the other cards keep draining the shared queue; rm the file
and it rejoins. A device hold never ends the run by itself. Admission (each start): the modeled peak VRAM of the running jobs plus
this one (rxq.ladder.tensor_vram x 1.1 + 0.6 GiB a process) <= dev["vram_gib"], and host MemAvailable >= its RAM + the
guard. A job alone always starts. Host-RAM guard: MemAvailable < guard_mb while jobs run -> the newest job is killed
and requeued (as memguard.sh; its dir is cleaned before the rerun). A failed job stops the run (PoolError) unless it
ran out of memory beside others: it is requeued to run alone. Prefetch: a thread reads the next jobs' inputs (group
Hessian, H_out files, the weights' byte ranges) into the page cache, so the job's loads hit RAM; no code path of the
job changes.

Every finished job appends a line to LOGDIR/jobs.jsonl {name, slot, mode, n, t0, t1, wall, rc, tensors, ...}.
CPU-standin tests: dev["device"] = "cpu" and env RXQ_CPU_STANDIN=1."""
import fcntl, json, os, queue, signal, struct, subprocess, sys, threading, time

from rxq import config as C, ladder, qwen35
from .worker import clean

GIB = 2 ** 30


class PoolError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------- jobs
def job_done(spec, out):
    if not os.path.exists(f"{out}/job.json"): return False
    return all(os.path.exists(f"{out}/t-{k}.safetensors") for k in spec["keys"])


def make_job(name, spec_path, out, spec=None, cfg=None):
    """a pool job: {name, spec (path), out, keys, K, vram (GiB), ram (GiB), est (R9700 s)}"""
    spec = spec or json.load(open(spec_path))
    cfg = cfg or C.load(spec.get("config"))
    K = spec["K"]
    vram = max(ladder.tensor_vram(cfg, k, K) for k in spec["keys"]) * 1.1 + 0.6
    down = any(qwen35.family(k) == "ffn_down" for k in spec["keys"])
    req = ladder.job_req(cfg, spec["keys"], K)
    return {"name": name, "spec": spec_path, "out": out, "keys": list(spec["keys"]), "K": K, "vram": round(vram, 2),
            "ram": 5.0 if down else 3.0, "est": req["est_s"], "_spec": spec}


def mem_available_gib():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"): return int(l.split()[1]) / 2 ** 20
    return 0.0


# ---------------------------------------------------------------------------------------------- input ranges
_HDR = {}


def _st_header(path):
    if path not in _HDR:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; _HDR[path] = (8 + n, json.loads(f.read(n)))
    return _HDR[path]


def input_ranges(spec):
    """[(path, offset, length)] of a job's input bytes: the group input Hessian, H_out files, the weights' ranges"""
    inp = spec["inputs"]; out = []
    seen = set()
    for key in spec["keys"]:
        _, _, _, hname, _ = qwen35.locate(key)
        hp = f"{inp['hessians']}/{hname}.safetensors"
        if inp.get("hin") and os.path.exists(f"{inp['hin']}/{hname}.safetensors"): hp = f"{inp['hin']}/{hname}.safetensors"
        if hp not in seen: seen.add(hp); out.append((hp, 0, os.path.getsize(hp)))
        if inp.get("hout") and os.path.exists(f"{inp['hout']}/{key}.safetensors"):
            p = f"{inp['hout']}/{key}.safetensors"; out.append((p, 0, os.path.getsize(p)))
        ck = inp["ckpt"]; ix = f"{ck}/model.safetensors.index.json"
        f = json.load(open(ix))["weight_map"][key + ".weight"] if os.path.exists(ix) else "model.safetensors"
        base, hdr = _st_header(f"{ck}/{f}")
        a, b = hdr[key + ".weight"]["data_offsets"]
        out.append((f"{ck}/{f}", base + a, b - a))
    return out


def warm(ranges, block=16 << 20, stop=None):
    """read the ranges (page cache); returns bytes read"""
    n = 0
    for p, off, ln in ranges:
        fd = os.open(p, os.O_RDONLY)
        try:
            end = off + ln
            while off < end:
                if stop is not None and stop.is_set(): return n
                b = os.pread(fd, min(block, end - off), off)
                if not b: break
                off += len(b); n += len(b)
        finally: os.close(fd)
    return n


def evict(ranges):
    """drop the ranges from the page cache (benchmarks: every arm starts cold)"""
    for p, off, ln in ranges:
        fd = os.open(p, os.O_RDONLY)
        try: os.posix_fadvise(fd, off, ln, os.POSIX_FADV_DONTNEED)
        finally: os.close(fd)


class Prefetcher(threading.Thread):
    def __init__(self, min_free_gib=12.0):
        super().__init__(daemon=True)
        self.q = queue.Queue(); self.stop = threading.Event(); self.done = set(); self.min_free = min_free_gib
        self.bytes = 0

    def want(self, job):
        if job["name"] not in self.done: self.done.add(job["name"]); self.q.put(job)

    def run(self):
        while not self.stop.is_set():
            try: j = self.q.get(timeout=0.5)
            except queue.Empty: continue
            if mem_available_gib() < self.min_free: continue
            try: self.bytes += warm(input_ranges(j["_spec"]), stop=self.stop)
            except OSError: pass


# ----------------------------------------------------------------------------------------------------- locks
class Locks:
    def __init__(self, card, ram):
        self.paths = (card, ram); self.fds = []

    def __enter__(self):
        card, ram = self.paths
        if card:
            fd = os.open(card, os.O_RDWR | os.O_CREAT, 0o644); fcntl.flock(fd, fcntl.LOCK_EX); self.fds.append(fd)
        if ram:
            fd = os.open(ram, os.O_RDWR | os.O_CREAT, 0o644); fcntl.flock(fd, fcntl.LOCK_SH); self.fds.append(fd)
        return self

    def __exit__(self, *e):
        for fd in reversed(self.fds): fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
        self.fds = []


# ----------------------------------------------------------------------------------------------------- slots
class Slot:
    """one concurrent job place: a persistent worker (mode persist) or one process a job (mode proc)"""

    def __init__(self, i, dev, mode, events, logdir, py):
        self.i, self.dev, self.mode, self.ev, self.logdir, self.py = i, dev, mode, events, logdir, py
        self.proc = None; self.job = None; self.t0 = None; self.jobs_run = 0

    def env(self):
        e = dict(os.environ); e.update({k: str(v) for k, v in (self.dev.get("env") or {}).items()})
        return e

    def _pre(self):
        nice = self.dev.get("nice", 5)
        return lambda: (os.nice(nice), os.setsid())

    def start(self, job):
        self.job = job; self.t0 = time.time()
        os.makedirs(job["out"], exist_ok=True)
        if self.mode == "proc":
            clean(job["out"])
            lf = open(f"{job['out']}/rxq.log", "a")
            p = subprocess.Popen([self.py, "-m", "rxq", "job", job["spec"], "-o", job["out"], "--device", str(self.dev["device"])],
                                 stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=self.env(),
                                 preexec_fn=self._pre())
            lf.close(); self.proc = p
            threading.Thread(target=self._wait_proc, args=(p, job), daemon=True).start()
            return
        if self.proc is None or self.proc.poll() is not None:
            wl = open(f"{self.logdir}/worker-{self.dev['name']}-{self.i}.log", "a")
            self.proc = subprocess.Popen([self.py, "-m", "rxrun.worker", "--device", str(self.dev["device"])],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=wl, env=self.env(),
                                         preexec_fn=self._pre(), text=True, bufsize=1)
            wl.close()
            threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        self.proc.stdin.write(json.dumps({"id": job["name"], "spec": job["spec"], "out": job["out"]}) + "\n")
        self.proc.stdin.flush()

    def _wait_proc(self, p, job):
        rc = p.wait()
        oom = False
        if rc:
            try: oom = "out of memory" in open(f"{job['out']}/rxq.log").read()[-4000:].lower()
            except OSError: pass
        self.ev.put((self, p, {"id": job["name"], "rc": rc, "oom": oom, "err": None if not rc else f"rc {rc}"}))

    def _read(self, p):
        for line in p.stdout:
            try: r = json.loads(line)
            except ValueError: continue
            if "ready" in r: continue
            self.ev.put((self, p, r))
        rc = p.wait()
        self.ev.put((self, p, {"id": None, "rc": rc if rc else -1, "exit": True}))

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try: os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL); self.proc.wait()
        self.proc = None

    def quit(self):
        if self.mode == "persist" and self.proc is not None and self.proc.poll() is None:
            try: self.proc.stdin.write('{"quit": true}\n'); self.proc.stdin.flush(); self.proc.wait(30)
            except (BrokenPipeError, subprocess.TimeoutExpired): self.kill()
        self.proc = None


# ------------------------------------------------------------------------------------------------------- run
class JobQueue:
    """the jobs not yet done, shared by the devices that run them (one thread a device): take(pred) hands the first
    job pred accepts to one device only; put_front(job) gives it back (a killed / requeued job)"""

    def __init__(self, jobs):
        self.lock = threading.Lock()
        self.jobs = [j for j in jobs if not job_done(j["_spec"], j["out"])]
        names = [j["name"] for j in jobs]
        if len(set(names)) != len(names): raise PoolError("job names must be unique (they key the run's records)")

    def __len__(self):
        with self.lock: return len(self.jobs)

    def take(self, pred):
        with self.lock:
            for j in self.jobs:
                if pred(j): self.jobs.remove(j); return j
            return None

    def peek(self, n):
        with self.lock: return list(self.jobs[:n])

    def put_front(self, j):
        with self.lock: self.jobs.insert(0, j)


def fits(dev, j):
    """the job may run on this device at all (dev["max_job_vram"]: a smaller card takes only the jobs that fit)"""
    return j["vram"] <= float(dev.get("max_job_vram", dev.get("vram_gib", 28.0)))


def dev_hold(hold, dev):
    """the per-card pause file of dev (None without a hold file)"""
    return f"{hold}-{dev['name']}" if hold else None


def paused(hold, dev):
    h = dev_hold(hold, dev)
    return bool(h) and os.path.exists(h)


def run_jobs(jobs, dev, logdir, mode="persist", hold=None, chunk=None, py=None, prefetch=True, guard_gib=3.0,
             on_done=None, tag=""):
    """run the jobs not yet done on dev = {name, device, env, lock, ram_lock, workers, vram_gib, max_job_vram, nice};
    jobs: a list, or a JobQueue shared with other devices' run_jobs threads. Returns {"done": n, "held": bool, "wall": s,
    "jobs": n}. Raises PoolError on a failed job."""
    py = py or sys.executable
    os.makedirs(logdir, exist_ok=True)
    q = jobs if isinstance(jobs, JobQueue) else JobQueue(jobs)
    n = int(dev.get("workers", 1))
    res = {"done": 0, "held": False, "wall": 0.0, "jobs": len(q)}
    t_all = time.time()
    said = False
    while q.peek(1):
        if hold and os.path.exists(hold): res["held"] = True; break
        if not any(fits(dev, j) for j in q.peek(10 ** 9)): break          # the rest is for bigger cards
        if paused(hold, dev):                                           # this card only: no locks held, no jobs
            if not said: print(f"rxrun.pool: {dev['name']} paused ({dev_hold(hold, dev)})", flush=True); said = True
            time.sleep(dev.get("pause_poll_s", 5)); continue
        if said: print(f"rxrun.pool: {dev['name']} resumed", flush=True); said = False
        with Locks(dev.get("lock"), dev.get("ram_lock")):
            done = _session(q, dev, logdir, mode, n, hold, py, prefetch, guard_gib, on_done, tag, chunk)
        res["done"] += len(done)
        if hold and os.path.exists(hold): res["held"] = True; break
        if not done and q.peek(1): time.sleep(2)                          # the others' jobs: wait for them
        elif q.peek(1) and chunk: time.sleep(dev.get("yield_s", 5))       # let a waiting job take the card
    res["wall"] = round(time.time() - t_all, 1)
    return res


def _session(q, dev, logdir, mode, n, hold, py, prefetch, guard_gib, on_done, tag, chunk=None):
    ev = queue.Queue(); budget = float(dev.get("vram_gib", 28.0)); started = 0
    slots = [Slot(i, dev, mode, ev, logdir, py) for i in range(n)]
    pf = Prefetcher() if prefetch else None
    if pf: pf.start()
    done = set(); excl = set(); last_kill = 0.0
    jl = open(f"{logdir}/jobs.jsonl", "a")
    try:
        while True:
            stop_new = (hold and os.path.exists(hold)) or paused(hold, dev) or (chunk and started >= chunk)
            busy = [s for s in slots if s.job]
            if not busy and (stop_new or not any(fits(dev, j) for j in q.peek(10 ** 9))): break
            if not stop_new:
                for s in slots:
                    if s.job: continue
                    busy = [x for x in slots if x.job]
                    if any(x.job["name"] in excl for x in busy): break
                    j = _pick(q, dev, busy, budget, guard_gib, excl)
                    if j is None: break
                    s.start(j); started += 1
                    if pf:
                        for k in q.peek(max(1, n)): pf.want(k)
            try: s, p, r = ev.get(timeout=1.0)
            except queue.Empty:
                busy = [x for x in slots if x.job]
                if len(busy) > 1 and mem_available_gib() < guard_gib and time.time() - last_kill > 10:
                    v = max(busy, key=lambda x: x.t0)            # the newest job goes back to the queue
                    print(f"rxrun.pool: MemAvailable {mem_available_gib():.1f} GiB < {guard_gib} GiB: killed {v.job['name']}, requeued", flush=True)
                    q.put_front(v.job); v.job = None; v.kill(); last_kill = time.time()
                continue
            if s.job is None or p is not s.proc: continue        # a late event of a killed / replaced process
            if r.get("exit") and s.job is not None and r.get("id") is None:
                r = {"id": s.job["name"], "rc": r["rc"] or -1, "err": "worker exited", "oom": False}
            if r.get("id") != s.job["name"]: continue
            j = s.job; t1 = time.time()
            busy_now = sum(1 for x in slots if x.job)
            rec = {"name": j["name"], "tag": tag, "slot": s.i, "dev": dev["name"], "mode": mode, "n": n, "t0": round(s.t0, 3),
                   "t1": round(t1, 3), "wall": round(t1 - s.t0, 3), "rc": r["rc"], "tensors": r.get("tensors"),
                   "est": j["est"], "vram": j["vram"], "concurrent": busy_now}
            s.job = None
            if r["rc"] == 3:                                       # another engine setup: a fresh worker
                s.proc = None; q.put_front(j); continue
            if r["rc"] != 0:
                if mode == "persist": s.proc = None
                rec["err"] = r.get("err"); jl.write(json.dumps(rec) + "\n"); jl.flush()
                if r.get("oom") and n > 1 and j["name"] not in excl:
                    print(f"rxrun.pool: {j['name']} out of memory beside other jobs: requeued to run alone", flush=True)
                    excl.add(j["name"]); q.put_front(j); continue
                raise PoolError(f"job {j['name']} failed (rc {r['rc']}: {r.get('err')}); log {j['out']}/rxq.log")
            if not job_done(j["_spec"], j["out"]):
                raise PoolError(f"job {j['name']}: reply rc 0 but outputs missing in {j['out']}")
            jl.write(json.dumps(rec) + "\n"); jl.flush()
            done.add(j["name"])
            if on_done: on_done(rec)
    finally:
        for s in slots:
            if s.job: s.kill()
            else: s.quit()
        if pf: pf.stop.set()
        jl.close()
    return done


def _pick(q, dev, busy, budget, guard_gib, excl):
    if not busy: return q.take(lambda j: fits(dev, j))
    used = sum(x.job["vram"] for x in busy)
    free = mem_available_gib()
    blocked = []

    def ok(j):
        if blocked: return False
        if j["name"] in excl: blocked.append(j); return False      # an exclusive job waits for the card to empty
        return fits(dev, j) and used + j["vram"] <= budget and free >= j["ram"] + guard_gib
    return q.take(ok)
