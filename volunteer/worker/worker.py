#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""worker.py --server URL --state DIR [--cache DIR --cache-gb G] [caps] [runtime paths] [--window HH:MM-HH:MM ...]

The donor side: lease -> fetch inputs (content-addressed cache with a disk cap) -> run the built-in job -> upload
(resumable) -> complete, with heartbeats throughout. Everything it does is in jobs/: the server names a job type and
sends parameters, which must match that type's schema; inputs must be the files the type allows for those parameters
and match their sha256; only the type's output files are uploaded, and only to the configured server.

Restart-safe: the current job and its phase live in STATE/current.json (written atomically); downloads go to
<blob>.part (resumed with HTTP ranges) and are renamed only after their sha256 matches; uploads resume from the
server's offset. A killed worker restarted with the same --state picks its job up again if its lease is still alive.
Stop: SIGINT / SIGTERM or the --stop-file once = finish the current job and exit (--on-stop abandon: give it back
now); twice = abandon now. Time windows: no new leases outside them."""
import argparse, fcntl, hashlib, json, os, re, shutil, signal, subprocess, sys, threading, time, traceback
import urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import API, read_json, write_json_atomic, sha256_file, NAME_RE, safe_relpath
import jobs

UA = "rxm-worker/0"


class APIError(Exception):
    def __init__(self, code, body): super().__init__(f"HTTP {code}: {body}"); self.code, self.body = code, body


class Client:
    def __init__(self, url, token=None, timeout=60):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def _req(self, method, path, body=None, headers=None, raw=False):
        h = {"User-Agent": UA, **(headers or {})}
        if self.token: h["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None and not isinstance(body, (bytes, bytearray)):
            data = json.dumps(body).encode(); h["Content-Type"] = "application/json"
        elif body is not None: data = bytes(body)
        r = urllib.request.Request(self.url + path, data=data, method=method, headers=h)
        try:
            resp = urllib.request.urlopen(r, timeout=self.timeout)
            return resp if raw else json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try: b = json.loads(e.read() or b"{}")
            except ValueError: b = {}
            raise APIError(e.code, b)

    def call(self, method, path, body=None, tries=None, stop=None, **kw):
        """retries transient failures (connection, 5xx) with backoff; 4xx are final"""
        d, n = 2.0, 0
        while True:
            try: return self._req(method, path, body, **kw)
            except APIError as e:
                if e.code < 500: raise
                err = e
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                err = e
            n += 1
            if (tries is not None and n >= tries) or (stop is not None and stop()): raise err
            log(f"server unreachable ({err!r}); retry in {d:.0f}s")
            time.sleep(d); d = min(d * 2, 120)

    def api(self, method, ep, body=None, **kw): return self.call(method, f"/api/{API}/{ep}", body, **kw)


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg, flush=True)


# ---------------------------------------------------------------- the input cache
class CacheFull(Exception): pass


class Cache:
    """blobs/<sha256> with a byte cap; least recently used blobs not needed by the current job are evicted"""
    def __init__(self, d, cap):
        self.d, self.cap = d, cap
        self.blobs = os.path.join(d, "blobs"); os.makedirs(self.blobs, exist_ok=True)
        self.ix_path = os.path.join(d, "index.json")
        self.ix = read_json(self.ix_path, {}) or {}
        on_disk = {f for f in os.listdir(self.blobs)}
        self.ix = {k: v for k, v in self.ix.items() if k in on_disk}       # forget what is gone
        for f in on_disk:                                                     # files nobody knows: drop
            if f not in self.ix and not f.endswith(".part"): os.remove(os.path.join(self.blobs, f))
        self.evicted = 0

    def used(self):
        return sum(os.path.getsize(os.path.join(self.blobs, f)) for f in os.listdir(self.blobs))

    def path(self, sha): return os.path.join(self.blobs, sha)

    def save(self): write_json_atomic(self.ix_path, self.ix)

    def make_room(self, need, pinned):
        if need > self.cap: raise CacheFull(f"job needs {need} bytes, cache cap {self.cap}")
        for f in os.listdir(self.blobs):                                      # stale partial downloads first
            if f.endswith(".part") and f[:-5] not in pinned: os.remove(os.path.join(self.blobs, f))
        while self.used() + need > self.cap:
            cand = sorted((v["last"], k) for k, v in self.ix.items() if k not in pinned)
            if not cand: raise CacheFull(f"cannot free {need} bytes: everything cached is needed now")
            _, k = cand[0]; os.remove(self.path(k)); del self.ix[k]; self.evicted += 1
            log(f"cache: evicted {k[:12]} ({self.evicted} so far)")
        self.save()

    def ensure(self, inputs, fetch):
        """inputs [{name, sha256, size}] -> {name: local path}; downloads what is missing. Returns (paths, bytes fetched)"""
        pinned = {i["sha256"] for i in inputs}
        missing = [i for i in inputs if not (os.path.exists(self.path(i["sha256"])) and
                                             os.path.getsize(self.path(i["sha256"])) == i["size"])]
        part = sum(os.path.getsize(self.path(i["sha256"]) + ".part") for i in missing if os.path.exists(self.path(i["sha256"]) + ".part"))
        self.make_room(sum(i["size"] for i in {i["sha256"]: i for i in missing}.values()) - part, pinned)
        got = 0
        for i in missing:
            if os.path.exists(self.path(i["sha256"])): continue               # same blob listed twice
            got += fetch(i, self.path(i["sha256"]) + ".part")
            if sha256_file(self.path(i["sha256"]) + ".part") != i["sha256"]:
                os.remove(self.path(i["sha256"]) + ".part")
                raise IOError(f"{i['name']}: sha256 mismatch after download")
            os.replace(self.path(i["sha256"]) + ".part", self.path(i["sha256"]))
        t = time.time()
        for i in inputs: self.ix[i["sha256"]] = {"size": i["size"], "last": t, "name": i["name"]}
        self.save()
        return {i["name"]: self.path(i["sha256"]) for i in inputs}, got


# ---------------------------------------------------------------- one job's runtime
class JobContext:
    def __init__(self, cfg, job, paths, out_dir, cancel):
        self.cfg, self.jtype, self.jversion, self.params = cfg, job["type"], job["version"], job["params"]
        self._paths, self.input_names = paths, list(paths)
        self.out_dir, self.stage_dir, self._cancel = out_dir, os.path.join(cfg.state, "stage"), cancel

    def input(self, name): return self._paths[name]
    def cancelled(self): return self._cancel.is_set()

    def run_cmd(self, argv, env=None, cwd=None, log=None):
        e = dict(os.environ); e.update(env or {})
        with open(log or os.devnull, "ab") as lf:
            p = subprocess.Popen(argv, env=e, cwd=cwd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            while p.poll() is None:
                if self._cancel.wait(0.5):
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(15)
                    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                    raise InterruptedError("cancelled")
            return p.returncode


class GpuLock:
    """flock on a shared lock file (e.g. a box's one-GPU-job-at-a-time lock), interruptible"""
    def __init__(self, path, cancel): self.path, self.cancel, self.fd = path, cancel, None
    def __enter__(self):
        if not self.path: return self
        self.fd = open(self.path, "a"); t0 = time.time(); said = False
        while True:
            try: fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if not said: log(f"waiting for {self.path}"); said = True
                if self.cancel.wait(2): raise InterruptedError("cancelled while waiting for the GPU lock")
        if said: log(f"got {self.path} after {time.time() - t0:.0f}s")
        return self
    def __exit__(self, *x):
        if self.fd: fcntl.flock(self.fd, fcntl.LOCK_UN); self.fd.close()


def in_windows(windows, t=None):
    if not windows: return True
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    for w in windows:
        a, b = [int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in w.split("-")]
        if (a <= m < b) if a <= b else (m >= a or m < b): return True
    return False


# ---------------------------------------------------------------- the worker
class Worker:
    def __init__(self, a):
        self.a = a
        a.state = os.path.abspath(a.state)
        os.makedirs(a.state, exist_ok=True)
        # what job modules read through ctx.cfg: this install's paths (never from the server)
        self.state, self.python, self.ourq, self.exl3_src = a.state, a.python, a.ourq, a.exl3_src
        self.device, self.image, self.gpu_name = a.device, a.image, a.gpu_name
        self.vram_cap_gb, self.capwrap = a.vram_cap_gb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "capwrap.py")
        self.cur_path = os.path.join(a.state, "current.json")
        self.stats_path = os.path.join(a.state, "worker.jsonl")
        tok = a.token or (open(a.token_file).read().strip() if a.token_file and os.path.exists(a.token_file) else None)
        self.c = Client(a.server, tok)
        if not tok:
            if not a.invite: sys.exit("no token: give --token / --token-file, or --invite CODE --name NAME to register")
            r = self.c.api("POST", "register", {"invite": a.invite, "name": a.name})
            tok = r["token"]; self.c.token = tok
            if a.token_file:
                fd = os.open(a.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f: f.write(tok + "\n")
            log(f"registered as worker {r['worker_id']}")
        self.cache = Cache(a.cache or os.path.join(a.state, "cache"), int(a.cache_gb * 1e9))
        self.stop, self.abandon = threading.Event(), threading.Event()
        self.cancel = threading.Event()
        impl = {n for n, m in jobs.REGISTRY.items() if m.IMPLEMENTED}
        self.types = sorted(impl & set(a.job_types.split(","))) if a.job_types else sorted(impl)
        self.caps = {"job_types": self.types, "versions": {n: jobs.REGISTRY[n].VERSION for n in self.types},
                     "gpu": bool(a.gpu_name), "gpu_name": a.gpu_name or "", "vram_gb": a.vram_gb, "ram_gb": a.ram_gb,
                     "disk_gb": a.cache_gb, "image": a.image}
        self.env = dict(kv.split("=", 1) for kv in a.env)
        self.jobs_done = 0

    # signals / stop file
    def on_signal(self, *_):
        if self.stop.is_set() or self.a.on_stop == "abandon":
            log("stop: abandoning the current job"); self.abandon.set(); self.cancel.set()
        else: log("stop: finishing the current job, then exiting (again to abandon it)")
        self.stop.set()

    def check_stop_file(self):
        if self.a.stop_file and os.path.exists(self.a.stop_file) and not self.stop.is_set():
            log("stop file seen"); self.on_signal()

    def record(self, **kw):
        with open(self.stats_path, "a") as f: f.write(json.dumps({"t": time.time(), **kw}) + "\n")

    # ------------------------------------------------------------ main loop
    def loop(self):
        cur = read_json(self.cur_path)
        if cur: self.resume(cur)
        idle = 0
        while not self.stop.is_set():
            self.check_stop_file()
            if self.stop.is_set(): break
            if self.a.max_jobs and self.jobs_done >= self.a.max_jobs: break
            if not in_windows(self.a.window):
                self.stop.wait(30); continue
            try: r = self.c.api("POST", "lease", {"caps": self.caps}, stop=self.stop.is_set)
            except APIError as e:
                log(f"lease refused: {e}")
                if e.code == 401: sys.exit("token rejected by the server")
                self.stop.wait(30); continue
            except Exception as e:
                if self.stop.is_set(): break
                raise
            if not r.get("job"):
                idle += 1
                if self.a.idle_exit and idle >= self.a.idle_exit: log("idle: exiting"); break
                self.stop.wait(min(r.get("retry_s", 30), self.a.poll)); continue
            idle = 0
            self.do_job({"job": r["job"], "phase": "fetch", "t_lease": time.time()})
        log("worker stopped")

    def resume(self, cur):
        j = cur["job"]
        try: hb = self.c.api("POST", "heartbeat", {"job_id": j["id"], "lease_id": j["lease_id"]}, tries=5)
        except Exception as e: hb = {"ok": False, "cancel": True, "error": repr(e)}
        if hb.get("cancel"):
            log(f"job {j['id']}: lease gone after restart ({hb}); dropping it")
            self.cleanup(cur); return
        log(f"job {j['id']}: resuming at phase {cur['phase']}")
        self.do_job(cur, resumed=True)

    def heartbeat_loop(self, job, done):
        while not done.wait(job.get("heartbeat_s", 30)):
            self.check_stop_file()
            try:
                hb = self.c.api("POST", "heartbeat", {"job_id": job["id"], "lease_id": job["lease_id"]}, tries=3)
                if hb.get("cancel"):
                    log(f"job {job['id']}: server says stop ({hb.get('state')}, job {hb.get('job_state')})"); self.cancel.set()
            except APIError as e:
                if e.code in (401, 403): self.cancel.set()
            except Exception as e:
                log(f"heartbeat failed: {e!r}")

    def save(self, cur): write_json_atomic(self.cur_path, cur)

    def dirs(self, job):
        d = os.path.join(self.a.state, "jobs", f"{job['id']}-{job['lease_id'][:8]}")
        return d, os.path.join(d, "out")

    def cleanup(self, cur):
        d, _ = self.dirs(cur["job"])
        shutil.rmtree(d, ignore_errors=True)
        if os.path.exists(self.cur_path): os.remove(self.cur_path)

    def fail(self, job, err, **kw):
        try: self.c.api("POST", "fail", {"job_id": job["id"], "lease_id": job["lease_id"], "error": err[-1500:], **kw}, tries=5)
        except Exception as e: log(f"could not report the failure: {e!r}")

    def do_job(self, cur, resumed=False):
        job = cur["job"]; self.cancel.clear()
        done = threading.Event()
        hb = threading.Thread(target=self.heartbeat_loop, args=(job, done), daemon=True); hb.start()
        st = {"job": job["id"], "type": job["type"], "resumed": resumed}
        try:
            # 1. is this a job we run, with parameters and inputs it allows? (no code from the server, ever)
            try:
                m = jobs.check_params(job["type"], job["params"])
                if not m.IMPLEMENTED: raise jobs.Rejected(f"{job['type']} is not implemented in this worker")
                if job["type"] not in self.types: raise jobs.Rejected(f"{job['type']} not enabled here")
                for i in job["inputs"]:
                    if not (safe_relpath(i.get("name")) and isinstance(i.get("size"), int) and i["size"] >= 0 and
                            isinstance(i.get("sha256"), str) and len(i["sha256"]) == 64 and all(c in "0123456789abcdef" for c in i["sha256"])):
                        raise jobs.Rejected(f"bad input entry {str(i)[:100]}")
                jobs.check_inputs(job["type"], job["params"], [i["name"] for i in job["inputs"]])
            except jobs.Rejected as e:
                log(f"job {job['id']}: REJECTED: {e}")
                self.fail(job, f"rejected by worker: {e}", reject=True); self.cleanup(cur); return
            jd, out = self.dirs(job); os.makedirs(out, exist_ok=True)
            # 2. fetch
            t0 = time.time()
            paths, got = self.cache.ensure(job["inputs"], self.fetch)
            st.update(fetch_s=round(time.time() - t0, 2), down_bytes=got, input_bytes=sum(i["size"] for i in job["inputs"]),
                      evicted=self.cache.evicted)
            if self.cancel.is_set(): raise InterruptedError("cancelled")
            # 3. run
            if cur["phase"] in ("fetch", "run"):
                cur["phase"] = "run"; self.save(cur)
                ctx = JobContext(self, job, paths, out, self.cancel)
                t0 = time.time()
                with GpuLock(self.a.gpu_lock if m.requires(job["params"]).get("gpu") else None, self.cancel):
                    t1 = time.time()
                    result = m.run(job["params"], ctx)
                st.update(lock_wait_s=round(t1 - t0, 2), run_s=round(time.time() - t1, 2))
                cur.update(phase="upload", result=result); self.save(cur)
            # 4. upload + complete
            t0 = time.time()
            up = self.upload_and_complete(job, out, cur.get("result", {}), m)
            st.update(upload_s=round(time.time() - t0, 2), **up)
            st["total_s"] = round(time.time() - cur.get("t_lease", time.time()), 2)
            self.record(**st); self.jobs_done += 1
            log(f"job {job['id']} done: {json.dumps(st)}")
            self.cleanup(cur)
        except InterruptedError:
            if self.abandon.is_set():
                log(f"job {job['id']}: abandoned"); self.fail(job, "abandoned by the worker", abandon=True)
            else: log(f"job {job['id']}: cancelled by the server")
            self.cleanup(cur)
        except CacheFull as e:
            log(f"job {job['id']}: {e}"); self.fail(job, f"cache: {e}", abandon=True); self.cleanup(cur)
        except Exception as e:
            tb = traceback.format_exc(); log(f"job {job['id']} FAILED: {e!r}")
            self.fail(job, tb); self.cleanup(cur)
        finally:
            done.set()

    def fetch(self, inp, part):
        """one input into part, resuming from its size with a Range request; returns bytes transferred"""
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > inp["size"]: os.remove(part); have = 0
        if have == inp["size"]: return 0
        r = self.c.call("GET", "/data/" + inp["name"], headers={"Range": f"bytes={have}-"}, raw=True, tries=5)
        if r.status == 200 and have: have = 0; mode = "wb"                    # server ignored the range
        else: mode = "ab"
        n = 0
        with open(part, mode) as f:
            while True:
                b = r.read(1 << 20)
                if not b: break
                f.write(b); n += len(b)
                if self.cancel.is_set(): raise InterruptedError("cancelled")
        return n

    def upload_and_complete(self, job, out, result, m):
        outs = m.outputs(job["params"]); req = m.required_outputs(job["params"])
        names = sorted(n for n in outs if os.path.exists(os.path.join(out, n)))
        if req - set(names): raise RuntimeError(f"job produced no {sorted(req - set(names))}")
        files = {}
        for n in names:
            p = os.path.join(out, n); sz = os.path.getsize(p)
            if sz > outs[n]: raise RuntimeError(f"{n}: {sz} bytes over the cap {outs[n]}")
            files[n] = {"sha256": sha256_file(p), "size": sz}
        sent = 0
        for attempt in range(3):
            for n in names:
                sent += self.upload(job, n, os.path.join(out, n), files[n]["size"])
            try:
                r = self.c.api("POST", "complete", {"job_id": job["id"], "lease_id": job["lease_id"], "files": files, "result": result})
                return {"up_bytes": sent, "duplicate": r.get("duplicate", False)}
            except APIError as e:
                if e.code == 422 and e.body.get("bad"): log(f"server rejected {e.body['bad']} (sha / size): re-uploading"); continue
                raise
        raise RuntimeError("upload kept failing the sha256 check")

    def upload(self, job, name, path, size):
        ep = f"upload/{job['id']}/{job['lease_id']}/{name}"
        try: off = self.c.api("GET", ep, tries=5)["size"]
        except APIError as e:
            if e.code == 409 and e.body.get("accepted"): return 0
            raise
        if off > size: off = None                                              # server holds junk: it drops it at complete
        if off is None or off == size: return 0
        sent, chunk = 0, self.a.chunk_mb << 20
        with open(path, "rb") as f:
            f.seek(off)
            while off < size:
                b = f.read(min(chunk, size - off))
                try: r = self.c.api("PUT", f"{ep}?offset={off}", b, tries=5)
                except APIError as e:
                    if e.code == 409 and "size" in e.body: off = e.body["size"]; f.seek(off); continue
                    if e.code == 409 and e.body.get("accepted"): return sent
                    raise
                off = r["size"]; sent += len(b)
                if self.cancel.is_set() and self.abandon.is_set(): raise InterruptedError("cancelled")
        return sent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    e = lambda k, d=None: os.environ.get("RXMW_" + k) or d       # the image's runtime config (a flag wins)
    ap.add_argument("--server", default=e("SERVER"), help="queue URL ($RXMW_SERVER)")
    ap.add_argument("--token", default=e("TOKEN"), help="worker token ($RXMW_TOKEN; prefer --token-file)")
    ap.add_argument("--token-file", default=e("TOKEN_FILE"), help="token file ($RXMW_TOKEN_FILE); written at registration")
    ap.add_argument("--invite", default=e("INVITE"), help="one-time invite code to register ($RXMW_INVITE)")
    ap.add_argument("--name", default=e("NAME", "worker"), help="worker name at registration ($RXMW_NAME)")
    ap.add_argument("--state", default=e("STATE"), help="state dir (current job, jobs/, stage/, worker.jsonl)")
    ap.add_argument("--cache", help="input cache dir (default STATE/cache)"); ap.add_argument("--cache-gb", type=float, default=float(e("CACHE_GB", 200)))
    ap.add_argument("--job-types", default=e("JOB_TYPES"), help="comma list (default: every implemented type)")
    ap.add_argument("--gpu-name", default=e("GPU_NAME", ""), help="declare a GPU (empty: CPU-only worker, no GPU jobs)")
    ap.add_argument("--vram-gb", type=float, default=float(e("VRAM_GB", 0))); ap.add_argument("--ram-gb", type=float, default=float(e("RAM_GB", 0)))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--vram-cap-gb", type=float, default=float(e("VRAM_CAP_GB", 0)), help="cap torch's allocator in job processes (shared card)")
    ap.add_argument("--gpu-lock", help="flock this file around every GPU job")
    ap.add_argument("--python", default=sys.executable); ap.add_argument("--ourq", help="quantize/ourq.py")
    ap.add_argument("--exl3-src", help="the fork's exllamav3_ext sources (ourq --frac-ext)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="extra environment for job processes")
    ap.add_argument("--image", default=e("IMAGE", "local"), help="image / install id reported with results ($RXMW_IMAGE)")
    ap.add_argument("--window", action="append", default=[], metavar="HH:MM-HH:MM",
                    help="local-time windows for new jobs (or $RXMW_WINDOWS, comma-separated; TZ sets the zone)")
    ap.add_argument("--on-stop", choices=("finish", "abandon"), default=e("ON_STOP", "finish")); ap.add_argument("--stop-file")
    ap.add_argument("--poll", type=float, default=30); ap.add_argument("--idle-exit", type=int, default=0)
    ap.add_argument("--max-jobs", type=int, default=0); ap.add_argument("--chunk-mb", type=int, default=16)
    a = ap.parse_args()
    if not a.server or not a.state: ap.error("--server and --state are required (or $RXMW_SERVER / $RXMW_STATE)")
    if not a.window: a.window = [x for x in (e("WINDOWS", "")).replace(" ", ",").split(",") if x]
    for w_ in a.window:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d", w_): ap.error(f"bad window {w_!r}: HH:MM-HH:MM")
    w = Worker(a)
    signal.signal(signal.SIGINT, w.on_signal); signal.signal(signal.SIGTERM, w.on_signal)
    log(f"worker up: types {w.types}, cache {a.cache_gb} GB, windows {a.window or 'always'}")
    w.loop()


if __name__ == "__main__":
    main()
