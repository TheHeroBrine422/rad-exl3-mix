#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""e2e_noop.py [WORKDIR] -- end-to-end tests of server + worker with noop jobs (localhost, ~2 min, a few MB).

E2E_SERVER=local (default) runs server.py as a subprocess; E2E_SERVER=docker runs the server container from
docker/server/compose.yaml (image $E2E_IMAGE, default rxm-queue:<VERSION>; project rxmq-e2e) with WORKDIR's db/, data/,
out/ and secrets/ bind-mounted, while the workers, admin.py and the test itself run on the host against the same files.
Both modes also restart the server mid-run three ways (graceful restart, SIGKILL, remove + recreate) and check that
every job is accepted exactly once with the right bytes.

Scenarios (TESTS.md): tokens / invites, 3 workers with one killed mid-job (lease expires, job reassigned, accepted
once), a killed worker restarted before / after its lease expired, duplicate completion, corrupted upload, upload
caps, disk-cap eviction and the disk-based lease filter, Range resume of a partial download, data access rules,
malformed jobs from the real server (known type, bad params) and from a fake malicious server (unknown type, extra
keys, path traversal, inputs the type does not allow, an unimplemented type), graceful stop (finish / abandon),
time windows."""
import collections, glob, hashlib, json, os, shutil, signal, sqlite3, subprocess, sys, threading, time
import urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from worker.worker import in_windows
from common.util import sha256_file

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/tmp/w13-e2e")
PORT = int(os.environ.get("E2E_PORT", "18751"))
URL = f"http://127.0.0.1:{PORT}"
PY = sys.executable
RESULTS = []
MODE = os.environ.get("E2E_SERVER", "local")
DBP, DATA, RES, SPOOL = f"{W}/db/queue.sqlite", f"{W}/data", f"{W}/out/results", f"{W}/out/spool"
ADMIN_TOKEN = "e2e-admin-" + hashlib.sha256(W.encode()).hexdigest()[:40]


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def admin(*args):
    r = subprocess.run([PY, "-I", f"{HERE}/server/admin.py", "--db", DBP, "--data", DATA, "--results", RES, *args],
                       capture_output=True, text=True)
    if r.returncode: print(r.stdout, r.stderr)
    return r.stdout


def db():
    c = sqlite3.connect(DBP, timeout=30); c.row_factory = sqlite3.Row; return c


def req(method, path, body=None, token=None, headers=None):
    h = dict(headers or {})
    if token: h["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    r = urllib.request.Request(URL + path, data=data, method=method, headers=h)
    try:
        resp = urllib.request.urlopen(r, timeout=30); b = resp.read()
        return resp.status, (json.loads(b) if b and resp.headers.get("Content-Type") == "application/json" else b)
    except urllib.error.HTTPError as e:
        b = e.read(); return e.code, (json.loads(b) if b else {})
    except (urllib.error.URLError, OSError) as e:          # incl. resets while a container's port proxy starts
        return None, {"error": repr(e)}


def worker(name, token, *extra, log=None):
    lf = open(f"{W}/{name}.log", "a")
    return subprocess.Popen([PY, "-I", f"{HERE}/worker/worker.py", "--server", URL, "--token", token, "--state", f"{W}/{name}",
                             "--poll", "0.5", *extra], stdout=lf, stderr=subprocess.STDOUT)


def wait(cond, timeout, step=0.2):
    t = time.time() + timeout
    while time.time() < t:
        if cond(): return True
        time.sleep(step)
    return cond()


def states():
    return {r["id"]: r["state"] for r in db().execute("SELECT id, state FROM jobs")}


def events(job=None, kind=None):
    q, a = "SELECT * FROM events WHERE 1", []
    if job is not None: q += " AND job_id=?"; a.append(job)
    if kind: q += " AND kind=?"; a.append(kind)
    return db().execute(q, a).fetchall()


def jobs_of(tag):
    return [r["id"] for r in db().execute("SELECT id FROM jobs WHERE level=? ORDER BY id", (tag,))]


def stats(name):
    p = f"{W}/{name}/worker.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ---------------------------------------------------------------- fake malicious server
class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    queue, fails, data_hits = [], [], []
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path.endswith("/lease"): self._send({"job": Fake.queue.pop(0) if Fake.queue else None, "retry_s": 0.2})
        elif self.path.endswith("/fail"): Fake.fails.append(b); self._send({"ok": True})
        else: self._send({"ok": True, "cancel": False})
    def do_GET(self):
        Fake.data_hits.append(self.path); self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()


# ---------------------------------------------------------------- the server under test: a subprocess or the container
class LocalServer:
    def start(self):
        self.p = subprocess.Popen([PY, "-I", f"{HERE}/server/server.py", "--db", DBP, "--data", DATA, "--spool", SPOOL,
                                   "--results", RES, "--port", str(PORT), "--lease-s", "6", "--heartbeat-s", "1.5", "--retry-s", "0.5",
                                   "--upload-max-mb", "48", "--admin-token-file", f"{W}/secrets/admin-token"],
                                  stdout=open(f"{W}/server.log", "a"), stderr=subprocess.STDOUT)
    def stop(self):
        if self.p.poll() is None: self.p.send_signal(signal.SIGTERM); self.p.wait(30)
        return self.p.returncode
    def restart(self): rc = self.stop(); self.start(); return f"SIGTERM exit {rc}"
    def arm(self): pass
    def crash(self, after_kill=lambda: None):
        self.p.kill(); self.p.wait(); after_kill(); self.start(); return "SIGKILL, respawned"
    def recreate(self): return self.restart()
    def down(self):
        if self.p.poll() is None: self.p.kill()


class DockerServer:
    """docker/server/compose.yaml with bind mounts into W (project rxmq-e2e); the port is published on loopback"""
    def __init__(self):
        self.envf = f"{W}/e2e.env"
        self.cmd = [os.environ.get("E2E_DOCKER", "docker"), "compose", "-f", f"{HERE}/docker/server/compose.yaml",
                    "-p", "rxmq-e2e", "--env-file", self.envf]
    def dc(self, *a, check=True):
        r = subprocess.run(self.cmd + list(a), capture_output=True, text=True)
        if check and r.returncode: print(r.stdout, r.stderr)
        return r
    def write_env(self):
        image = os.environ.get("E2E_IMAGE") or "rxm-queue:" + open(f"{HERE}/VERSION").read().strip()
        env = {"RXMQ_IMAGE": image, "RXMQ_DB_DIR": f"{W}/db", "RXMQ_DATA_DIR": DATA, "RXMQ_OUT_DIR": f"{W}/out",
               "RXMQ_SOURCES_DIR": f"{W}/sources", "RXMQ_SECRETS_DIR": f"{W}/secrets", "RXMQ_PUBLISH": f"127.0.0.1:{PORT}",
               "RXMQ_LEASE_S": "6", "RXMQ_HEARTBEAT_S": "1.5", "RXMQ_RETRY_S": "0.5", "RXMQ_UPLOAD_MAX_MB": "48",
               "RXMQ_ADMIN_TOKEN_FILE": "/run/secrets/rxmq/admin-token"}
        if os.path.exists("/sys/fs/selinux/enforce") and open("/sys/fs/selinux/enforce").read().strip() == "1":
            env["RXMQ_SELINUX_RELABEL"] = "1"
        with open(self.envf, "w") as f: f.writelines(f"{k}={v}\n" for k, v in env.items())
    def cid(self): return self.dc("ps", "-q", "server").stdout.strip()
    def inspect(self, fmt):
        return subprocess.run(["docker", "inspect", "-f", fmt, self.cid()], capture_output=True, text=True).stdout.strip()
    def start(self): self.dc("up", "-d", "server")
    def restart(self): t = time.time(); self.dc("restart", "server"); return f"docker compose restart ({time.time() - t:.1f}s)"
    def arm(self):                             # look up the PID now: the docker CLI takes ~1 s, the kill must not
        self.armed = (int(self.inspect("{{.RestartCount}}") or 0), int(self.inspect("{{.State.Pid}}")))
    def crash(self, after_kill=lambda: None):
        n0, pid = getattr(self, "armed", None) or (int(self.inspect("{{.RestartCount}}") or 0), int(self.inspect("{{.State.Pid}}")))
        self.armed = None
        os.kill(pid, signal.SIGKILL)                                        # the server process (PID 1 in the container)
        after_kill()
        back = wait(lambda: int(self.inspect("{{.RestartCount}}") or 0) > n0 and self.inspect("{{.State.Running}}") == "true", 30)
        return f"SIGKILL of PID 1; restart policy brought it back: {back} (RestartCount {self.inspect('{{.RestartCount}}')})"
    def recreate(self):
        old = self.cid(); self.dc("down"); self.dc("up", "-d", "server")
        return f"docker compose down + up (new container: {self.cid() != old})"
    def down(self): self.dc("down", "-v", check=False)


def main():
    srv = DockerServer() if MODE == "docker" else LocalServer()
    if MODE == "docker": srv.down()                                          # a leftover run holds W's mounts
    if os.path.exists(W): shutil.rmtree(W)
    os.makedirs(f"{DATA}/test"); os.makedirs(f"{DATA}/models/x@1")
    for d in ("db", "out/results", "out/spool", "sources", "secrets"): os.makedirs(f"{W}/{d}")
    for n in "abcd":
        with open(f"{DATA}/test/{n}.bin", "wb") as f: f.write(hashlib.sha256(n.encode()).digest() * 9375)   # 300 kB
    open(f"{DATA}/models/x@1/config.json", "w").write("{}")
    fd = os.open(f"{W}/secrets/admin-token", os.O_WRONLY | os.O_CREAT, 0o400)
    with os.fdopen(fd, "w") as f: f.write(ADMIN_TOKEN + "\n")
    if MODE == "docker":                       # the container runs as 10001: it owns what it writes (db, out, its secret)
        srv.write_env()
        for d in ("db", "out", "secrets"):
            for dp, dn, fn in os.walk(f"{W}/{d}"):
                for x in [dp] + [os.path.join(dp, f) for f in fn]: os.chown(x, 10001, 10001)
    procs = []
    try:
        srv.start()                            # first, so the server (not the host admin) creates the database
        assert wait(lambda: req("GET", "/healthz")[0] == 200, 60), "server did not start"
        admin("manifest")
        run(procs, srv)
    finally:
        for p in procs:
            if p.poll() is None: p.kill()
        if MODE == "docker":
            open(f"{W}/server.log", "w").write(srv.dc("logs", "--no-color", "server", check=False).stdout)
        srv.down()
    n_ok = sum(r[1] for r in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    json.dump(RESULTS, open(f"{W}/results.json", "w"), indent=1)
    sys.exit(0 if n_ok == len(RESULTS) else 1)


def run(procs, srv):
    ok("/healthz without a token -> 200", req("GET", "/healthz")[0] == 200)
    if MODE == "docker": docker_checks(srv)
    # ---- tokens and invites
    ok("no token -> 401", req("POST", "/api/v0/lease", {"caps": {"job_types": ["noop"]}})[0] == 401)
    ok("bad token -> 401", req("POST", "/api/v0/lease", {"caps": {"job_types": ["noop"]}}, token="x" * 43)[0] == 401)
    ok("bad invite -> 403", req("POST", "/api/v0/register", {"invite": "nope", "name": "w"})[0] == 403)
    inv = admin("invite", "--uses", "3").strip()
    toks = {}
    for n in ("wa", "wb", "wc"):
        c, b = req("POST", "/api/v0/register", {"invite": inv, "name": n}); toks[n] = b.get("token")
    ok("invite registers 3 workers", all(toks.values()))
    ok("used-up invite -> 403", req("POST", "/api/v0/register", {"invite": inv, "name": "wd"})[0] == 403)
    t_client = admin("add-worker", "client").split()[3]

    # ---- data access rules
    ok("data without a lease -> 403", req("GET", "/data/test/a.bin", token=t_client)[0] == 403)
    ok("data not in manifest -> 404", req("GET", "/data/test/zzz.bin", token=t_client)[0] == 404)
    ok("data path traversal -> 404", req("GET", "/data/../q.sqlite", token=t_client)[0] in (400, 404))

    # ---- 3 workers, one killed mid-job
    admin("enqueue-noop", "9", "--seconds", "3", "--size", "200000", "--inputs", "test/a.bin", "--tag", "kill")
    ids = jobs_of("kill")
    wa = worker("wa", toks["wa"], "--idle-exit", "30"); wb = worker("wb", toks["wb"], "--idle-exit", "30")
    wc = worker("wc", toks["wc"], "--idle-exit", "30"); procs += [wa, wb, wc]
    wid = {n: db().execute("SELECT id FROM workers WHERE name=?", (n,)).fetchone()[0] for n in ("wa", "wb", "wc")}
    victim = None
    if wait(lambda: db().execute("SELECT job_id FROM leases WHERE worker_id=? AND state='active'", (wid["wa"],)).fetchone(), 15):
        time.sleep(1.0)
        victim = db().execute("SELECT job_id FROM leases WHERE worker_id=? AND state='active'", (wid["wa"],)).fetchone()[0]
        wa.send_signal(signal.SIGKILL); wa.wait()
    ok("worker wa killed mid-job", victim is not None, f"job {victim}")
    ok("all 9 jobs accepted", wait(lambda: all(states()[i] == "accepted" for i in ids), 60))
    ev = [e["kind"] for e in events(victim)]
    ok("killed job: lease expired, reassigned, accepted once",
       ev.count("expire") >= 1 and ev.count("lease") >= 2 and ev.count("accept") + ev.count("accept_late") == 1, str(ev))
    acc = db().execute("SELECT accepted_by FROM jobs WHERE id=?", (victim,)).fetchone()[0]
    ok("killed job accepted from another worker", acc != wid["wa"], f"by {acc}")
    ok("results match the expected bytes", all(
        sha256_file(f"{RES}/{i}/out.bin") == hashlib.sha256(__import__("jobs.noop", fromlist=["x"]).content(
            json.loads(db().execute("SELECT params FROM jobs WHERE id=?", (i,)).fetchone()[0])["seed"], 200000)).hexdigest() for i in ids))
    # restart the killed worker after its lease expired and the job was done elsewhere: it must drop the job cleanly
    wa2 = worker("wa", toks["wa"], "--idle-exit", "2"); procs.append(wa2); wa2.wait(30)
    ok("restarted worker drops its dead lease", "dropping it" in open(f"{W}/wa.log").read() and not os.path.exists(f"{W}/wa/current.json"))
    for p in (wb, wc): p.wait(30)

    # ---- killed and restarted before the lease expires: resumes the same lease
    admin("enqueue-noop", "1", "--seconds", "4", "--size", "1000", "--seed", "100", "--tag", "resume")
    j = jobs_of("resume")[0]
    w = worker("wb", toks["wb"], "--idle-exit", "2"); procs.append(w)
    wait(lambda: states()[j] == "leased", 10); time.sleep(1); w.send_signal(signal.SIGKILL); w.wait()
    w = worker("wb", toks["wb"], "--idle-exit", "2"); procs.append(w); w.wait(30)
    ev = [e["kind"] for e in events(j)]
    ok("restart within the lease resumes it", states()[j] == "accepted" and ev.count("lease") == 1 and "expire" not in ev, str(ev))

    # ---- duplicate completion, corrupted upload, caps (a hand-written client)
    admin("enqueue-noop", "2", "--seconds", "0", "--size", "5000", "--seed", "200", "--tag", "client")
    caps = {"caps": {"job_types": ["noop"], "disk_gb": 1}}
    c, r = req("POST", "/api/v0/lease", caps, token=t_client); job = r["job"]
    body = __import__("jobs.noop", fromlist=["x"]).content(job["params"]["seed"], 5000)
    sha = hashlib.sha256(body).hexdigest()
    up = f"/api/v0/upload/{job['id']}/{job['lease_id']}"
    ok("upload of a non-output name -> 403", req("PUT", f"{up}/evil.sh?offset=0", b"x", token=t_client)[0] == 403)
    ok("upload over the cap -> 413", req("PUT", f"{up}/out.bin?offset=0", b"x" * 5001, token=t_client)[0] == 413)
    bad = bytearray(body); bad[100] ^= 1
    req("PUT", f"{up}/out.bin?offset=0", bytes(bad), token=t_client)
    c, r = req("POST", "/api/v0/complete", {"job_id": job["id"], "lease_id": job["lease_id"], "files": {"out.bin": {"sha256": sha, "size": 5000}}}, token=t_client)
    ok("corrupted upload rejected (sha mismatch -> 422)", c == 422 and r.get("bad") == ["out.bin"], f"{c} {r}")
    ok("corrupted upload deleted on the server", req("GET", f"{up}/out.bin", token=t_client)[1].get("size") == 0)
    ok("wrong offset -> 409 with the resume point", req("PUT", f"{up}/out.bin?offset=7", body, token=t_client)[0] == 409)
    req("PUT", f"{up}/out.bin?offset=0", body[:3000], token=t_client)
    c, r = req("PUT", f"{up}/out.bin?offset=3000", body[3000:], token=t_client)
    ok("chunked upload resumes at the offset", c == 200 and r["size"] == 5000)
    c, r = req("POST", "/api/v0/complete", {"job_id": job["id"], "lease_id": job["lease_id"], "files": {"out.bin": {"sha256": sha, "size": 5000}}, "result": {"x": 1}}, token=t_client)
    ok("re-uploaded result accepted", c == 200 and r.get("duplicate") is False, f"{c} {r}")
    c, r = req("POST", "/api/v0/complete", {"job_id": job["id"], "lease_id": job["lease_id"], "files": {"out.bin": {"sha256": "0" * 64, "size": 5000}}}, token=t_client)
    ok("duplicate completion ignored", c == 200 and r.get("duplicate") is True, f"{c} {r}")
    ok("upload to an accepted job -> 409", req("PUT", f"{up}/out.bin?offset=0", body, token=t_client)[0] == 409)
    ok("accepted result unchanged", sha256_file(f"{RES}/{job['id']}/out.bin") == sha and
       [e["kind"] for e in events(job["id"])].count("accept") == 1)
    c, r = req("POST", "/api/v0/lease", caps, token=t_client); job2 = r["job"]
    ok("another worker's lease -> 403", req("POST", "/api/v0/heartbeat", {"job_id": job2["id"], "lease_id": job2["lease_id"]}, token=toks["wc"])[0] == 403)
    req("POST", "/api/v0/fail", {"job_id": job2["id"], "lease_id": job2["lease_id"], "error": "test", "abandon": True}, token=t_client)
    r2 = db().execute("SELECT state, attempts FROM jobs WHERE id=?", (job2["id"],)).fetchone()
    ok("abandon returns the job without counting an attempt", tuple(r2) == ("queued", 0), str(tuple(r2)))
    admin("cancel", str(job2["id"]))
    # the server-wide upload ceiling (--upload-max-mb 48 = 50331648 B) under a job type's own cap (64 MiB)
    admin("enqueue-noop", "1", "--seconds", "0", "--size", str(64 << 20), "--seed", "250", "--tag", "bigout")
    c, r = req("POST", "/api/v0/lease", caps, token=t_client); job3 = r["job"]
    u3 = f"/api/v0/upload/{job3['id']}/{job3['lease_id']}/out.bin"
    c1, _ = req("PUT", f"{u3}?offset=0", b"x" * ((48 << 20) - 10), token=t_client)
    c, r = req("PUT", f"{u3}?offset={(48 << 20) - 10}", b"x" * 11, token=t_client)
    ok("upload over the server's ceiling -> 413", c1 == 200 and c == 413 and "50331648" in r.get("error", ""), f"{c1} {c} {r}")
    admin("cancel", str(job3["id"]))

    # ---- read-only admin summary
    S = "/api/v0/admin/summary"
    ok("admin summary: no / worker / wrong token -> 401", req("GET", S)[0] == 401 and req("GET", S, token=toks["wb"])[0] == 401
       and req("GET", S, token=ADMIN_TOKEN[:-1] + "x")[0] == 401)
    c, r = req("GET", S, token=ADMIN_TOKEN)
    ok("admin summary with the admin token", c == 200 and any(l["level"] == "kill" and l["state"] == "accepted" and l["n"] == 9
       for l in r.get("levels", [])) and "token" not in json.dumps(r.get("workers")), f"{c} {len(r.get('events', []))} events")

    # ---- disk cap: eviction, and jobs larger than the cache are not leased to it
    t_small = admin("add-worker", "small").split()[3]
    for i, n in enumerate("abca"):
        admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--seed", str(300 + i), "--inputs", f"test/{n}.bin", "--tag", "disk")
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--seed", "310", "--inputs", "test/a.bin,test/b.bin,test/c.bin", "--tag", "big")
    w = worker("small", t_small, "--cache-gb", "0.0007", "--idle-exit", "4"); procs.append(w); w.wait(40)
    st = stats("small"); used = sum(os.path.getsize(os.path.join(f"{W}/small/cache/blobs", f)) for f in os.listdir(f"{W}/small/cache/blobs"))
    ok("disk cap: 4 jobs done with eviction", all(states()[i] == "accepted" for i in jobs_of("disk")) and st and st[-1]["evicted"] >= 2,
       f"evicted {st[-1]['evicted'] if st else None}, cache {used} B <= 700000")
    ok("cache stays under its cap", used <= 700_000)
    ok("job bigger than the cache not leased to it", states()[jobs_of("big")[0]] == "queued")
    # Range resume: half of d.bin already in the cache as .part
    dsha = sha256_file(f"{W}/data/test/d.bin")
    os.makedirs(f"{W}/rng/cache/blobs", exist_ok=True)
    open(f"{W}/rng/cache/blobs/{dsha}.part", "wb").write(open(f"{W}/data/test/d.bin", "rb").read()[:120000])
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--seed", "320", "--inputs", "test/d.bin", "--tag", "range")
    t_rng = admin("add-worker", "rng").split()[3]
    admin("cancel", str(jobs_of("big")[0]))
    w = worker("rng", t_rng, "--idle-exit", "2"); procs.append(w); w.wait(30)
    st = stats("rng")
    ok("partial download resumed with a Range request", st and st[-1]["down_bytes"] == 300000 - 120000 and states()[jobs_of("range")[0]] == "accepted",
       f"down {st[-1]['down_bytes'] if st else None} of 300000")

    # ---- malformed jobs through the real server (the queue holds a known type with bad parameters)
    c = db()
    for p in ({"seconds": "3", "size": 1, "seed": 1}, {"seconds": 1, "size": 1, "seed": 1, "cmd": "rm -rf /"}):
        c.execute("INSERT INTO jobs(type, version, params, inputs, requires, input_bytes, state, max_attempts, level, created, updated) "
                  "VALUES('noop', 1, ?, '[]', '{}', 0, 'queued', 2, 'malformed', 0, 0)", (json.dumps(p),))
    c.commit()
    wb2 = worker("wb", toks["wb"], "--idle-exit", "4"); wc2 = worker("wc", toks["wc"], "--idle-exit", "4"); procs += [wb2, wc2]
    wb2.wait(40); wc2.wait(40)
    mids = jobs_of("malformed")
    ok("malformed jobs rejected by workers, then failed", all(states()[i] == "failed" for i in mids) and
       all(len(events(i, "reject")) == 2 for i in mids), str([states()[i] for i in mids]))
    ok("workers survive malformed jobs", wb2.returncode == 0 and wc2.returncode == 0)

    # ---- a fake malicious server
    good = {"type": "noop", "version": 1, "params": {"seconds": 0, "size": 1, "seed": 1}, "inputs": [], "lease_id": "f" * 32, "heartbeat_s": 5}
    Fake.queue = [dict(good, id=1, type="shell", params={"cmd": "id"}),
                  dict(good, id=2, params={"seconds": 0, "size": 1, "seed": 1, "argv": ["sh"]}),
                  dict(good, id=3, params={"seconds": 0, "size": 1, "seed": 1, "inputs": ["test/../../etc/passwd"]},
                       inputs=[{"name": "test/../../etc/passwd", "sha256": "0" * 64, "size": 1}]),
                  dict(good, id=4, params={"seconds": 0, "size": 1, "seed": 1, "inputs": ["test/a.bin"]},
                       inputs=[{"name": "test/a.bin", "sha256": "0" * 64, "size": 1}, {"name": "models/x@1/config.json", "sha256": "0" * 64, "size": 1}]),
                  dict(good, id=5, type="eval_mix", version=0, params={"model": "m", "revision": "r", "recipe": "a", "golden": "g", "kv": "bf16"}),
                  dict(good, id=6, type="quant_layer", params={"model": "m", "revision": "r", "arch": "qwen3_5", "K": 3, "codebook": "i8",
                       "level": "l", "layer": 1, "stamp": "0" * 16, "settings": {"sigma_reg": 0.025, "sigma_reg_in": None, "sigma_reg_out": None,
                       "scaled_hessian": True, "qtf2": True, "hout": None, "seed_offset": 0}})]
    fs = ThreadingHTTPServer(("127.0.0.1", PORT + 1), Fake); threading.Thread(target=fs.serve_forever, daemon=True).start()
    lf = open(f"{W}/fake.log", "w")
    p = subprocess.Popen([PY, "-I", f"{HERE}/worker/worker.py", "--server", f"http://127.0.0.1:{PORT + 1}", "--token", "t", "--state", f"{W}/fake",
                          "--poll", "0.2", "--idle-exit", "3"], stdout=lf, stderr=subprocess.STDOUT); procs.append(p); p.wait(30); fs.shutdown()
    rej = {f["job_id"]: f for f in Fake.fails}
    ok("fake server: all 6 bad jobs rejected", sorted(rej) == [1, 2, 3, 4, 5, 6] and all(f.get("reject") for f in rej.values()),
       "; ".join(f"{k}: {v['error'][19:80]}" for k, v in sorted(rej.items())))
    ok("fake server: no download attempted", not Fake.data_hits, str(Fake.data_hits))

    # ---- graceful stop: finish, then abandon
    admin("enqueue-noop", "1", "--seconds", "3", "--size", "100", "--seed", "400", "--tag", "stopfin")
    j = jobs_of("stopfin")[0]; w = worker("wc", toks["wc"]); procs.append(w)
    wait(lambda: states()[j] == "leased", 10); time.sleep(0.5); w.send_signal(signal.SIGTERM); w.wait(30)
    ok("SIGTERM: finishes the current job, then exits", states()[j] == "accepted" and w.returncode == 0)
    admin("enqueue-noop", "1", "--seconds", "30", "--size", "100", "--seed", "401", "--tag", "stopab")
    j = jobs_of("stopab")[0]; w = worker("wc", toks["wc"], "--on-stop", "abandon"); procs.append(w)
    wait(lambda: states()[j] == "leased", 10); time.sleep(0.5); t0 = time.time(); w.send_signal(signal.SIGTERM); w.wait(30)
    r = db().execute("SELECT state, attempts FROM jobs WHERE id=?", (j,)).fetchone()
    ok("--on-stop abandon: job handed back at once, attempt not counted", tuple(r) == ("queued", 0) and time.time() - t0 < 5, str(tuple(r)))
    admin("cancel", str(j))

    # ---- time windows
    lt = time.localtime(); m = lt.tm_hour * 60 + lt.tm_min
    hm = lambda x: f"{(x // 60) % 24:02d}:{x % 60:02d}"
    ok("window logic", in_windows([f"{hm(m - 5)}-{hm(m + 5)}"]) and not in_windows([f"{hm(m + 10)}-{hm(m + 20)}"]) and
       in_windows([f"{hm(m + 10)}-{hm(m + 5)}"]) and in_windows([]))
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--seed", "500", "--tag", "win")
    j = jobs_of("win")[0]
    w = worker("wb", toks["wb"], "--window", f"{hm(m + 30)}-{hm(m + 40)}"); procs.append(w); time.sleep(4)
    ok("outside its window a worker leases nothing", states()[j] == "queued")
    w.send_signal(signal.SIGTERM); w.wait(10)
    admin("cancel", str(j))

    # ---- a worker configured by environment only (the image's way): invite -> token file (0600) -> token file
    inv = admin("invite").strip(); tf = f"{W}/envw/token"
    admin("enqueue-noop", "2", "--seconds", "0", "--size", "100", "--seed", "550", "--tag", "envw")
    env = {k: v for k, v in os.environ.items() if not k.startswith("RXMW_")}
    env.update(RXMW_SERVER=URL, RXMW_STATE=f"{W}/envw", RXMW_TOKEN_FILE=tf, RXMW_INVITE=inv, RXMW_NAME="envw", RXMW_IMAGE="e2e-env")
    a = [PY, "-I", f"{HERE}/worker/worker.py", "--poll", "0.5", "--idle-exit", "2", "--max-jobs", "1"]
    r1 = subprocess.run(a, env=env, capture_output=True, text=True, timeout=60)
    env.pop("RXMW_INVITE")
    r2 = subprocess.run(a, env=env, capture_output=True, text=True, timeout=60)
    wrow = db().execute("SELECT caps FROM workers WHERE name='envw'").fetchall()
    ok("worker from env only: registers once, token file 0600, reuses it", r1.returncode == 0 and r2.returncode == 0 and
       len(wrow) == 1 and os.stat(tf).st_mode & 0o777 == 0o600 and json.loads(wrow[0][0])["image"] == "e2e-env" and
       all(states()[i] == "accepted" for i in jobs_of("envw")), f"rc {r1.returncode}/{r2.returncode}, workers {len(wrow)}, "
       f"mode {oct(os.stat(tf).st_mode & 0o777) if os.path.exists(tf) else None}")

    # ---- the server goes away mid-run: graceful restart, crash (SIGKILL), remove + recreate
    for k, how in enumerate(("restart", "crash", "recreate")):
        tag = f"rs-{how}"
        admin("enqueue-noop", "6", "--seconds", "3", "--size", "200000", "--seed", str(600 + 10 * k), "--tag", tag)
        ids = jobs_of(tag)
        ws = [worker(n, toks[n], "--idle-exit", "20") for n in ("wb", "wc")]; procs += ws
        mid_ok = wait(lambda: sum(states()[i] == "accepted" for i in ids) >= 1 and sum(states()[i] == "leased" for i in ids) >= 2, 40, 0.1)
        mid = collections.Counter(states()[i] for i in ids)
        t0 = time.time(); how_s = getattr(srv, how)()
        back = wait(lambda: req("GET", "/healthz")[0] == 200, 60)
        ok(f"{how}: server back mid-run", mid_ok and back, f"{how_s}; healthz after {time.time() - t0:.1f}s; jobs at the time {dict(mid)}")
        done = wait(lambda: all(states()[i] == "accepted" for i in ids), 120)
        for w in ws: w.wait(60)
        acc = {i: [e["kind"] for e in events(i)] for i in ids}
        once = all(v.count("accept") + v.count("accept_late") == 1 for v in acc.values())
        seed = lambda i: json.loads(db().execute("SELECT params FROM jobs WHERE id=?", (i,)).fetchone()[0])["seed"]
        exact = all(os.path.exists(f"{RES}/{i}/out.bin") and sha256_file(f"{RES}/{i}/out.bin") == hashlib.sha256(
            __import__("jobs.noop", fromlist=["x"]).content(seed(i), 200000)).hexdigest() for i in ids)
        n_exp = sum(v.count("expire") for v in acc.values()); n_late = sum(v.count("accept_late") for v in acc.values())
        integ = db().execute("PRAGMA integrity_check").fetchone()[0]
        ok(f"{how}: all 6 jobs accepted once each, byte-exact, db intact", done and once and exact and integ == "ok",
           f"expired leases {n_exp}, late accepts {n_late}, integrity {integ}")
        ok(f"{how}: workers rode it out", all(w.returncode == 0 for w in ws), str([w.returncode for w in ws]))
    # ---- killed mid-upload: the partial upload survives in the spool and the worker resumes it
    admin("enqueue-noop", "1", "--seconds", "0", "--size", str(40 << 20), "--seed", "700", "--tag", "rs-upload")
    j = jobs_of("rs-upload")[0]; n0 = len(stats("wb")); srv.arm()
    w = worker("wb", toks["wb"], "--idle-exit", "20", "--chunk-mb", "1"); procs.append(w)
    part = lambda: max([os.path.getsize(f) for f in glob.glob(f"{SPOOL}/{j}/*/out.bin")] or [0])
    hit = wait(lambda: 8 << 20 <= part() <= 32 << 20, 60, 0.002); s0 = part()
    s1 = []; how_s = srv.crash(lambda: s1.append(part())); s1 = s1[0]   # spool right after the kill
    done = wait(lambda: states()[j] == "accepted", 90); w.wait(60)
    st = stats("wb")[n0:]; up = st[-1].get("up_bytes") if st else None
    seed = json.loads(db().execute("SELECT params FROM jobs WHERE id=?", (j,)).fetchone()[0])["seed"]
    exact = done and sha256_file(f"{RES}/{j}/out.bin") == hashlib.sha256(__import__("jobs.noop", fromlist=["x"]).content(seed, 40 << 20)).hexdigest()
    ok("crash mid-upload: partial upload kept, resumed, accepted once, byte-exact",
       hit and s1 >= s0 - (1 << 20) and s1 > 0 and exact and up is not None and up <= (41 << 20) and
       [e["kind"] for e in events(j)].count("accept") + [e["kind"] for e in events(j)].count("accept_late") == 1,
       f"{how_s}; spool {s0 / 2**20:.1f} MiB before the kill, {s1 / 2**20:.1f} MiB just after; worker sent {None if up is None else up / 2**20:.1f} MiB of 40")
    if MODE == "docker":
        ok("container healthy at the end", wait(lambda: srv.inspect("{{.State.Health.Status}}") == "healthy", 40),
           srv.inspect("{{.State.Health.Status}}"))


def docker_checks(srv):
    ok("container: healthcheck turns healthy", wait(lambda: srv.inspect("{{.State.Health.Status}}") == "healthy", 40),
       srv.inspect("{{.State.Health.Status}}"))
    uid = srv.dc("exec", "-T", "server", "id", "-u").stdout.strip()
    ps = srv.dc("exec", "-T", "server", "cat", "/proc/1/status").stdout
    puid = [l.split()[1] for l in ps.splitlines() if l.startswith("Uid:")]
    ok("container: server runs as uid 10001 (non-root)", uid == "10001" and puid == ["10001"], f"exec {uid}, PID 1 {puid}")
    ok("container: read-only rootfs, no capabilities, restart unless-stopped",
       srv.inspect("{{.HostConfig.ReadonlyRootfs}}") == "true" and srv.inspect("{{.HostConfig.RestartPolicy.Name}}") == "unless-stopped"
       and "ALL" in srv.inspect("{{.HostConfig.CapDrop}}"),
       f"{srv.inspect('{{.HostConfig.ReadonlyRootfs}}')} {srv.inspect('{{.HostConfig.CapDrop}}')} {srv.inspect('{{.HostConfig.RestartPolicy.Name}}')}")
    r = srv.dc("exec", "-T", "server", "rxmq", "admin", "invite", "--note", "from-inside")
    ok("container: `rxmq admin` works inside the running server", r.returncode == 0 and len(r.stdout.strip()) >= 20)
    r = srv.dc("run", "--rm", "-T", "admin", "list")
    ok("container: `compose run --rm admin` works beside it", r.returncode == 0, (r.stderr or "").strip()[-200:])
    ok("container: port published on loopback only", srv.dc("port", "server", "8740").stdout.strip() == f"127.0.0.1:{PORT}",
       srv.dc("port", "server", "8740").stdout.strip())


if __name__ == "__main__":
    main()
