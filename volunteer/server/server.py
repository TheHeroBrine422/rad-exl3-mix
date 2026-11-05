#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""server.py --db QUEUE.sqlite --data DATAROOT --spool DIR --results DIR [--bind 127.0.0.1] [--port 8740]

The job queue (stdlib only). Workers authenticate with a bearer token (from /register with an invite code, or
admin.py add-worker). Plain HTTP: put it behind a TLS terminator or a tunnel before anything leaves localhost / LAN.
Every flag can also come from the environment as RXMQ_<FLAG> (e.g. RXMQ_DB, RXMQ_PORT, RXMQ_ADMIN_TOKEN_FILE); a flag
on the command line wins. The container image (docker/server/) configures it that way.

  POST /api/v0/register   {invite, name}                              -> {worker_id, token}
  POST /api/v0/lease      {caps}                                      -> {job: {...}} | {job: null, retry_s}
  POST /api/v0/heartbeat  {job_id, lease_id}                          -> {ok, cancel, lease_expires}
  GET  /api/v0/upload/J/LEASE/NAME                                    -> {size} (resume point)
  PUT  /api/v0/upload/J/LEASE/NAME?offset=N   body: bytes             -> {size}
  POST /api/v0/complete   {job_id, lease_id, files: {name: {sha256, size}}, result}
  POST /api/v0/fail       {job_id, lease_id, error, abandon, reject}
  GET  /data/RELPATH      (Range: bytes=a-[b])  only files in the manifest that a job leased by this worker lists
  GET  /api/v0/status
  GET  /healthz           no auth: 200 if the database answers (container healthcheck)
  GET  /api/v0/admin/summary  read-only queue overview, only with --admin-token-file, bearer = that token

There are no admin endpoints that change anything: admin.py works on the database and the data root directly, on
this host (or in the server container: `rxmq admin ...`)."""
import argparse, errno, hashlib, hmac, json, os, re, secrets, shutil, signal, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs, unquote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import API, NAME_RE, safe_relpath, sha256_file, read_json
from common.schema import validate, SchemaError
import jobs
from server.store import Store

CHUNK_MAX = 64 << 20
JSON_MAX = 1 << 20
CAPS = {"type": "object", "required": ["job_types"], "properties": {
    "job_types": {"type": "array", "maxItems": 32, "items": {"type": "string", "pattern": "[a-z0-9_]{1,32}"}},
    "gpu": {"type": "boolean"}, "gpu_name": {"type": "string", "maxLength": 128},
    "vram_gb": {"type": "number", "minimum": 0, "maximum": 4096}, "ram_gb": {"type": "number", "minimum": 0, "maximum": 65536},
    "disk_gb": {"type": "number", "minimum": 0, "maximum": 1e6}, "image": {"type": "string", "maxLength": 128},
    "versions": {"type": "object", "additionalProperties": {"type": "integer"}}}}


class HTTPError(Exception):
    def __init__(self, code, msg, **extra): super().__init__(msg); self.code, self.msg, self.extra = code, msg, extra


class Queue:
    def __init__(self, a):
        self.a = a
        self.store = Store(a.db)
        self.data = os.path.abspath(a.data); self.spool = os.path.abspath(a.spool); self.results = os.path.abspath(a.results)
        for d in (self.spool, self.results): os.makedirs(d, exist_ok=True)
        self._man, self._man_mtime = {}, None
        self.lease_s, self.hb_s = a.lease_s, a.heartbeat_s
        self.chunk_max = int(a.chunk_max_mb * (1 << 20))
        self.upload_max = int(a.upload_max_mb * (1 << 20)) if a.upload_max_mb else None
        self.spool_max = int(a.spool_max_gb * 1e9) if a.spool_max_gb else None
        self.admin_hash = None
        if a.admin_token_file:
            tok = open(a.admin_token_file).read().strip()
            if len(tok) < 32: sys.exit(f"{a.admin_token_file}: admin token shorter than 32 characters")
            self.admin_hash = hashlib.sha256(tok.encode()).digest()

    def manifest(self):
        p = os.path.join(self.data, "manifest.json")
        m = os.path.getmtime(p) if os.path.exists(p) else None
        if m != self._man_mtime:
            try:
                self._man = (read_json(p, {}) or {}).get("files", {}); self._man_mtime = m
            except (OSError, ValueError) as e:              # e.g. written 0600 by another uid: keep the last good one
                print(f"manifest.json unreadable, keeping the previous one ({len(self._man)} files): {e!r}", flush=True)
        return self._man

    def data_path(self, rel):
        if not safe_relpath(rel) or rel not in self.manifest(): raise HTTPError(404, "not in manifest")
        return os.path.join(self.data, rel)

    # ------------------------------------------------------------ lease
    def lease(self, w, caps):
        try: validate(caps, CAPS, "caps")
        except SchemaError as e: raise HTTPError(400, str(e))
        self.store.expire()
        types = set(caps["job_types"]) & {n for n, m in jobs.REGISTRY.items() if m.IMPLEMENTED}
        vers = caps.get("versions", {})
        disk = caps.get("disk_gb", 0) * 1e9
        now = time.time()
        with self.store.tx() as db:
            db.execute("UPDATE workers SET last_seen=?, caps=? WHERE id=?", (now, json.dumps(caps), w["id"]))
            if db.execute("SELECT COUNT(*) FROM leases WHERE worker_id=? AND state='active'", (w["id"],)).fetchone()[0] >= self.a.max_leases:
                return {"job": None, "retry_s": 30, "why": "worker has its maximum of active leases"}
            for j in db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY priority DESC, id LIMIT 500"):
                if j["type"] not in types or vers.get(j["type"], j["version"]) != j["version"]: continue
                r = json.loads(j["requires"])
                if r.get("gpu") and not caps.get("gpu"): continue
                if r.get("ram_gb", 0) > caps.get("ram_gb", 0) or r.get("vram_gb", 0) > caps.get("vram_gb", 0): continue
                if j["input_bytes"] > disk: continue
                lid = secrets.token_hex(16); exp = now + self.lease_s
                db.execute("INSERT INTO leases VALUES(?,?,?,?,?, 'active', NULL, NULL)", (lid, j["id"], w["id"], now, exp))
                db.execute("UPDATE jobs SET state='leased', attempts=attempts+1, lease_id=?, lease_owner=?, lease_expires=?, "
                           "updated=? WHERE id=?", (lid, w["id"], exp, now, j["id"]))
                self.store.event(db, "lease", j["id"], w["id"], lid)
                man = self.manifest()
                return {"job": {"id": j["id"], "type": j["type"], "version": j["version"], "params": json.loads(j["params"]),
                                "inputs": [{"name": n, "sha256": man[n]["sha256"], "size": man[n]["size"]} for n in json.loads(j["inputs"])],
                                "outputs": jobs.get(j["type"]).outputs(json.loads(j["params"])),
                                "lease_id": lid, "lease_expires": exp, "heartbeat_s": self.hb_s}}
        return {"job": None, "retry_s": self.a.retry_s}

    def _lease(self, db, w, job_id, lease_id):
        l = db.execute("SELECT * FROM leases WHERE id=? AND job_id=?", (str(lease_id), job_id)).fetchone()
        if l is None or l["worker_id"] != w["id"]: raise HTTPError(403, "not your lease")
        j = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return l, j

    def heartbeat(self, w, b):
        now = time.time()
        with self.store.tx() as db:
            l, j = self._lease(db, w, int(b["job_id"]), b["lease_id"])
            db.execute("UPDATE workers SET last_seen=? WHERE id=?", (now, w["id"]))
            if l["state"] != "active" or j["state"] in ("accepted", "cancelled", "failed"):
                late_ok = l["state"] == "expired" and j["state"] in ("queued", "leased")
                return {"ok": False, "cancel": not late_ok, "state": l["state"], "job_state": j["state"]}
            exp = now + self.lease_s
            db.execute("UPDATE leases SET expires=? WHERE id=?", (exp, l["id"]))
            db.execute("UPDATE jobs SET lease_expires=? WHERE id=?", (exp, j["id"]))
            return {"ok": True, "cancel": False, "lease_expires": exp}

    def fail(self, w, b):
        with self.store.tx() as db:
            l, j = self._lease(db, w, int(b["job_id"]), b["lease_id"])
            if l["state"] not in ("active", "expired"): return {"ok": True, "ignored": l["state"]}
            kind = "abandon" if b.get("abandon") else "reject" if b.get("reject") else "fail"
            err = f"{kind}: {str(b.get('error', ''))[:1500]}"
            self.store.end_lease(db, l, {"abandon": "abandoned", "reject": "rejected", "fail": "failed"}[kind],
                                 error=err, count_attempt=kind != "abandon")
            self.store.event(db, kind, j["id"], w["id"], l["id"], {"error": err[:500]})
        return {"ok": True}

    # ------------------------------------------------------------ uploads
    def _upload_target(self, w, job_id, lease_id, name):
        if not NAME_RE.match(name): raise HTTPError(400, "bad name")
        with self.store.tx() as db:
            l, j = self._lease(db, w, job_id, lease_id)
            if j["state"] == "accepted": raise HTTPError(409, "job already accepted", accepted=True)
            if j["state"] in ("cancelled", "failed") or l["state"] not in ("active", "expired"):
                raise HTTPError(409, f"lease {l['state']}, job {j['state']}")
            outs = jobs.get(j["type"]).outputs(json.loads(j["params"]))
        if name not in outs: raise HTTPError(403, f"{name} is not an output of {j['type']}")
        d = os.path.join(self.spool, str(job_id), lease_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name), outs[name]

    def upload_size(self, w, job_id, lease_id, name):
        p, _ = self._upload_target(w, job_id, lease_id, name)
        return {"size": os.path.getsize(p) if os.path.exists(p) else 0}

    def upload(self, w, job_id, lease_id, name, offset, n, rfile):
        p, cap = self._upload_target(w, job_id, lease_id, name)
        cur = os.path.getsize(p) if os.path.exists(p) else 0
        if offset != cur: raise HTTPError(409, "offset mismatch", size=cur)
        if n > self.chunk_max: raise HTTPError(413, f"chunk too large (max {self.chunk_max} bytes)")
        if self.upload_max is not None: cap = min(cap, self.upload_max)
        if cur + n > cap: raise HTTPError(413, f"{name} over its cap of {cap} bytes")
        if self.spool_max is not None and self.spool_bytes() + n > self.spool_max:
            raise HTTPError(507, "server upload spool is full, retry later", size=cur)
        with open(p, "ab") as f:
            left = n
            while left:
                b = rfile.read(min(left, 1 << 20))
                if not b: break
                f.write(b); left -= len(b)
        if left: raise HTTPError(400, "short body", size=os.path.getsize(p))
        return {"size": cur + n}

    def spool_bytes(self):
        n = 0
        for dp, _, fn in os.walk(self.spool):
            for f in fn:
                try: n += os.path.getsize(os.path.join(dp, f))
                except OSError: pass
        return n

    # ------------------------------------------------------------ completion
    def complete(self, w, b):
        job_id, lease_id = int(b["job_id"]), str(b["lease_id"])
        files, result = b.get("files"), b.get("result", {})
        if not isinstance(files, dict) or not isinstance(result, dict) or len(json.dumps(result)) > 64 << 10:
            raise HTTPError(400, "bad body")
        with self.store.tx() as db:
            l, j = self._lease(db, w, job_id, lease_id)
            if j["state"] == "accepted":
                self.store.event(db, "duplicate", job_id, w["id"], lease_id)
                return {"ok": True, "duplicate": True, "yours": j["accepted_lease"] == lease_id}
            if j["state"] in ("cancelled", "failed") or l["state"] not in ("active", "expired"):
                raise HTTPError(409, f"lease {l['state']}, job {j['state']}")
            params = json.loads(j["params"])
        m = jobs.get(j["type"])
        outs = m.outputs(params)
        if set(files) - set(outs): raise HTTPError(400, f"unexpected files {sorted(set(files) - set(outs))}")
        miss = m.required_outputs(params) - set(files)
        if miss: raise HTTPError(400, f"missing outputs {sorted(miss)}")
        d = os.path.join(self.spool, str(job_id), lease_id)
        bad = []
        for n, meta in files.items():                      # hash outside the lock (hundreds of MB)
            p = os.path.join(d, n)
            if not os.path.exists(p) or os.path.getsize(p) != meta.get("size") or sha256_file(p) != meta.get("sha256"):
                bad.append(n)
                if os.path.exists(p): os.remove(p)
        if bad:
            with self.store.tx() as db: self.store.event(db, "upload_bad", job_id, w["id"], lease_id, {"files": bad})
            raise HTTPError(422, "uploaded bytes do not match the declared sha256 / size: re-upload", bad=bad)
        try:
            m.accept(params, {n: os.path.join(d, n) for n in files}, result, lambda rel: self.data_path(rel))
        except Exception as e:
            with self.store.tx() as db:
                l2 = db.execute("SELECT * FROM leases WHERE id=?", (lease_id,)).fetchone()
                self.store.end_lease(db, l2, "rejected", error=f"result rejected: {e}")
                self.store.event(db, "result_rejected", job_id, w["id"], lease_id, {"error": str(e)[:500]})
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPError(422, f"result rejected: {e}", rejected=True)
        with self.store.tx() as db:
            j = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if j["state"] == "accepted":                   # someone else won the race while we hashed
                self.store.event(db, "duplicate", job_id, w["id"], lease_id)
                return {"ok": True, "duplicate": True, "yours": False}
            if j["state"] in ("cancelled", "failed"): raise HTTPError(409, f"job {j['state']}")
            dst = os.path.join(self.results, str(job_id))
            if os.path.exists(dst): shutil.rmtree(dst)
            try: os.replace(d, dst)
            except OSError as e:                           # spool and results on different filesystems
                if e.errno != errno.EXDEV: raise
                shutil.copytree(d, dst + ".part"); os.replace(dst + ".part", dst); shutil.rmtree(d)
            now = time.time()
            late = j["lease_id"] != lease_id
            db.execute("UPDATE leases SET state='done', ended=? WHERE id=?", (now, lease_id))
            if late and j["lease_id"]: db.execute("UPDATE leases SET state='superseded', ended=? WHERE id=?", (now, j["lease_id"]))
            db.execute("UPDATE jobs SET state='accepted', accepted_lease=?, accepted_by=?, accepted_at=?, result=?, files=?, "
                       "lease_id=NULL, lease_owner=NULL, lease_expires=NULL, updated=? WHERE id=?",
                       (lease_id, w["id"], now, json.dumps(result), json.dumps(files), now, job_id))
            self.store.event(db, "accept_late" if late else "accept", job_id, w["id"], lease_id)
        shutil.rmtree(os.path.join(self.spool, str(job_id)), ignore_errors=True)
        return {"ok": True, "duplicate": False}

    def may_read(self, w, rel):
        for r in self.store.q("SELECT j.inputs FROM leases l JOIN jobs j ON j.id=l.job_id WHERE l.worker_id=? AND l.state='active'",
                              (w["id"],)):
            if rel in json.loads(r["inputs"]): return True
        return False

    def status(self):
        rows = self.store.q("SELECT type, state, COUNT(*) n FROM jobs GROUP BY type, state")
        return {"jobs": [dict(r) for r in rows], "workers": self.store.q("SELECT COUNT(*) FROM workers")[0][0]}

    def health(self):
        self.store.q("SELECT 1")
        if not os.path.isdir(self.data) or not os.access(self.spool, os.W_OK) or not os.access(self.results, os.W_OK):
            raise HTTPError(503, "data / spool / results not usable")
        return {"ok": True}

    def admin_ok(self, token):
        return self.admin_hash is not None and token is not None and \
            hmac.compare_digest(hashlib.sha256(token.encode()).digest(), self.admin_hash)

    def admin_summary(self):
        """read-only: job counts per level and state, workers, the last 50 events (no tokens, no params)"""
        q = self.store.q
        return {"t": time.time(),
                "levels": [dict(r) for r in q("SELECT level, type, state, COUNT(*) n, SUM(check_state='pass') checked "
                                              "FROM jobs GROUP BY level, type, state ORDER BY level")],
                "workers": [{"id": r["id"], "name": r["name"], "last_seen": r["last_seen"], "disabled": r["disabled"],
                             "gpu": (json.loads(r["caps"] or "{}")).get("gpu_name", "")} for r in q("SELECT * FROM workers ORDER BY id")],
                "active_leases": q("SELECT COUNT(*) FROM leases WHERE state='active'")[0][0],
                "spool_bytes": self.spool_bytes(),
                "events": [dict(r) for r in q("SELECT t, job_id, worker_id, kind FROM events ORDER BY id DESC LIMIT 50")]}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "rxm-queue/0"
    Q: Queue = None

    def log_message(self, fmt, *args):
        if self.Q.a.verbose: sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, obj=None, body=None, headers=()):
        b = body if body is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json" if body is None else "application/octet-stream")
        self.send_header("Content-Length", str(len(b)))
        for k, v in headers: self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > JSON_MAX: raise HTTPError(413, "body too large")
        try:
            b = json.loads(self.rfile.read(n) or b"{}")
        except ValueError: raise HTTPError(400, "bad json")
        if not isinstance(b, dict): raise HTTPError(400, "bad json")
        return b

    def _worker(self):
        a = self.headers.get("Authorization", "")
        w = self.Q.store.auth(a[7:] if a.startswith("Bearer ") else None)
        if w is None: raise HTTPError(401, "bad token")
        return w

    def _dispatch(self, method):
        try:
            u = urlsplit(self.path); parts = [unquote(x) for x in u.path.strip("/").split("/")]
            Q = self.Q
            if method == "GET" and parts == ["healthz"]: return self._send(200, Q.health())
            if method == "GET" and parts[:4] == ["api", API, "admin", "summary"] and len(parts) == 4:
                a = self.headers.get("Authorization", "")
                if Q.admin_hash is None: raise HTTPError(404, "no such endpoint")
                if not Q.admin_ok(a[7:] if a.startswith("Bearer ") else None): raise HTTPError(401, "bad admin token")
                return self._send(200, Q.admin_summary())
            if method == "GET" and parts[:1] == ["data"]:
                return self._data(self._worker(), "/".join(parts[1:]))
            if parts[:2] != ["api", API]: raise HTTPError(404, "no such endpoint")
            ep = parts[2:]
            if method == "POST" and ep == ["register"]:
                b = self._json(); name = str(b.get("name", ""))[:64]
                if not NAME_RE.match(name): raise HTTPError(400, "bad name")
                r = Q.store.register(str(b.get("invite", "")), name)
                if r is None: raise HTTPError(403, "bad invite")
                return self._send(200, {"worker_id": r[0], "token": r[1]})
            w = self._worker()
            if method == "POST" and ep == ["lease"]: return self._send(200, Q.lease(w, self._json().get("caps", {})))
            if method == "POST" and ep == ["heartbeat"]: return self._send(200, Q.heartbeat(w, self._json()))
            if method == "POST" and ep == ["complete"]: return self._send(200, Q.complete(w, self._json()))
            if method == "POST" and ep == ["fail"]: return self._send(200, Q.fail(w, self._json()))
            if method == "GET" and ep == ["status"]: return self._send(200, Q.status())
            if ep[:1] == ["upload"] and len(ep) == 4:
                job_id, lease_id, name = int(ep[1]), ep[2], ep[3]
                if not re.fullmatch("[0-9a-f]{32}", lease_id): raise HTTPError(400, "bad lease id")
                if method == "GET": return self._send(200, Q.upload_size(w, job_id, lease_id, name))
                if method == "PUT":
                    off = int(parse_qs(u.query).get("offset", ["0"])[0])
                    return self._send(200, Q.upload(w, job_id, lease_id, name, off, int(self.headers.get("Content-Length") or 0), self.rfile))
            raise HTTPError(404, "no such endpoint")
        except HTTPError as e:
            if method == "PUT" and e.code != 200: self.close_connection = True    # unread body
            self._send(e.code, {"error": e.msg, **e.extra})
        except (KeyError, ValueError, TypeError) as e:
            self.close_connection = True
            self._send(400, {"error": f"bad request: {e!r}"[:300]})
        except Exception as e:
            traceback.print_exc(); self.close_connection = True
            self._send(500, {"error": "internal"})

    def _data(self, w, rel):
        p = self.Q.data_path(rel)
        if not self.Q.may_read(w, rel): raise HTTPError(403, "no active lease lists this file")
        size = os.path.getsize(p); a, b = 0, size - 1; code = 200
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng.strip())
            if not m: raise HTTPError(416, "bad range")
            a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else size - 1
            b = min(b, size - 1)
            if a > b: raise HTTPError(416, "range not satisfiable")
            code = 206
        n = b - a + 1
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream"); self.send_header("Content-Length", str(n))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206: self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers(); self.wfile.flush()
        with open(p, "rb") as f:
            self.connection.sendfile(f, a, n)

    def do_GET(self): self._dispatch("GET")
    def do_POST(self): self._dispatch("POST")
    def do_PUT(self): self._dispatch("PUT")


def env(name, default=None, conv=str):
    v = os.environ.get("RXMQ_" + name)
    return default if v in (None, "") else conv(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=env("DB")); ap.add_argument("--data", default=env("DATA"))
    ap.add_argument("--spool", default=env("SPOOL")); ap.add_argument("--results", default=env("RESULTS"))
    ap.add_argument("--bind", default=env("BIND", "127.0.0.1")); ap.add_argument("--port", type=int, default=env("PORT", 8740, int))
    ap.add_argument("--lease-s", type=float, default=env("LEASE_S", 120, float), help="lease length; every heartbeat renews it")
    ap.add_argument("--heartbeat-s", type=float, default=env("HEARTBEAT_S", 30, float))
    ap.add_argument("--retry-s", type=float, default=env("RETRY_S", 30, float), help="what an idle worker is told to wait")
    ap.add_argument("--max-leases", type=int, default=env("MAX_LEASES", 1, int), help="active leases per worker")
    ap.add_argument("--chunk-max-mb", type=float, default=env("CHUNK_MAX_MB", CHUNK_MAX >> 20, float), help="largest upload PUT")
    ap.add_argument("--upload-max-mb", type=float, default=env("UPLOAD_MAX_MB", 0, float),
                    help="ceiling per uploaded file on top of each job type's own caps (0: the type caps only)")
    ap.add_argument("--spool-max-gb", type=float, default=env("SPOOL_MAX_GB", 0, float),
                    help="refuse upload chunks (507, the worker retries) while the spool holds this much (0: no cap)")
    ap.add_argument("--admin-token-file", default=env("ADMIN_TOKEN_FILE"),
                    help="enables the read-only GET /api/v0/admin/summary for this bearer token")
    ap.add_argument("--verbose", action="store_true", default=env("VERBOSE", False, lambda v: v not in ("0", "false", "no")))
    a = ap.parse_args()
    miss = [f"--{k} (RXMQ_{k.upper()})" for k in ("db", "data", "spool", "results") if not getattr(a, k)]
    if miss: ap.error("missing " + ", ".join(miss))
    Handler.Q = Queue(a)
    def reaper():
        while True:
            time.sleep(min(5.0, a.lease_s / 4))
            try: Handler.Q.store.expire()
            except Exception: traceback.print_exc()
    threading.Thread(target=reaper, daemon=True).start()
    srv = ThreadingHTTPServer((a.bind, a.port), Handler); srv.daemon_threads = True
    # SIGTERM (docker stop, systemd): stop accepting, let serve_forever return. Every state change is its own SQLite
    # transaction and uploads resume from the spool, so even a SIGKILL loses nothing but in-flight requests.
    def on_term(signum, _):
        print(f"signal {signum}: shutting down", flush=True)
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    print(f"queue on http://{a.bind}:{srv.server_address[1]}  db={a.db}  data={a.data}  results={a.results}  "
          f"admin summary {'on' if a.admin_token_file else 'off'}", flush=True)
    srv.serve_forever()
    srv.server_close()
    with Handler.Q.store.lock: Handler.Q.store.db.close()
    print("stopped", flush=True)


if __name__ == "__main__":
    main()
