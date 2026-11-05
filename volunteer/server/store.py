# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The queue's state: one SQLite file (WAL), shared by the HTTP server and the admin CLI.

jobs.state: queued -> leased -> accepted, or failed (max_attempts used up / admin), or cancelled (admin).
A lease is one attempt by one worker; leases.state: active -> done (its result accepted) | expired (missed its
heartbeat) | abandoned (the worker gave it back) | failed (the worker reported an error) | rejected (the worker
refused the job, or the server refused its result) | superseded (another lease's result was accepted first)."""
import hashlib, json, os, secrets, sqlite3, threading, time

SCHEMA = """
CREATE TABLE IF NOT EXISTS workers(id INTEGER PRIMARY KEY, name TEXT NOT NULL, token_hash TEXT UNIQUE NOT NULL,
  created REAL, last_seen REAL, caps TEXT, disabled INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS invites(code_hash TEXT PRIMARY KEY, uses_left INTEGER NOT NULL, created REAL, note TEXT);
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY, type TEXT NOT NULL, version INTEGER NOT NULL, params TEXT NOT NULL,
  inputs TEXT NOT NULL, requires TEXT NOT NULL, input_bytes INTEGER NOT NULL, state TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 5, level TEXT, ukey TEXT UNIQUE,
  lease_id TEXT, lease_owner INTEGER, lease_expires REAL, created REAL, updated REAL,
  accepted_lease TEXT, accepted_by INTEGER, accepted_at REAL, result TEXT, files TEXT, error TEXT,
  check_state TEXT, check_detail TEXT);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, priority, id);
CREATE INDEX IF NOT EXISTS jobs_level ON jobs(level);
CREATE TABLE IF NOT EXISTS leases(id TEXT PRIMARY KEY, job_id INTEGER NOT NULL, worker_id INTEGER NOT NULL, issued REAL,
  expires REAL, state TEXT NOT NULL, ended REAL, note TEXT);
CREATE INDEX IF NOT EXISTS leases_worker ON leases(worker_id, state);
CREATE INDEX IF NOT EXISTS leases_job ON leases(job_id);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, t REAL, job_id INTEGER, worker_id INTEGER, lease_id TEXT,
  kind TEXT, detail TEXT);
"""


def h(s): return hashlib.sha256(s.encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    # ------------------------------------------------------------ plumbing
    def tx(self):
        store = self
        class T:
            def __enter__(s): store.lock.acquire(); store.db.execute("BEGIN IMMEDIATE"); return store.db
            def __exit__(s, et, ev, tb):
                try: store.db.execute("COMMIT" if et is None else "ROLLBACK")
                finally: store.lock.release()
        return T()

    def q(self, sql, args=()):
        with self.lock: return self.db.execute(sql, args).fetchall()

    def event(self, db, kind, job_id=None, worker_id=None, lease_id=None, detail=None):
        db.execute("INSERT INTO events(t, job_id, worker_id, lease_id, kind, detail) VALUES(?,?,?,?,?,?)",
                   (time.time(), job_id, worker_id, lease_id, kind, None if detail is None else json.dumps(detail)))

    # ------------------------------------------------------------ workers
    def add_invite(self, uses=1, note=""):
        code = secrets.token_urlsafe(18)
        with self.tx() as db: db.execute("INSERT INTO invites VALUES(?,?,?,?)", (h(code), uses, time.time(), note))
        return code

    def register(self, invite, name):
        token = secrets.token_urlsafe(32)
        with self.tx() as db:
            r = db.execute("SELECT uses_left FROM invites WHERE code_hash=?", (h(invite),)).fetchone() if invite else None
            if r is None or r["uses_left"] <= 0: return None
            db.execute("UPDATE invites SET uses_left=uses_left-1 WHERE code_hash=?", (h(invite),))
            cur = db.execute("INSERT INTO workers(name, token_hash, created) VALUES(?,?,?)", (name, h(token), time.time()))
            self.event(db, "register", worker_id=cur.lastrowid, detail={"name": name})
        return cur.lastrowid, token

    def add_worker(self, name):
        token = secrets.token_urlsafe(32)
        with self.tx() as db:
            cur = db.execute("INSERT INTO workers(name, token_hash, created) VALUES(?,?,?)", (name, h(token), time.time()))
        return cur.lastrowid, token

    def auth(self, token):
        if not token: return None
        r = self.q("SELECT * FROM workers WHERE token_hash=? AND disabled=0", (h(token),))
        return r[0] if r else None

    # ------------------------------------------------------------ jobs
    def enqueue(self, db, jtype, version, params, inputs, requires, input_bytes, level=None, ukey=None, priority=0,
                max_attempts=5):
        if ukey and db.execute("SELECT 1 FROM jobs WHERE ukey=?", (ukey,)).fetchone(): return None
        t = time.time()
        cur = db.execute("INSERT INTO jobs(type, version, params, inputs, requires, input_bytes, state, priority, "
                         "max_attempts, level, ukey, created, updated) VALUES(?,?,?,?,?,?,'queued',?,?,?,?,?,?)",
                         (jtype, version, json.dumps(params, sort_keys=True), json.dumps(inputs), json.dumps(requires),
                          input_bytes, priority, max_attempts, level, ukey, t, t))
        self.event(db, "enqueue", job_id=cur.lastrowid)
        return cur.lastrowid

    def expire(self, now=None):
        """leases that missed their heartbeat: lease expired, job back to the queue (or failed: no attempts left)"""
        now = now or time.time(); n = 0
        with self.tx() as db:
            for l in db.execute("SELECT * FROM leases WHERE state='active' AND expires<?", (now,)).fetchall():
                db.execute("UPDATE leases SET state='expired', ended=? WHERE id=?", (now, l["id"]))
                j = db.execute("SELECT * FROM jobs WHERE id=?", (l["job_id"],)).fetchone()
                if j and j["state"] == "leased" and j["lease_id"] == l["id"]:
                    st = "failed" if j["attempts"] >= j["max_attempts"] else "queued"
                    db.execute("UPDATE jobs SET state=?, lease_id=NULL, lease_owner=NULL, lease_expires=NULL, updated=?, "
                               "error=COALESCE(error, '') || ? WHERE id=?", (st, now, f"[lease {l['id'][:8]} expired]", j["id"]))
                self.event(db, "expire", l["job_id"], l["worker_id"], l["id"]); n += 1
        return n

    def end_lease(self, db, lease, state, job_state=None, error=None, count_attempt=True):
        now = time.time()
        db.execute("UPDATE leases SET state=?, ended=?, note=? WHERE id=?", (state, now, error, lease["id"]))
        j = db.execute("SELECT * FROM jobs WHERE id=?", (lease["job_id"],)).fetchone()
        if j["state"] == "leased" and j["lease_id"] == lease["id"]:
            att = j["attempts"] - (0 if count_attempt else 1)
            st = job_state or ("failed" if att >= j["max_attempts"] else "queued")
            db.execute("UPDATE jobs SET state=?, attempts=?, lease_id=NULL, lease_owner=NULL, lease_expires=NULL, updated=?, "
                       "error=? WHERE id=?", (st, att, now, (error or "")[:2000], j["id"]))
