# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""python -m rxrun.multi CONF [--status] [--follow] -- one queue over several hosts' cards (w120, 11-12; w105 README
"Hosts"). Runs where every host is reachable (the VM: box2 is only reachable through its reverse forward) and stays
light: it holds one ssh session a host, a JSON state file and the queue; all card / CPU work happens on the hosts.

Units (the queue, in order of preference):
  eval   one line of CONF "evals" (a jsonl the search appends to: {"tag", "mix", "slices"}, hostd.py's eval unit). An
         eval goes only to the host that holds every level its mix uses (level dirs stay where they were built:
         box2's uplink is not ours) -- an eval whose levels sit on two hosts is reported as blocked.
  level  the next ladder levels (settings file's rxq.ladder.levels, CONF passes / levels subset, ladder order), up to
         the host's unit_levels in one unit (one rxrun.run there, one swj pass for the unit), within its limits
         (k_min / k_max, max_unit_est_h of the model's estimate). Each level is built on exactly one host.
Hosts (CONF "hosts"): {name, cmd: argv that starts `python -m rxrun.hostd HOSTCONF` there (ssh line; [] -> local, tests),
python, hostconf, unit_levels, slots {level, eval}, reconnect_s}. Plain ssh lines with stdin passed through (not
ssh-box2.sh, which base64-wraps the command and drops stdin). A refused connection to box2 (127.0.0.1:2241) means the
user's forward is down: the host stays disconnected and the others go on (status says so).
Same staged tree: every hello carries the settings stamp and hostd's tree stamp (sources of the staged tree + the
fork); a host whose stamps differ from the first host's (or CONF "expect") is refused: restage it (README "Multi-host").
Restart: ROOT/state.jsonl records assignments and results; units in flight go back to the host they were on (hostd
reports them running / finished; a unit process outlives a dropped ssh). Results already recorded are not redone.
Hold: touch ROOT/HOLD (no new units; running ones finish). Status: ROOT/status.txt.
Outputs: ROOT/results.jsonl (one line per eval x slice: tag, bytes, bpw, positions, kld, p99, top1, slice, host, id,
t; failed evals: tag, fail, err), ROOT/docs/<tag>-<slice>.json (per-part per-document KL sums), ROOT/levels.jsonl
(level -> host, jobs, wall)."""
import argparse, json, os, queue, subprocess, sys, threading, time

from rxq import config as C, ladder

CST = -6 * 3600


def cst(t=None):
    return time.strftime("%m-%d %H:%M:%S", time.gmtime((t or time.time()) + CST)) + " CST"


def load(p):
    if p.endswith(".toml"):
        import tomllib
        return tomllib.load(open(p, "rb"))
    return json.load(open(p))


def mix_levels(mx, trunk):
    if "levels" in mx: return sorted(set(mx["levels"].values()))
    if "trunk" in mx: return [mx["trunk"]]
    return sorted({mx["parity"]["even"], mx["parity"]["odd"]})


class Conn:
    """one host's hostd session; events -> the coordinator's queue as (name, msg)"""

    def __init__(self, h, q):
        self.h, self.q, self.name = h, q, h["name"]
        self.p = None; self.hello = None; self.t_try = 0.0; self.refused = None

    def open(self):
        self.t_try = time.time()
        argv = list(self.h.get("cmd") or []) + [self.h.get("python", sys.executable), "-m", "rxrun.hostd", self.h["hostconf"]]
        try:
            self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            self.q.put((self.name, {"eof": True, "err": repr(e)})); return
        threading.Thread(target=self._read, args=(self.p,), daemon=True).start()
        threading.Thread(target=self._err, args=(self.p,), daemon=True).start()

    def _read(self, p):
        for line in p.stdout:
            try: self.q.put((self.name, json.loads(line)))
            except ValueError: pass
        p.wait(); self.q.put((self.name, {"eof": True, "rc": p.returncode, "err": "".join(self.errtail)[-300:]}))

    def _err(self, p):
        self.errtail = []
        for line in p.stderr:
            self.errtail = (self.errtail + [line])[-10:]

    def send(self, o):
        try: self.p.stdin.write(json.dumps(o) + "\n"); self.p.stdin.flush(); return True
        except (BrokenPipeError, OSError, AttributeError): return False

    def up(self): return self.p is not None and self.p.poll() is None and self.hello is not None and not self.refused

    def close(self):
        if self.p and self.p.poll() is None:
            self.send({"quit": True})
            try: self.p.wait(10)
            except subprocess.TimeoutExpired: self.p.kill()


class Multi:
    def __init__(self, conf, follow=False):
        self.c = conf; self.R = conf["root"]; os.makedirs(f"{self.R}/docs", exist_ok=True)
        self.follow = follow
        self.cfg = C.load(conf["settings"]); self.st = C.settings_stamp(self.cfg)
        passes = conf.get("passes", ["1", "2", "3"])
        lv = [v for v in ladder.levels(self.cfg) if v["pass"] in passes]
        if conf.get("levels"): lv = [v for v in lv if v["name"] in conf["levels"] or f"{ladder.fmt_k(v['K'])}-{v['codebook']}" in conf["levels"]]
        self.levels = lv; self.byname = {v["name"]: v for v in lv}
        self.est = {}
        self.q = queue.Queue(); self.conns = {h["name"]: Conn(h, self.q) for h in conf["hosts"]}
        self.ref = dict(conf.get("expect") or {})
        # state
        self.inflight = {}; self.done = {}; self.fail = {}; self.lfail = {}; self.loc = {}; self.blocked = set()
        sp = f"{self.R}/state.jsonl"
        if os.path.exists(sp):
            for l in open(sp):
                try: e = json.loads(l)
                except ValueError: continue
                if e["ev"] == "assign": self.inflight[e["id"]] = e
                elif e["ev"] == "result":
                    self.inflight.pop(e["id"], None)
                    if e["rc"] == 0: self.done[e["id"]] = e
                    else:
                        self.fail[e["id"]] = self.fail.get(e["id"], 0) + 1
                        for n in e.get("levels") or []: self.lfail[n] = self.lfail.get(n, 0) + 1
                elif e["ev"] == "level_at": self.loc[e["level"]] = e["host"]
        self.trunk = None

    # -------------------------------------------------------------------------------------------- bookkeeping
    def rec(self, e):
        e = dict(e, t=cst())
        with open(f"{self.R}/state.jsonl", "a") as f: f.write(json.dumps(e) + "\n")

    def log(self, msg):
        line = f"{cst()} {msg}"; print(line, flush=True)
        with open(f"{self.R}/multi.log", "a") as f: f.write(line + "\n")

    def level_est_h(self, v):
        if v["name"] not in self.est:
            self.est[v["name"]] = sum(j["params"]["req"]["est_s"] for j in ladder.level_jobs(self.cfg, v, "m", "r", None, layers=self.c.get("layers"))) / 3600
        return self.est[v["name"]]

    def evals(self):
        p = self.c.get("evals")
        if not p or not os.path.exists(p): return []
        out = []
        for l in open(p):
            try: e = json.loads(l)
            except ValueError: continue
            if "tag" in e: out.append(e)
        return out

    def held(self): return os.path.exists(f"{self.R}/HOLD")

    def levels_left(self):
        busy = {n for e in self.inflight.values() if e["kind"] == "level" for n in e["levels"]}
        return [v for v in self.levels if v["name"] not in self.loc and v["name"] not in busy]

    def busy(self, host, kind): return sum(1 for e in self.inflight.values() if e["host"] == host and e["kind"] == kind)

    # ----------------------------------------------------------------------------------------------- hosts
    def on_hello(self, cn, m):
        cn.hello = m
        mine = {"settings_stamp": m["settings_stamp"], "tree_stamp": m["tree_stamp"]}
        if m["settings_stamp"] != self.st:
            cn.refused = f"settings stamp {m['settings_stamp']} != the coordinator's {self.st}"
        else:
            for k, v in mine.items():
                if k in self.ref and self.ref[k] != v: cn.refused = f"{k} {v} != {self.ref[k]} (the first host's / CONF expect): restage"
            if not cn.refused: self.ref.update({k: v for k, v in mine.items() if k not in self.ref})
        if cn.refused:
            self.log(f"{cn.name}: REFUSED: {cn.refused}"); cn.close(); return
        for n in m.get("levels_done", []):
            if n in self.byname and self.loc.get(n) != cn.name:
                if n in self.loc: self.log(f"WARN level {n} DONE on {cn.name} and on {self.loc[n]}: keeping {self.loc[n]}"); continue
                self.loc[n] = cn.name; self.rec({"ev": "level_at", "level": n, "host": cn.name})
        self.log(f"{cn.name}: connected (pid {m['pid']}, settings {m['settings_stamp']}, tree {m['tree_stamp']} / {m.get('tree_files')} "
                 f"files, slots {m['slots']}, kinds {m['kinds']}, {len(m.get('levels_done', []))} levels here, units {m.get('units')})")
        for i, e in self.inflight.items():                  # resend: hostd ignores a unit it already has
            if e["host"] == cn.name and m.get("units", {}).get(i) != "finished": cn.send(e["req"])

    def on_result(self, cn, m):
        i = m["id"]; e = self.inflight.pop(i, None); r = m.get("result") or {}
        if e is None:
            if i in self.done: cn.send({"ack": i}); return
            e = {"id": i, "kind": r.get("kind", "?"), "host": cn.name}
        self.rec({"ev": "result", "id": i, "rc": m["rc"], "host": cn.name, "err": r.get("err"), "levels": e.get("levels")})
        if m["rc"] == 0:
            self.done[i] = e
            if e["kind"] == "level":
                with open(f"{self.R}/levels.jsonl", "a") as f:
                    for n, v in r.get("levels", {}).items():
                        self.loc[n] = cn.name; self.rec({"ev": "level_at", "level": n, "host": cn.name})
                        f.write(json.dumps({"level": n, "host": cn.name, **v, "t": cst()}) + "\n")
                self.log(f"{cn.name}: levels {list(r.get('levels', {}))} DONE ({r.get('wall', 0) / 3600:.2f} h)")
            elif e["kind"] == "eval":
                with open(f"{self.R}/results.jsonl", "a") as f:
                    for s, v in r.get("slices", {}).items():
                        docs = {p: pv.pop("docs", None) for p, pv in v["parts"].items()}
                        json.dump(docs, open(f"{self.R}/docs/{r['tag']}-{s}.json", "w"))
                        f.write(json.dumps({**v, "bpw": r.get("bpw"), "host": cn.name, "id": i, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
                self.log(f"{cn.name}: eval {r.get('tag')} done: " + ", ".join(f"{s} {v['kld']:.5f}" for s, v in r.get("slices", {}).items()))
        else:
            self.fail[i] = self.fail.get(i, 0) + 1
            for n in e.get("levels") or []: self.lfail[n] = self.lfail.get(n, 0) + 1
            self.log(f"{cn.name}: unit {i} FAILED ({self.fail[i]}x): {r.get('err')}")
            if e["kind"] == "eval" and self.fail[i] > int(self.c.get("retries", 1)):
                with open(f"{self.R}/results.jsonl", "a") as f:
                    f.write(json.dumps({"tag": e.get("tag"), "fail": True, "err": r.get("err"), "host": cn.name, "id": i}) + "\n")
        cn.send({"ack": i})

    # ----------------------------------------------------------------------------------------------- assign
    def assign(self, cn, kind, req, extra):
        e = {"ev": "assign", "id": req["id"], "host": cn.name, "kind": kind, "req": req, **extra}
        if not cn.send(req): return False
        self.inflight[req["id"]] = e; self.rec(e)
        self.log(f"{cn.name}: start {kind} {req['id']}")
        return True

    def schedule(self):
        if self.held(): return
        from rxq import mix as M
        if self.trunk is None: self.trunk = M.trunk_names()
        retries = int(self.c.get("retries", 1))
        for cn in self.conns.values():
            if not cn.up(): continue
            h = cn.h; m = cn.hello; slots = m["slots"]; lim = m.get("limits", {})
            # evals
            if "eval" in m["kinds"]:
                for ev in self.evals():
                    if self.busy(cn.name, "eval") >= int(slots.get("eval", 1)): break
                    i = f"ev-{ev['tag']}"
                    if i in self.done or i in self.inflight or self.fail.get(i, 0) > retries: continue
                    need = mix_levels(ev["mix"], self.trunk)
                    where = {self.loc.get(n) for n in need}
                    if None in where: continue                         # a level not built yet
                    if len(where) > 1:
                        if i not in self.blocked:
                            self.blocked.add(i); self.log(f"eval {ev['tag']} BLOCKED: its levels are on {sorted(where)} (copy level dirs first)")
                        continue
                    if where != {cn.name}: continue
                    self.assign(cn, "eval", dict(ev, id=i, kind="eval", attempt=self.fail.get(i, 0)), {"tag": ev["tag"]})
            # levels
            if "level" in m["kinds"] and self.busy(cn.name, "level") < int(slots.get("level", 1)):
                left = [v for v in self.levels_left() if lim.get("k_min", -1) <= v["K"] <= lim.get("k_max", 99)
                        and self.lfail.get(v["name"], 0) <= retries]
                take, h_est = [], 0.0
                for v in left:
                    e_h = self.level_est_h(v)
                    if take and (len(take) >= int(h.get("unit_levels", 1)) or h_est + e_h > float(lim.get("max_unit_est_h", 1e9))): break
                    take.append(v); h_est += e_h
                if take:
                    i = "lv-" + take[0]["name"] + (f"+{len(take) - 1}" if len(take) > 1 else "")
                    self.assign(cn, "level", {"id": i, "kind": "level", "levels": [v["name"] for v in take], "attempt": self.fail.get(i, 0)},
                                {"levels": [v["name"] for v in take], "est_h": round(h_est, 2)})

    def finished(self):
        if self.inflight: return False
        retries_ = int(self.c.get("retries", 1))
        if any(self.lfail.get(v["name"], 0) <= retries_ for v in self.levels_left()): return False
        retries = int(self.c.get("retries", 1))
        for ev in self.evals():
            i = f"ev-{ev['tag']}"
            if i not in self.done and self.fail.get(i, 0) <= retries and i not in self.blocked: return False
        return True

    def status(self):
        nd = sum(1 for v in self.levels if v["name"] in self.loc)
        hs = []
        for n, cn in self.conns.items():
            s = "up" if cn.up() else f"REFUSED ({cn.refused[:60]})" if cn.refused else "down"
            busy = [i for i, e in self.inflight.items() if e["host"] == n]
            hs.append(f"{n} {s} {busy}")
        line = (f"{cst()} | levels {nd}/{len(self.levels)} | evals done {sum(1 for i in self.done if i.startswith('ev-'))}, "
                f"failed {sum(1 for i, k in self.fail.items() if i.startswith('ev-'))}, blocked {len(self.blocked)} | " + " | ".join(hs)
                + (" | HOLD" if self.held() else ""))
        open(f"{self.R}/status.txt.part", "w").write(line + "\n"); os.replace(f"{self.R}/status.txt.part", f"{self.R}/status.txt")
        return line

    # ------------------------------------------------------------------------------------------------- main
    def run(self):
        self.log(f"rxrun.multi start: settings {self.st}, {len(self.levels)} levels, hosts {list(self.conns)}, "
                 f"{len(self.inflight)} units in flight from the state file")
        for cn in self.conns.values(): cn.open()
        last_st = 0.0
        try:
            while True:
                try: name, m = self.q.get(timeout=1.0)
                except queue.Empty: name, m = None, None
                if m is not None:
                    cn = self.conns[name]
                    if "hello" in m: self.on_hello(cn, m)
                    elif m.get("eof"):
                        if not cn.refused: self.log(f"{name}: disconnected (rc {m.get('rc')}; {m.get('err', '')[-200:].strip()}); units stay there")
                        cn.hello = None if not cn.refused else cn.hello
                    elif m.get("event") == "started": pass
                    elif "rc" in m and "id" in m: self.on_result(cn, m)
                for cn in self.conns.values():                    # reconnect
                    if not cn.refused and (cn.p is None or cn.p.poll() is not None) and time.time() - cn.t_try > float(cn.h.get("reconnect_s", 60)):
                        cn.hello = None; cn.open()
                self.schedule()
                if time.time() - last_st > 5: self.status(); last_st = time.time()
                if not self.follow and self.finished():
                    self.log("rxrun.multi: nothing left"); break
                if self.held() and not self.inflight:
                    self.log("HOLD: nothing in flight, exit"); break
        finally:
            self.status()
            for cn in self.conns.values(): cn.close()
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxrun.multi")
    ap.add_argument("conf"); ap.add_argument("--status", action="store_true"); ap.add_argument("--follow", action="store_true")
    a = ap.parse_args(argv)
    M = Multi(load(a.conf), follow=a.follow)
    if a.status: print(open(f"{M.R}/status.txt").read().strip() if os.path.exists(f"{M.R}/status.txt") else "no status yet"); return 0
    return M.run()


if __name__ == "__main__":
    sys.exit(main())
