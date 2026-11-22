#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qq.py -- the global QAD queue (w222). Queue root ROOT holds jobs.txt (one job per line, taken in order), claims/ (KEY -> "CARD PID HOST TIME"), done/ (KEY -> "ok|cancelled RC TIME"),
failed/ (KEY -> "RC TIME"), cancel/ (KEY: an empty file cancels a job), runs/, scores/. A job is skipped when it is done (marker, or all outputs exist), failed, cancelled, claimed by a live
dispatcher, or not ready (a missing `@needs` path / checkpoint). Job lines (shlex syntax, `#` comments):
  T ARM SIZE BUDGETS READS flags...   train (qad220.py, budgets = comma list, READS = comma list of ckpt dir names read on j217, or -), then the reads
  R ARM SIZE CKPTDIR                  read-only: the served read of an existing checkpoint dir (absolute, or relative to runs/); score name ARM-<basename CKPTDIR>
  B ARM SIZE flags...                 short run (bench / gate), no reads; done when the trainer exits 0
  directives anywhere after the fixed fields:  @needs=PATH (absolute, or relative to runs/; the job waits until it exists)   @ctl=ARM (control arm for the paired column)
  `{runs}` in any field expands to ROOT/runs.
usage: qq.py peek ROOT | claim ROOT CARD PID | finish ROOT KEY ok|fail|cancelled RC | status ROOT | key LINE..."""
import fcntl, os, shlex, socket, sys, time


def parse(line, root):
    t = shlex.split(line.replace("{runs}", f"{root}/runs"), comments=True)
    if not t: return None
    kind, rest = t[0], t[1:]
    dirs = [x for x in rest if x.startswith("@")]; rest = [x for x in rest if not x.startswith("@")]
    d = dict(x[1:].split("=", 1) for x in dirs if "=" in x)
    if kind == "T":
        assert len(rest) >= 4, f"T needs ARM SIZE BUDGETS READS: {line}"
        arm, size, bud, reads = rest[:4]; j = dict(kind="T", arm=arm, size=size, budgets=bud, reads=[] if reads == "-" else reads.split(","), flags=rest[4:], key=f"T-{arm}")
    elif kind == "R":
        assert len(rest) == 3, f"R needs ARM SIZE CKPT: {line}"
        arm, size, ck = rest; ck = ck if ck.startswith("/") else f"{root}/runs/{ck}"
        j = dict(kind="R", arm=arm, size=size, ckptdir=ck, flags=[], key=f"R-{arm}-{os.path.basename(ck)}")
    elif kind == "B":
        assert len(rest) >= 2, f"B needs ARM SIZE: {line}"
        j = dict(kind="B", arm=rest[0], size=rest[1], budgets="2.5e5,5e5", reads=[], flags=rest[2:], key=f"B-{rest[0]}")
    else: raise ValueError(f"unknown job kind {kind}: {line}")
    j["needs"] = d.get("needs"); j["ctl"] = d.get("ctl")
    if j["needs"] and not j["needs"].startswith("/"): j["needs"] = f"{root}/runs/{j['needs']}"
    return j


def jobs(root):
    out, seen = [], set()
    for ln in open(f"{root}/jobs.txt"):
        if not ln.strip() or ln.lstrip().startswith("#"): continue
        j = parse(ln, root)
        if j is None or j["key"] in seen: continue
        seen.add(j["key"]); out.append(j)
    return out


def last_ckpt(budgets): return "ckpt-%.0e" % float(budgets.split(",")[-1])


def outputs_done(root, j):
    if j["kind"] == "R": return os.path.exists(f"{root}/scores/{j['arm']}-{os.path.basename(j['ckptdir'])}.json")
    if j["kind"] == "B": return False
    if not os.path.exists(f"{root}/runs/{j['arm']}/{last_ckpt(j['budgets'])}/.done"): return False
    return all(os.path.exists(f"{root}/scores/{j['arm']}-{ck}.json") for ck in j["reads"])


def alive(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f: return b"qcard.sh" in f.read()
    except OSError: return False


def state(root, j):
    """('done'|'failed'|'cancelled'|'running'|'stale'|'waiting'|'ready', detail)"""
    k = j["key"]
    if os.path.exists(f"{root}/done/{k}"): return "done", open(f"{root}/done/{k}").read().strip()
    if os.path.exists(f"{root}/cancel/{k}"): return "cancelled", ""
    if os.path.exists(f"{root}/failed/{k}"): return "failed", open(f"{root}/failed/{k}").read().strip()
    if outputs_done(root, j): return "done", "outputs exist"
    cf = f"{root}/claims/{k}"
    stale = False
    if os.path.exists(cf):
        c = open(cf).read().split()
        if len(c) >= 3 and c[2] == socket.gethostname() and not alive(int(c[1])): stale = True
        else: return "running", f"card {c[0]} pid {c[1]}"
    need = j["needs"] or (f"{j['ckptdir']}/.done" if j["kind"] == "R" else None)
    if need and not os.path.exists(need): return "waiting", f"needs {need}"
    return ("stale" if stale else "ready"), ("claim of a dead dispatcher (resume)" if stale else "")


def locked(root):
    f = open(f"{root}/queue.lock", "a"); fcntl.flock(f, fcntl.LOCK_EX); return f


def main():
    a = sys.argv[1:]
    if a[0] == "key":
        print(parse(" ".join(a[1:]), "ROOT")["key"]); return 0
    root = a[1]
    for d in ("claims", "done", "failed", "cancel", "runs", "scores"): os.makedirs(f"{root}/{d}", exist_ok=True)
    if a[0] in ("peek", "claim"):
        lk = locked(root)
        for j in jobs(root):
            s, _ = state(root, j)
            if s not in ("ready", "stale"): continue
            if a[0] == "claim":
                open(f"{root}/claims/{j['key']}", "w").write(f"{a[2]} {a[3]} {socket.gethostname()} {time.strftime('%F %T')}\n")
                line = [j["kind"], j["key"], j["arm"], j["size"], j.get("budgets") or "-", ",".join(j.get("reads", [])) or "-", j.get("ckptdir") or "-", j["ctl"] or "-", s] + j["flags"]
                print(" ".join(shlex.quote(x) for x in line))
            return 0
        return 1
    if a[0] == "finish":
        k, what, rc = a[2], a[3], a[4]
        lk = locked(root)
        if what == "fail": open(f"{root}/failed/{k}", "w").write(f"rc {rc} {time.strftime('%F %T')}\n")
        else: open(f"{root}/done/{k}", "w").write(f"{what} rc {rc} {time.strftime('%F %T')}\n")
        try: os.remove(f"{root}/claims/{k}")
        except OSError: pass
        return 0
    if a[0] == "status":
        for i, j in enumerate(jobs(root), 1):
            s, d = state(root, j); print(f"{i:3d} {s:9s} {j['key']:44s} {d}")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__": sys.exit(main())
