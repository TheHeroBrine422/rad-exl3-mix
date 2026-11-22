#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qq.py -- the global QAD queue (w222). Queue root ROOT holds jobs.txt (one job per line, taken in order), claims/ (KEY -> "CARD PID HOST TIME"), done/ (KEY -> "ok|cancelled RC TIME"),
failed/ (KEY -> "RC TIME"), cancel/ (KEY: an empty file cancels a job), runs/, scores/. A job is skipped when it is done (marker, or all outputs exist), failed, cancelled, claimed by a live
dispatcher, or not ready (a missing `@needs` path / checkpoint). Job lines (shlex syntax, `#` comments):
  T ARM SIZE BUDGETS READS flags...   train (qad220.py, budgets = comma list, READS = comma list of ckpt dir names read on j217, or -), then the reads
  R ARM SIZE CKPTDIR                  read-only: the served read of an existing checkpoint dir (absolute, or relative to runs/); score name ARM-<basename CKPTDIR>
  B ARM SIZE flags...                 short run (bench / gate), no reads; done when the trainer exits 0
  U ARM BPW                           the served read of a RELEASED container as it is (BPW = 2.00 / 3.50 / 5.00 ...; downloaded from the HF release into QROOT/rel/host-emb when missing), score name ARM (w222 fragility test)
  M ARM SIZE BUDGETS TAPDIR flags...  two-card job (w222: the teacher tap pass, `--make-taps TAPDIR`): claimed only by a dispatcher on a card in @cards=A[,B..], which also takes the lock of @partner=P (its
                                      other card; the partner's dispatcher takes no new job while the claim is live, so P frees up) and gives the trainer both cards; done when TAPDIR/.done exists
  directives anywhere after the fixed fields:  @cards=N,N  @partner=N  (M jobs)  @needs=PATH[,PATH..] (absolute, or relative to runs/; the job waits until all exist)   @budgets=B (B lines: the trainer's --budgets, default 2.5e5,5e5)   `{lam}` / `{lam*0.3}` in a line = the first number of QROOT/lambda.txt (x the factor; the line also waits for that file)   @ctl=ARM (control arm for the paired column)
  `{runs}` in any field expands to ROOT/runs.
usage: qq.py peek ROOT [CARD] | claim ROOT CARD PID [--caps M] | unclaim ROOT KEY | finish ROOT KEY ok|fail|cancelled RC | status ROOT | key LINE..."""
import fcntl, os, shlex, socket, sys, time


def parse(line, root):
    lamf = f"{root}/lambda.txt"; haslam = "{lam" in line
    if haslam and os.path.exists(lamf):                         # {lam} = the rule's lambda, {lam*0.3} = a multiple of it
        import re
        lv = float(open(lamf).read().split()[0]); line = re.sub(r"\{lam(?:\*([0-9.eE+-]+))?\}", lambda m: "%.6g" % (lv * float(m.group(1) or 1)), line)
    t = shlex.split(line.replace("{runs}", f"{root}/runs").replace("{ext}", f"{root}/ext"), comments=True)
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
    elif kind == "U":
        assert len(rest) == 2, f"U needs ARM BPW: {line}"
        j = dict(kind="U", arm=rest[0], size=rest[1], budgets=None, reads=[], flags=[], key=f"U-{rest[0]}")
    elif kind == "M":
        assert len(rest) >= 4, f"M needs ARM SIZE BUDGETS TAPDIR: {line}"
        arm, size, bud, td = rest[:4]; j = dict(kind="M", arm=arm, size=size, budgets=bud, reads=[], ckptdir=td, flags=rest[4:], key=f"M-{arm}")
        assert "cards" in d and "partner" in d, f"M needs @cards= and @partner=: {line}"
    elif kind == "B":
        assert len(rest) >= 2, f"B needs ARM SIZE: {line}"
        j = dict(kind="B", arm=rest[0], size=rest[1], budgets=d.get("budgets", "2.5e5,5e5"), reads=[], flags=rest[2:], key=f"B-{rest[0]}")
    else: raise ValueError(f"unknown job kind {kind}: {line}")
    j["needs"] = d.get("needs"); j["ctl"] = d.get("ctl"); j["cards"] = d["cards"].split(",") if "cards" in d else None; j["partner"] = d.get("partner")
    j["needs"] = [x if x.startswith("/") else f"{root}/runs/{x}" for x in j["needs"].split(",")] if j["needs"] else []
    if haslam: j["needs"].append(lamf)            # a {lam} line waits for QROOT/lambda.txt (written once from the gradient-noise pass)
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
    if j["kind"] == "M": return os.path.exists(f"{j['ckptdir']}/.done")
    if j["kind"] == "U": return os.path.exists(f"{root}/scores/{j['arm']}.json")
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
    need = j["needs"] or ([f"{j['ckptdir']}/.done"] if j["kind"] == "R" else [])
    miss = [x for x in need if not os.path.exists(x)]
    if miss: return "waiting", "needs " + ", ".join(miss)
    return ("stale" if stale else "ready"), ("claim of a dead dispatcher (resume)" if stale else "")


def reserved(root):
    """cards held for a live M claim's partner: their dispatchers take no new job"""
    out = set()
    for j in jobs(root):
        if j["kind"] == "M":
            cf = f"{root}/claims/{j['key']}"
            if os.path.exists(cf):
                c = open(cf).read().split()
                if len(c) >= 3 and not (c[2] == socket.gethostname() and not alive(int(c[1]))): out.add(j["partner"])
    return out


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
        card = a[2] if a[0] == "claim" else (a[2] if len(a) > 2 else None)
        caps = set(a[a.index("--caps") + 1].split(",")) if "--caps" in a else set()
        if card is not None and card in reserved(root): return 1
        for j in jobs(root):
            s, _ = state(root, j)
            if s not in ("ready", "stale"): continue
            if j["cards"] is not None and card is not None and card not in j["cards"]: continue
            if j["kind"] == "M" and a[0] == "claim" and "M" not in caps: continue
            if a[0] == "claim":
                open(f"{root}/claims/{j['key']}", "w").write(f"{a[2]} {a[3]} {socket.gethostname()} {time.strftime('%F %T')}\n")
                line = [j["kind"], j["key"], j["arm"], j["size"], j.get("budgets") or "-", ",".join(j.get("reads", [])) or "-", j.get("ckptdir") or "-", j["ctl"] or "-", s] + (["partner=" + j["partner"]] if j["kind"] == "M" else []) + j["flags"]
                print(" ".join(shlex.quote(x) for x in line))
            return 0
        return 1
    if a[0] == "unclaim":
        lk = locked(root)
        try: os.remove(f"{root}/claims/{a[2]}")
        except OSError: pass
        return 0
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
