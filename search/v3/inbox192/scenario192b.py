# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""scenario192b.py -- the fixed run behind the identity test of test_inbox192b.py (#192-02, w174): Driver.step() x 4 on the fake hosts
of confirm189 with inbox files waiting (a decide, a probe, a bad file, a file that is still being written) and NO round-robin keys in
CONF. Run it on HEAD and on this tree; the JSON it writes (time stamps dropped, tmp paths replaced) must be equal:

  python3 -I search/v3/inbox192/scenario192b.py --tree TREE --out FILE

TREE: a directory with search/v3 (the tree under test; its confirm189 harness is the one imported)."""
import argparse, json, os, re, sys, tempfile, time

ap = argparse.ArgumentParser(); ap.add_argument("--tree", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.abspath(a.tree), "search", "v3", "confirm189"))
import test_confirm189 as C    # noqa: E402  (the harness of the tree under test)
import menu as MN              # noqa: E402

assert os.path.realpath(C.V3).startswith(os.path.realpath(a.tree)), C.V3
STAMP = re.compile(r"^\d\d-\d\d \d\d:\d\d:\d\d CST ")


def fill(D, size, n=0):
    """an assignment in the size's window (as test_inbox192.fill): the uniform level under the top, trunk tensors bumped while in the window"""
    lo, hi = D.window(size); M = D.M
    under = [x for x in M.levels() if M.bytes(M.uniform(x)) <= hi]
    a_ = M.uniform(under[-1])
    order = [t for t in M.T if t["w"] not in (MN.HEAD, MN.EMB)]; order = order[n:] + order[:n]
    changed = True
    while changed:
        changed = False
        for t in order:
            names = [o["o"] for o in t["opts"]]; i = names.index(a_[t["w"]])
            if i + 1 < len(names):
                nb = M.bytes(a_) - M.opt(t["w"], a_[t["w"]])["bytes"] + M.opt(t["w"], names[i + 1])["bytes"]
                if nb <= hi: a_[t["w"]] = names[i + 1]; changed = True
    return a_


def drop(D, name, body, mtime):
    p = f"{D.R}/inbox/{name}"; json.dump(body, open(p, "w")); os.utime(p, (mtime, mtime))


with tempfile.TemporaryDirectory() as tmp:
    cp = C.make_env(tmp, nd=24); D, H = C.boot(cp)
    now = time.time()
    drop(D, "a.json", {"size": 2.0, "mode": "decide", "assign": fill(D, 2.0, 1), "note": "s"}, now - 400)
    drop(D, "p.json", {"size": 2.25, "mode": "probe", "assign": fill(D, 2.25, 2), "note": "s"}, now - 300)
    drop(D, "r.json", {"size": 2.0, "mode": "decide", "assign": {"blk.99.nope.weight": "x"}}, now - 200)
    with open(f"{D.R}/inbox/junk.json", "w") as f: f.write("{not json")                  # written just now: still being written
    C.steps(D, H, 4)
    ev = [dict({k: v for k, v in e.items() if k != "t"}) for e in C.events(D)]
    log = [STAMP.sub("", l.rstrip("\n")) for l in open(f"{D.R}/driver.log")]
    status = STAMP.sub("", open(f"{D.R}/status.txt").read().rstrip("\n"))
    box = {d: sorted(f for f in os.listdir(f"{D.R}/inbox/{d}") if f.endswith(".json")) if os.path.isdir(f"{D.R}/inbox/{d}") else [] for d in ("", "done", "rejected")}
    out = {"events": ev, "queued": H.queued(), "log": log, "status": status, "inbox": box}
    text = json.dumps(out, sort_keys=True).replace(tmp, "<tmp>")
    open(a.out, "w").write(text)
    print(f"{len(ev)} events, {len(out['queued'])} queued")
