# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_inbox192.py -- the driver inbox of the #183 search driver (#192, w171): ROOT/inbox/*.json, oldest file first.
"decide" (a manual candidate, judged like a generated one: stage 1 with the early stop, tentative -> confirmation, pool
rules) and "probe" (a whole read at any size: fit rows, no decision, no pool, no incumbent; optional extra slice reads, never
decisions or fit rows). Also: the checks and rejects (unknown weight / option, unassigned weight, window, probe tolerance, mode,
stopped size, extra slices, bad JSON, a reused name), restart safety (a recorded file is never resubmitted), inbox_max, the
oldest-first order, the status segment, and identity: with the inbox empty the driver emits the events of the HEAD driver on
the recorded #183 replay (w165 snapshot, as w167) and on a copy of the live run (dryrun192.py, as w168).

  python3 -I search/v3/inbox192/test_inbox192.py      (exit 0 iff every test passes; CPU, about a minute)

The driver runs its real loop (Driver.step) against the fake hosts of confirm189/test_confirm189.py (imported: env, host, boot,
identity). Baseline tree for the identity: RXM_POOL_BASE or `git archive HEAD` (see that file). RXM_LIVE_RUN: the live run dir
copied for the live identity (default /workspace/rad-local/workers/lead-srch/run); a missing one is SKIPPED."""
import json, math, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
sys.path.insert(0, V3); sys.path.insert(0, os.path.join(V3, "confirm189"))
import menu as MN          # noqa: E402
import test_confirm189 as C  # noqa: E402  (env, Host, boot, steps, identity: the same tree as this file)

T = []
LIVE = os.environ.get("RXM_LIVE_RUN", "/workspace/rad-local/workers/lead-srch/run")
T183 = "/data/q27/w131/slices/golden-t183"


def test(f): T.append(f); return f


# ------------------------------------------------------------------------------------------------ helpers
def env(tmp, inbox_max=6, **kw):
    """the w167 environment (K = 3, sizes 2.0 / 3.0, confirm pairs s186a -> s186b, s186c -> s186d) plus a staged slice t183 for the reads"""
    cp = C.make_env(tmp, **kw)
    conf = json.load(open(cp)); conf["inbox_max"] = inbox_max
    reg = json.load(open(conf["registry"]))
    reg["slices"]["t183"] = {"parts": {}, "mix": "/gold/t183", "alias": T183, "host_dirs": {}, "docs": C.slice_docs("t183", 24, 9),
                             "status": "staged", "decisions": {}, "built": "2026-11-15T00:00:00Z"}
    json.dump(reg, open(conf["registry"], "w")); json.dump(conf, open(cp, "w"))
    return cp


def fill(D, size, n=0):
    """an assignment whose menu bytes lie in the size's window (driver.window): the uniform level under the top, then trunk tensors
    bumped up one option while the bytes stay under the top (n rotates the order: distinct assignments)"""
    lo, hi = D.window(size); M = D.M
    under = [x for x in M.levels() if M.bytes(M.uniform(x)) <= hi]
    a = M.uniform(under[-1])
    order = [t for t in M.T if t["w"] not in (MN.HEAD, MN.EMB)]; order = order[n:] + order[:n]
    changed = True
    while changed:
        changed = False
        for t in order:
            names = [o["o"] for o in t["opts"]]; i = names.index(a[t["w"]])
            if i + 1 < len(names):
                nb = M.bytes(a) - M.opt(t["w"], a[t["w"]])["bytes"] + M.opt(t["w"], names[i + 1])["bytes"]
                if nb <= hi: a[t["w"]] = names[i + 1]; changed = True
    assert lo <= M.bytes(a) <= hi, (size, M.bpw(a))
    return a


def body(D, size, mode="decide", a=None, **kw):
    return dict({"size": size, "assign": a if a is not None else fill(D, size), "mode": mode, "note": "test"}, **kw)


def drop(D, name, b, mtime=None):
    p = f"{D.R}/inbox/{name}"; json.dump(b, open(p, "w"))
    if mtime is not None: os.utime(p, (mtime, mtime))
    return p


def events(D): return C.events(D)


def log(D): return open(f"{D.R}/driver.log").read()


def inbox_tags(D, kind):
    return [t for t, e in D.sub.items() if e["kind"] == kind]


# ------------------------------------------------------------------------------------------------ paths
@test
def decide_tentative_then_confirmed():
    """a decide file (size 2.0, in the window) is submitted as kind manual ahead of the driver's own moves, tag <prefix>-2-manual-<n>,
    the usual early-stop block; its stage-1 accept is tentative and confirmed on s186b exactly as a generated candidate (w167):
    ACCEPT (confirmed), the manual's tag is the incumbent, the stage-2 decision carries both log r's; the file is in done/ and the
    inbox state event (name, tag, mode) is recorded before the move"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp); D, H = C.boot(cp); I = D.inc[2.0]
        drop(D, "a.json", body(D, 2.0, a=fill(D, 2.0, 1)))
        C.quiet(D.step)                                                    # submitted
        m = inbox_tags(D, "manual"); assert len(m) == 1 and m[0].startswith("t189-2-manual-"), m
        A = m[0]; H.set(D, A, s186a=0.90, s186b=0.94)
        ev = events(D); ib = [e for e in ev if e["ev"] == "inbox"]
        assert len(ib) == 1 and ib[0]["tag"] == A and ib[0]["mode"] == "decide" and ib[0]["file"] == "a.json", ib
        assert os.path.exists(f"{D.R}/inbox/done/a.json") and not os.path.exists(f"{D.R}/inbox/a.json")
        assert [e["ev"] for e in ev].index("inbox") > [e["ev"] for e in ev].index("submit"), "the record comes after the submit"
        se = [e for e in ev if e["ev"] == "submit" and e["tag"] == A][0]
        assert se["slice"] == "s186a" and se["early"] == (not se.get("audit")) and se["ref"] == I, se
        C.steps(D, H)                                                      # stage 1 (the host scores A; the verdict is tentative)
        assert A in D.tent and D.inc[2.0] == I and D.reg.d["slices"]["s186a"]["decisions"] == {"2.0": 1}, (D.tent, D.inc)
        ce = C.confirm_evals(D)
        assert len(ce) == 2 and {D.sub[x]["of"] for x in ce} == {A, I} and all(D.sub[x]["slice"] == "s186b" for x in ce), ce
        C.steps(D, H)                                                      # stage 2
        assert not D.tent and D.inc[2.0] == A, (D.inc, D.members(2.0))
        d2 = [e for e in events(D) if e["ev"] == "decision" and e.get("stage") == 2]
        assert len(d2) == 1 and d2[0]["decision"] == "accept" and d2[0]["tag"] == A and d2[0]["z"] < -1, d2
        assert abs(d2[0]["logr1"] - math.log(0.90)) < 0.02 and abs(d2[0]["logr"] - math.log(0.94)) < 0.02, d2
        assert any(e["ev"] == "incumbent" and e["tag"] == A for e in events(D))
        assert D.kind_stats["manual"] == [1, 1], D.kind_stats["manual"]
        assert "ACCEPT (confirmed)" in log(D) and "INBOX a.json" in log(D)


@test
def probe_fit_rows_and_reads_no_decision():
    """a probe (size 2.25, not a CONF size) with an extra slice (t183): scored whole on the active slice (kind probe) and its read queued
    as kind read (tag ibx-t183-<probe tag>), both answered; the probe's rows are fit rows; no decision, no pool, no incumbent and no
    registry count; the read is neither a fit row nor a decision; frontier.json has no probe or read point; status shows no inbox"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp); D, H = C.boot(cp); inc = dict(D.inc); pool = {z: list(D.members(z)) for z in D.sizes}
        drop(D, "p.json", body(D, 2.25, "probe", extra_slices=[{"name": "t183", "golden": T183}]))
        C.quiet(D.step)
        P = inbox_tags(D, "probe"); X = inbox_tags(D, "read")
        assert len(P) == 1 and X == [f"ibx-t183-{P[0]}"], (P, X)
        assert D.sub[P[0]]["early"] is False and D.sub[P[0]]["size"] == 2.25 and D.sub[X[0]]["slice"] == "t183"
        C.steps(D, H, 2)
        ev = events(D)
        assert not [e for e in ev if e["ev"] == "decision" and e.get("tag") in P + X], "a probe or read wrote a decision"
        assert not [e for e in ev if e["ev"] in ("tentative", "confirm", "pool", "incumbent", "reselect") and e.get("tag") in P + X]
        assert D.inc == inc and {z: list(D.members(z)) for z in D.sizes} == pool and D.dec[2.0] == 0 and D.dec[3.0] == 0
        assert D.reg.d["slices"]["s186a"]["decisions"] == {} and D.reg.d["slices"]["t183"]["decisions"] == {}
        assert any(k[0] == P[0] for k in D.obs) and not any(k[0] in X for k in D.obs), "probe rows are fit rows; reads are not"
        assert P[0] in D.res and not D.res[P[0]].get("fail") and X[0] in D.res and not D.res[X[0]].get("fail")
        fr = json.load(open(f"{D.R}/frontier.json"))
        assert not any(p["tag"] in P + X for b in fr.values() for p in b["points"]), "a probe is a frontier point"
        assert os.path.exists(f"{D.R}/inbox/done/p.json")
        assert "probe of 2.25 scored" in log(D) and "(no decision)" in log(D) and "inbox read of" in log(D)
        assert "inbox" not in open(f"{D.R}/status.txt").read(), "an empty inbox shows no status segment"


@test
def decide_waits_for_the_scored_incumbent():
    """a decide file dropped before the incumbents are scored waits (the size's incumbent is not on the active slice yet): nothing is
    submitted, the file stays in inbox/ with the status segment; once the incumbent is scored it goes out as manual"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp); D = C.driver(cp); H = C.Host(cp)
        drop(D, "w.json", body(D, 3.0, a=fill(D, 3.0, 2)))
        C.quiet(D.step)                                                    # incumbents submitted, none scored yet
        assert not inbox_tags(D, "manual") and os.path.exists(f"{D.R}/inbox/w.json")
        assert "| inbox 1 (0 outstanding)" in open(f"{D.R}/status.txt").read()
        C.steps(D, H, 2)
        assert len(inbox_tags(D, "manual")) == 1 and os.path.exists(f"{D.R}/inbox/done/w.json")


@test
def rejects_with_reasons():
    """each bad file moves to rejected/ with its reason in driver.log and submits nothing: an unknown weight, an unknown option, an
    unassigned weight, decide at a size not in CONF, decide out of the size's window, a probe more than 0.02 from its size, a bad
    mode, decide at a stopped size, a malformed extra slice; bad JSON written just now waits in inbox/ (not rejected yet)"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp); D, H = C.boot(cp); n0 = len(D.sub)
        good = fill(D, 2.0); k0 = next(iter(good))
        open(f"{D.R}/STOP-3", "w").close()                                  # size 3.0 stopped by hand (rule of the alert-only monitor)
        cases = {
            "r_weight": (body(D, 2.0, a={**good, "blk.99.nope.weight": "q27v3-x"}), "unknown weight"),
            "r_opt": (body(D, 2.0, a={**good, k0: "q99-no-such-option"}), "unknown option"),
            "r_miss": (body(D, 2.0, a={k: v for k, v in good.items() if k != k0}), "unassigned"),
            "r_size": (body(D, 2.25), "not in CONF sizes"),
            "r_window": (body(D, 2.0, a=fill(D, 3.0)), "outside"),
            "r_probe": (body(D, 2.5, "probe", a=fill(D, 2.0)), "more than 0.02"),
            "r_mode": (body(D, 2.0, mode="sweep"), "neither decide nor probe"),
            "r_stop": (body(D, 3.0), "is stopped"),
            "r_extra": (body(D, 2.0, extra_slices=[{"name": "", "golden": "x"}]), "extra_slices"),
        }
        for i, (name, (b, _)) in enumerate(cases.items()): drop(D, f"{name}.json", b, mtime=time.time() - 200 + i)
        with open(f"{D.R}/inbox/junk.json", "w") as f: f.write("{not json")  # written just now: still being written
        C.quiet(D.step)
        assert len(D.sub) == n0, "a rejected file submitted an eval"
        assert os.path.exists(f"{D.R}/inbox/junk.json") and not os.path.exists(f"{D.R}/inbox/rejected/junk.json")
        for name, (_, reason) in cases.items():
            assert os.path.exists(f"{D.R}/inbox/rejected/{name}.json") and not os.path.exists(f"{D.R}/inbox/{name}.json"), name
            line = [l for l in log(D).splitlines() if f"inbox {name}.json rejected:" in l]
            assert line and reason in line[0], (name, line)
        # bad JSON that is old (not being written any more) is rejected
        with open(f"{D.R}/inbox/junk.json", "w") as f: f.write("{not json")
        os.utime(f"{D.R}/inbox/junk.json", (time.time() - 3600, time.time() - 3600))
        C.quiet(D.step)
        assert os.path.exists(f"{D.R}/inbox/rejected/junk.json") and "not valid JSON" in log(D)


@test
def restart_mid_inbox():
    """a driver restarted with inbox evals outstanding rebuilds the inbox state from the events: a recorded file is never resubmitted
    (no second eval line); a file still in inbox/ whose record exists is moved to done/ without an eval; a file dropped while the
    driver was down is submitted at the first pass after the restart; a file of a recorded name with other bytes is rejected"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp); D, H = C.boot(cp)
        drop(D, "a.json", body(D, 2.0, a=fill(D, 2.0, 2)))
        drop(D, "b.json", body(D, 2.25, "probe", a=fill(D, 2.25, 1), extra_slices=[{"name": "t183", "golden": T183}]))
        C.quiet(D.step)
        A = inbox_tags(D, "manual"); P = inbox_tags(D, "probe"); X = inbox_tags(D, "read")
        assert len(A) == 1 and len(P) == 1 and len(X) == 1
        nq = len(H.queued()); D2 = C.driver(cp)
        assert set(D2.inbox_rec) == {"a.json", "b.json"} and D2.inbox_tags == D.inbox_tags and D2.inbox_out() == D.inbox_out()
        C.quiet(D2.step); assert len(H.queued()) == nq, "a restart resubmitted a recorded file"
        # a crash between the record and the move: the file is back in inbox/
        shutil.copy2(f"{D.R}/inbox/done/a.json", f"{D.R}/inbox/a.json")
        drop(D2, "c.json", body(D2, 3.0, a=fill(D2, 3.0, 3)))              # dropped while the driver was down
        D3 = C.driver(cp)
        C.quiet(D3.step)
        assert len(H.queued()) == nq + 1, "the recorded file was resubmitted or the new one was not submitted"
        assert not os.path.exists(f"{D.R}/inbox/a.json") and os.path.exists(f"{D.R}/inbox/done/a.json")
        assert "already submitted as" in log(D3) and os.path.exists(f"{D.R}/inbox/done/c.json")
        assert len(inbox_tags(D3, "manual")) == 2
        # a different file under a recorded name is never submitted
        drop(D3, "b.json", body(D3, 2.25, "probe", a=fill(D3, 2.25, 0)))
        C.quiet(D3.step)
        assert len(H.queued()) == nq + 1 and os.path.exists(f"{D3.R}/inbox/rejected/b.json")
        assert "name already used by a different file" in log(D3)
        D4 = C.driver(cp); assert D4.inbox_tags == D3.inbox_tags and set(D4.inbox_rec) == {"a.json", "b.json", "c.json"}


@test
def inbox_max_bounds_outstanding_evals():
    """CONF inbox_max = 2: a probe with one extra read takes two, so the next files wait in inbox/ (in order) until those answers come;
    then the two waiting probes go out together"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, inbox_max=2); D, H = C.boot(cp); t0 = time.time() - 100
        drop(D, "p1.json", body(D, 2.25, "probe", a=fill(D, 2.25, 0), extra_slices=[{"name": "t183", "golden": T183}]), mtime=t0)
        drop(D, "p2.json", body(D, 2.5, "probe", a=fill(D, 2.5, 1)), mtime=t0 + 1)
        drop(D, "p3.json", body(D, 2.75, "probe", a=fill(D, 2.75, 2)), mtime=t0 + 2)
        C.quiet(D.step)
        assert len(D.inbox_out()) == 2 and os.path.exists(f"{D.R}/inbox/p2.json") and os.path.exists(f"{D.R}/inbox/p3.json")
        assert len(inbox_tags(D, "probe")) == 1 and len(inbox_tags(D, "read")) == 1
        assert "| inbox 2 (2 outstanding)" in open(f"{D.R}/status.txt").read()
        assert len(D.outstanding()) == 2                                  # the inbox's evals are queue evals (queue_depth counts them)
        C.steps(D, H)                                                     # answers p1 and its read; p2 and p3 go out
        assert not os.path.exists(f"{D.R}/inbox/p2.json") and not os.path.exists(f"{D.R}/inbox/p3.json")
        assert len(inbox_tags(D, "probe")) == 3 and len(D.inbox_out()) == 2
        C.steps(D, H)
        assert not D.inbox_out() and not [f for f in os.listdir(f"{D.R}/inbox") if f.endswith(".json")]
        assert "inbox" not in open(f"{D.R}/status.txt").read()


@test
def oldest_file_first():
    """the oldest file (mtime) goes first even when its name sorts later; the status segment shows the waiting files and the outstanding
    evals only while the inbox holds some"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, inbox_max=1); D, H = C.boot(cp); t0 = time.time() - 100
        drop(D, "a.json", body(D, 2.25, "probe", a=fill(D, 2.25, 0)), mtime=t0 + 50)
        drop(D, "z.json", body(D, 2.5, "probe", a=fill(D, 2.5, 0)), mtime=t0)
        C.quiet(D.step)
        assert [D.sub[t]["size"] for t in inbox_tags(D, "probe")] == [2.5], "the older file (z) must go first"
        assert "| inbox 1 (1 outstanding)" in open(f"{D.R}/status.txt").read()
        C.steps(D, H)
        assert [D.sub[t]["size"] for t in inbox_tags(D, "probe")] == [2.5, 2.25]
        assert "| inbox 0 (1 outstanding)" in open(f"{D.R}/status.txt").read()
        C.steps(D, H)
        assert "inbox" not in open(f"{D.R}/status.txt").read()


@test
def empty_inbox_identity_183():
    """the inbox empty: HEAD and this tree emit the same events on the recorded #183 replay (w165 snapshot; w167's identity test)"""
    C.identity(C.SNAP183, 200)


@test
def empty_inbox_identity_live():
    """the inbox empty: the same events, queued evals, log, status, incumbents and pools as HEAD on a copy of the live run (3 steps)"""
    if not os.path.exists(f"{LIVE}/drv/state.jsonl"): print(f"   (SKIP: no live run at {LIVE})"); return
    with tempfile.TemporaryDirectory() as tmp:
        base = C.baseline_tree(tmp)
        if base is None: print("   (SKIP: no baseline tree: set RXM_POOL_BASE)"); return
        outs = []
        for tree, name in ((base, "head"), (C.TREE, "work")):
            out = f"{tmp}/{name}.json"
            r = subprocess.run([sys.executable, "-I", f"{HERE}/dryrun192.py", "--tree", tree, "--snap", LIVE, "--out", out, "--steps", "3"],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(json.load(open(out)))
        h, w = outs
        for k in ("events", "queued", "log", "status", "inc", "pool"): assert h[k] == w[k], f"{k} differs from HEAD"
        assert not any(w["inbox"].values()), w["inbox"]                  # nothing moved to done/ or rejected/
        print(f"   {len(w['events'])} events, {len(w['queued'])} queued identical (status: {w['status']})")


# ------------------------------------------------------------------------------------------------ run
def main():
    bad = 0
    for f in T:
        try: f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback; bad += 1; print(f"FAIL {f.__name__}: {e!r}"); traceback.print_exc()
    print(f"{len(T) - bad} / {len(T)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
