# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_inbox192b.py -- the round-robin of the driver inbox (#192-02, w174): CONF "inbox_share" (the free queue slots shared between
the inbox files and the driver's own moves) and "inbox_pending_max" (files beyond N waiting in inbox/ move to inbox/overflow/).
Checks: the share holds (0.5 of 26 with 100 files waiting), the own moves still go in when the inbox is full, a decide waiting for
its incumbent holds no slot, a two-eval probe holds the last slot until two are free, one free slot takes a decide before an older
probe (once), the overflow (newest beyond N, a young file waits), restart in the share path (no resubmission, no duplicate tag),
the validation of the keys, and identity with both keys absent: HEAD and this tree emit the same events on a scenario with inbox
files (scenario192b.py) and on a pinned copy of the live run with the 15 consensus files (fresh, not yet recorded).

  python3 -I search/v3/inbox192/test_inbox192b.py      (exit 0 iff every test passes; CPU, about a minute)

The own moves are a stand-in for generate() (the fake hosts give no fits, so the real generator makes nothing): the share tests drive
the real step loop with it. Baseline tree for the identity: RXM_POOL_BASE or `git archive HEAD` (as test_confirm189: run it before the
patch is committed, or HEAD is the patched tree). RXM_LIVE_RUN (default the lead's run dir; only read: copied once), RXM_CONSENSUS
(default the lead's consensus192 dir); a missing one is SKIPPED."""
import glob, json, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
sys.path.insert(0, V3); sys.path.insert(0, os.path.join(V3, "confirm189")); sys.path.insert(0, HERE)
import test_confirm189 as C  # noqa: E402  (env, Host, boot, steps, events, baseline_tree, TREE)
import test_inbox192 as I    # noqa: E402  (the #192 suite's helpers: env, fill, body, drop, T183; its identity test is not called here)

T = []
LIVE = os.environ.get("RXM_LIVE_RUN", I.LIVE)
CONS = os.environ.get("RXM_CONSENSUS", "/workspace/rad-local/workers/lead-srch/consensus192")


def test(f): T.append(f); return f


# ------------------------------------------------------------------------------------------------ helpers
def env(tmp, depth=26, **conf):
    """the #192 environment (K = 3, sizes 2.0 / 3.0, the t183 slice) with queue_depth and the round-robin keys (conf) set"""
    inbox_max = conf.pop("inbox_max", 6)
    cp = I.env(tmp, inbox_max=inbox_max)
    c = json.load(open(cp)); c["queue_depth"] = depth; c.update(conf); json.dump(c, open(cp, "w"))
    return cp


def own(D):
    """stand-in for generate(): up to `room` own candidates per call (kind probe: answered with no decision), on the active slice;
    returns the count of candidates made, in a one-element list (the next candidate's number)"""
    k = [0]
    def gen(act, room=None):
        cap = int(D.c.get("queue_depth", 6)) - len(D.outstanding())
        if room is not None: cap = min(int(room), cap)
        made = 0
        while made < cap:
            k[0] += 1
            C.quiet(D.submit, D.newtag(2.0, "probe"), I.fill(D, 2.0, k[0]), 2.0, "probe", act, False)
            made += 1
        return made
    D.generate = gen
    return k


def counts(D):
    """(inbox evals outstanding, own evals outstanding)"""
    n_in = len(D.inbox_out()); return n_in, len(D.outstanding()) - n_in


def status(D): return open(f"{D.R}/status.txt").read()


def waiting(D): return sorted(f for f in os.listdir(f"{D.R}/inbox") if f.endswith(".json"))


def probes(D, n, t0, size=2.25):
    """n one-eval probes, oldest first (mtime t0 + i); distinct assignments"""
    for i in range(n): I.drop(D, f"p{i:03d}.json", I.body(D, size, "probe", a=I.fill(D, size, i)), mtime=t0 + i)


def log(D): return open(f"{D.R}/driver.log").read()


# ------------------------------------------------------------------------------------------------ the share
@test
def share_holds_half_of_26():
    """CONF inbox_share 0.5, queue_depth 26, inbox_max 26, 100 probes waiting: one step fills the 26 slots 13 inbox / 13 own (round
    robin, the inbox first on the tie); the other 87 wait in inbox/; status: inbox waiting 87, outstanding 13 / 26"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, depth=26, inbox_max=26, inbox_share=0.5); D, H = C.boot(cp); k = own(D)
        assert not D.outstanding(), "boot left evals outstanding"
        probes(D, 100, time.time() - 1000)
        C.quiet(D.step)
        assert counts(D) == (13, 13), counts(D)
        assert k[0] == 13 and len(waiting(D)) == 87 and len(os.listdir(f"{D.R}/inbox/done")) == 13
        assert "| inbox waiting 87, outstanding 13 / 26" in status(D), status(D)


@test
def keys_absent_the_inbox_takes_the_slots_first():
    """the same 100 probes with the keys absent (the #192 behaviour): the inbox takes 26 slots (inbox_max 26), the own moves none;
    status: inbox 74 (26 outstanding)"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, depth=26, inbox_max=26); D, H = C.boot(cp); k = own(D)
        probes(D, 100, time.time() - 1000)
        C.quiet(D.step)
        assert counts(D) == (26, 0) and k[0] == 0, counts(D)
        assert "| inbox 74 (26 outstanding)" in status(D), status(D)


@test
def own_moves_still_go_in_when_the_inbox_is_full():
    """inbox_max 6 (the hard cap) with share 0.5 and queue_depth 26: the inbox holds 6, the own moves fill the other 20 slots"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, depth=26, inbox_max=6, inbox_share=0.5); D, H = C.boot(cp); k = own(D)
        probes(D, 100, time.time() - 1000)
        C.quiet(D.step)
        assert counts(D) == (6, 20) and k[0] == 20, (counts(D), k)
        assert "| inbox waiting 94, outstanding 6 / 6" in status(D), status(D)


@test
def share_one_gives_the_inbox_first_and_the_own_moves_the_rest():
    """inbox_share 1.0 (inbox priority): the inbox takes up to its cap first (inbox_max 4), the own moves the other 2 of 6 slots"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, depth=6, inbox_max=4, inbox_share=1.0); D, H = C.boot(cp); k = own(D)
        probes(D, 10, time.time() - 1000)
        C.quiet(D.step)
        assert counts(D) == (4, 2), counts(D)


@test
def odd_depth_round_robin_gives_the_inbox_the_extra_slot():
    """queue_depth 5, share 0.5: the ties alternate (the inbox first), so the inbox holds 3 and the own moves 2"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, depth=5, inbox_max=5, inbox_share=0.5); D, H = C.boot(cp); k = own(D)
        probes(D, 10, time.time() - 1000)
        C.quiet(D.step)
        assert counts(D) == (3, 2), counts(D)


@test
def decide_waiting_for_its_incumbent_holds_no_slot():
    """a decide whose size has no scored incumbent yet waits in inbox/ and takes no slot: the own moves fill the queue (depth 6)"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, depth=6, inbox_max=6, inbox_share=0.5); D = C.driver(cp); own(D)
        I.drop(D, "d.json", I.body(D, 3.0, a=I.fill(D, 3.0, 2)))
        C.quiet(D.step)
        assert counts(D)[0] == 0 and len(D.outstanding()) == 6, counts(D)
        assert os.path.exists(f"{D.R}/inbox/d.json") and not D.inbox_rec


@test
def two_eval_probe_holds_the_last_slot():
    """depth 4, inbox_max 4, share 0.5: three own evals are out, one slot free. The oldest file is a probe with one extra read (two
    evals), a one-eval probe waits behind it. The slot is held for the two-eval probe (no own move, no one-eval probe takes it); when
    one own eval answers, the two slots go to the two-eval probe and its read"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, depth=4, inbox_max=4, inbox_share=0.5); D, H = C.boot(cp); k = own(D)
        D.generate(D.reg.active(), room=3)
        assert len(D.outstanding()) == 3
        t0 = time.time() - 1000
        I.drop(D, "p.json", I.body(D, 2.25, "probe", a=I.fill(D, 2.25, 0), extra_slices=[{"name": "t183", "golden": I.T183}]), mtime=t0)
        I.drop(D, "q.json", I.body(D, 2.5, "probe", a=I.fill(D, 2.5, 1)), mtime=t0 + 1)
        C.quiet(D.step)
        assert counts(D) == (0, 3) and waiting(D) == ["p.json", "q.json"], (counts(D), waiting(D))
        H.run(D, only=[t for t in D.outstanding()][:1])                    # one own eval answers: two slots are free
        C.quiet(D.step)
        assert counts(D) == (2, 2) and waiting(D) == ["q.json"], (counts(D), waiting(D))
        assert os.path.exists(f"{D.R}/inbox/done/p.json") and len(D.inbox_out()) == 2
        assert sorted(D.sub[t]["kind"] for t in D.inbox_out()) == ["probe", "read"]


@test
def one_slot_takes_the_decide_before_an_older_probe_once():
    """with one slot free a waiting decide goes before an older one-eval probe; the probe is then the next one-slot pick (inbox_passed),
    so a probe is passed over at most once; a two-eval probe with one slot is held (-1), and goes with two (2)"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, inbox_share=0.5); D, H = C.boot(cp); act = D.reg.active()
        t0 = time.time() - 1000
        I.drop(D, "p.json", I.body(D, 2.25, "probe", a=I.fill(D, 2.25, 0)), mtime=t0)
        I.drop(D, "d.json", I.body(D, 2.0, a=I.fill(D, 2.0, 1)), mtime=t0 + 1)
        assert D.inbox_take(act, 1) == 1 and [D.sub[t]["kind"] for t in D.inbox_out()] == ["manual"], D.inbox_out()
        assert D.inbox_passed == "p.json" and os.path.exists(f"{D.R}/inbox/p.json")
        assert D.inbox_take(act, 1) == 1 and os.path.exists(f"{D.R}/inbox/done/p.json") and not os.path.exists(f"{D.R}/inbox/p.json")
        I.drop(D, "x.json", I.body(D, 2.5, "probe", a=I.fill(D, 2.5, 2), extra_slices=[{"name": "t183", "golden": I.T183}]), mtime=t0 + 2)
        assert D.inbox_take(act, 1) == -1 and os.path.exists(f"{D.R}/inbox/x.json") and len(D.inbox_out()) == 2
        assert D.inbox_take(act, 2) == 2 and os.path.exists(f"{D.R}/inbox/done/x.json") and len(D.inbox_out()) == 4


# ------------------------------------------------------------------------------------------------ overflow
@test
def overflow_moves_the_newest_beyond_pending_max_and_skips_a_young_file():
    """CONF inbox_pending_max 200 (inbox_max 6, no share): 210 old probes and one file written just now are waiting. The 10 newest old
    files move to inbox/overflow/ with a log line each; the young file (it may still be written) stays; then the oldest 6 go out.
    status: inbox waiting 195, outstanding 6 / 6"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, inbox_max=6, inbox_pending_max=200); D = C.driver(cp)
        probes(D, 210, time.time() - 3600)
        I.drop(D, "zz.json", I.body(D, 2.25, "probe", a=I.fill(D, 2.25, 3)))
        C.quiet(D.step)
        over = sorted(os.listdir(f"{D.R}/inbox/overflow"))
        assert over == [f"p{i:03d}.json" for i in range(200, 210)], over
        assert sum("moved to inbox/overflow/" in l for l in log(D).splitlines()) == 10 and "inbox p200.json overflow: 211 waiting" in log(D)
        assert waiting(D) == sorted([f"p{i:03d}.json" for i in range(6, 200)] + ["zz.json"]), waiting(D)[:3]
        assert counts(D)[0] == 6 and "| inbox waiting 195, outstanding 6 / 6" in status(D), status(D)


# ------------------------------------------------------------------------------------------------ restart
@test
def restart_in_the_share_path_never_resubmits():
    """share 0.5, depth 6: after one step (3 inbox, 3 own) a restarted driver rebuilds the inbox from the events (the same evals, no
    new eval while the queue is full); a file copied back into inbox/ after its record is moved to done/ with no eval; no tag is queued
    twice"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = env(tmp, depth=6, inbox_max=6, inbox_share=0.5); D, H = C.boot(cp); own(D)
        probes(D, 10, time.time() - 1000)
        C.quiet(D.step)
        assert counts(D) == (3, 3), counts(D)
        nq = len(H.queued()); D2 = C.driver(cp); own(D2)
        assert set(D2.inbox_rec) == set(D.inbox_rec) and D2.inbox_out() == D.inbox_out()
        C.quiet(D2.step)
        assert len(H.queued()) == nq, "a restart queued an eval with the queue full"
        shutil.copy2(f"{D.R}/inbox/done/p000.json", f"{D.R}/inbox/p000.json")
        C.quiet(D2.step)
        assert len(H.queued()) == nq and os.path.exists(f"{D.R}/inbox/done/p000.json") and not os.path.exists(f"{D.R}/inbox/p000.json")
        assert "already submitted as" in log(D2)
        tags = [q["tag"] for q in H.queued()]
        assert len(tags) == len(set(tags)), "a tag is queued twice"


# ------------------------------------------------------------------------------------------------ validation
@test
def bad_round_robin_keys_are_refused():
    """inbox_share 0 or above 1, and inbox_pending_max below 1, are refused when the driver starts"""
    for bad in ({"inbox_share": 0}, {"inbox_share": 1.5}, {"inbox_pending_max": 0}):
        with tempfile.TemporaryDirectory() as tmp:
            cp = env(tmp, **bad); raised = False
            try: C.driver(cp)
            except ValueError as ex: raised = "CONF" in str(ex)
            assert raised, bad


# ------------------------------------------------------------------------------------------------ identity (keys absent)
def run(tree, args, out):
    r = subprocess.run([sys.executable, "-I", f"{HERE}/scenario192b.py", "--tree", tree, "--out", out] + args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.load(open(out))


@test
def identity_absent_keys_on_a_scenario_with_inbox_files():
    """both keys absent: HEAD and this tree emit the same events, queue, log, status and inbox folders on scenario192b.py (a decide, a
    probe, a bad file, a file being written; four steps, all through the #192 path)"""
    with tempfile.TemporaryDirectory() as tmp:
        base = C.baseline_tree(tmp)
        if base is None: print("   (SKIP: no baseline tree: set RXM_POOL_BASE)"); return
        h = run(base, [], f"{tmp}/head.json"); w = run(C.TREE, [], f"{tmp}/work.json")
        for k in ("events", "queued", "log", "status", "inbox"): assert h[k] == w[k], f"{k} differs from HEAD"
        assert any(e["ev"] == "inbox" for e in w["events"]), "the scenario submitted no inbox file"
        assert w["inbox"]["rejected"] == ["r.json"] and w["inbox"][""] == ["junk.json"], w["inbox"]
        assert any("inbox r.json rejected: unknown weight" in l for l in w["log"]), w["log"][-3:]
        print(f"   {len(w['events'])} events, {len(w['queued'])} queued identical (inbox {w['inbox']})")


@test
def identity_absent_keys_on_a_pinned_live_copy_with_the_consensus_files():
    """both keys absent, on one pinned copy of the live run (the inbox records removed, so the 15 consensus files are fresh; the copy
    is made once, for both trees): events, queue, log, status, incumbents and pools equal HEAD's (dryrun192.py, 3 steps)"""
    cons = sorted(glob.glob(f"{CONS}/*.json"))
    if not os.path.exists(f"{LIVE}/drv/state.jsonl") or not cons: print(f"   (SKIP: no live run at {LIVE} or no consensus files)"); return
    with tempfile.TemporaryDirectory() as tmp:
        base = C.baseline_tree(tmp)
        if base is None: print("   (SKIP: no baseline tree: set RXM_POOL_BASE)"); return
        snap = f"{tmp}/snap"; os.makedirs(f"{snap}/drv/assign")
        for f in ("driver.json", "registry.json", "menu.json"): shutil.copy(f"{LIVE}/{f}", f"{snap}/{f}")
        for f in os.listdir(f"{LIVE}/drv/assign"): shutil.copy(f"{LIVE}/drv/assign/{f}", f"{snap}/drv/assign/{f}")
        with open(f"{LIVE}/drv/state.jsonl") as i, open(f"{snap}/drv/state.jsonl", "w") as o:
            for l in i:                                                      # the live driver may be appending: a partial last line is dropped
                if not l.endswith("\n"): break
                if '"ev": "inbox"' not in l: o.write(l)
        outs = []
        for tree, name in ((base, "head"), (C.TREE, "work")):
            out = f"{tmp}/{name}.json"
            r = subprocess.run([sys.executable, "-I", f"{I.HERE}/dryrun192.py", "--tree", tree, "--snap", snap, "--out", out, "--steps", "3",
                                "--conf-set", "queue_depth=40", "--conf-set", "inbox_max=20"] + sum((["--inbox", f] for f in cons), []),
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(json.load(open(out)))
        h, w = outs
        for k in ("events", "queued", "log", "status", "inc", "pool", "inbox"): assert h[k] == w[k], f"{k} differs from HEAD"
        print(f"   {len(w['events'])} events, {len(w['queued'])} queued, {len(w['log'])} log lines identical ({len(cons)} files)")


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
