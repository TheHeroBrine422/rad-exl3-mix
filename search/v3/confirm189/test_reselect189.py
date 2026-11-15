# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_reselect189.py -- the one-shot re-selection of past incumbents on a fresh slice (#189-02, RULES-confirm rule 9, w168):
the candidate set (distinct assignment hashes of incumbents and pool members at or after CONF from_line, plus the current
incumbent), the start after the go-live check, a restart mid-re-selection, the z rule (keep / switch), the pool (top 3 of the
ranking with the new incumbent kept) and its rescore on the active slice, tentatives waiting, the registry role (never rotated
to, refusals), the plot (no decision event, nothing chained, a level reset on the new incumbent's own read), and CONF
"reselect" absent = the HEAD driver event for event on the live state (on_result replay and step replay).

  python3 -I search/v3/confirm189/test_reselect189.py      (exit 0 iff every test passes; CPU, ~30 s)

Harness: test_confirm189's fake hosts and environment (s186a-d) plus s186e (role reselect) and s186f (role confirm, in no
pair). Identity: env RXM_RS_SNAP (a copy of the live run with confirm on, default /workspace/rad-local/workers/w168/snap2);
baseline tree RXM_POOL_BASE or `git archive HEAD`. A missing snapshot is SKIPPED."""
import json, math, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
SEARCH = os.path.dirname(V3)
sys.path.insert(0, HERE); sys.path.insert(0, V3)
import test_confirm189 as TC   # noqa: E402  (its environment, fake hosts and helpers)
import poolrule as PR          # noqa: E402
import slices as SL            # noqa: E402

SNAP = os.environ.get("RXM_RS_SNAP", "/workspace/rad-local/workers/w168/snap2")
T = []
driver, quiet, Host, cand, steps, events, boot = TC.driver, TC.quiet, TC.Host, TC.cand, TC.steps, TC.events, TC.boot


def test(f): T.append(f); return f


def make_env(tmp, rs_role="reselect", **kw):
    """test_confirm189's environment + s186e (role rs_role) and s186f (role confirm, in no CONF pair)"""
    nd = kw.get("nd", 24); cp = TC.make_env(tmp, **kw); rp = json.load(open(cp))["registry"]; reg = json.load(open(rp))
    for i, n in ((4, "s186e"), (5, "s186f")):
        reg["slices"][n] = {"parts": {}, "mix": f"/gold/{n}", "alias": f"/data/q27/w164/slices/golden-{n}", "host_dirs": {},
                            "docs": TC.slice_docs(n, nd, i), "status": "staged", "decisions": {}, "built": f"2026-11-14T00:0{i}:00Z"}
    if rs_role: reg["slices"]["s186e"]["role"] = rs_role
    reg["slices"]["s186f"]["role"] = "confirm"
    json.dump(reg, open(rp, "w"))
    return cp


def add_reselect(cp, from_line, sizes, top=3, z=-1.0):
    c = json.load(open(cp)); c["reselect"] = {"slice": "s186e", "from_line": from_line, "sizes": list(sizes), "top": top, "z": z}
    json.dump(c, open(cp, "w"))


def nlines(D): return sum(1 for _ in open(f"{D.R}/state.jsonl"))


def line_of(D, pred):
    for i, e in enumerate(events(D), 1):
        if pred(e): return i
    return None


def rs_evals(D, tag=None):
    return [t for t, e in D.sub.items() if e["kind"] == "reselect" and (tag is None or e["hash"] == D.sub[tag]["hash"])]


def hset(D, tags): return {D.sub[t]["hash"] for t in tags}


def accept(D, H, size, k, j, **fs):
    """a direct stage-1 accept (CONF confirm absent): the candidate becomes incumbent (and joins the pool)"""
    A = cand(D, size, k, j); H.set(D, A, **fs); steps(D, H)
    assert D.inc[size] == A, (size, D.inc[size], A)
    return A


def history(tmp, sizes=(2.0, 3.0)):
    """K = 3, no confirm: 2.0: I -> A (add) -> B (add) -> C (replaces the worst, I: I leaves the pool); 3.0: J -> X (add)"""
    cp = make_env(tmp, sizes=sizes, confirm=False); D, H = boot(cp)
    I, J = D.inc[2.0], D.inc.get(3.0)
    A = accept(D, H, 2.0, 2, 0, s186a=0.95); B = accept(D, H, 2.0, 2, 1, s186a=0.92); C = accept(D, H, 2.0, 2, 2, s186a=0.85)
    assert sorted(D.members(2.0)) == sorted([A, B, C]), D.members(2.0)
    X = accept(D, H, 3.0, 4, 0, s186a=0.9) if 3.0 in sizes else None
    return cp, D, H, dict(I=I, A=A, B=B, C=C, J=J, X=X)


# ------------------------------------------------------------------------------------------------ candidate set
@test
def candidate_set_dedupe_and_from_line():
    """the candidates: distinct hashes of incumbent tags and pool members at or after from_line (a line AT from_line counts), the
    current incumbent included; a rotation retag is the same model (one eval, both tags listed); a model whose last appearance is
    before from_line is out; the evals are whole (no early stop) on s186e, kind reselect"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp, K=1, sizes=(2.0,), confirm=False); D, H = boot(cp); I = D.inc[2.0]
        A = accept(D, H, 2.0, 2, 0, s186a=0.95)
        B = accept(D, H, 2.0, 2, 1, s186a=0.9); LB = line_of(D, lambda e: e["ev"] == "incumbent" and e["tag"] == B)
        C = accept(D, H, 2.0, 2, 2, s186a=0.85)
        D.c["rotate_after"] = 1; quiet(D.step); D.c["rotate_after"] = 10 ** 6; steps(D, H, 2)
        C2 = D.inc[2.0]
        assert C2 != C and D.sub[C2]["hash"] == D.sub[C]["hash"] and D.reg.active() == "s186c"
        for fl, want in ((LB, {B, C}), (LB + 1, {C})):
            sub = f"{tmp}/fl{fl}"; shutil.copytree(tmp, sub, ignore=shutil.ignore_patterns("fl*"))
            c = json.load(open(f"{sub}/conf.json"))
            for k in ("root", "menu", "registry", "evals", "results", "docs"): c[k] = c[k].replace(tmp, sub)
            json.dump(c, open(f"{sub}/conf.json", "w")); add_reselect(f"{sub}/conf.json", fl, [2.0])
            D2 = driver(f"{sub}/conf.json"); quiet(D2.step)
            p = D2.rs_plan["plan"]; assert len(p) == 1 and p[0]["size"] == 2.0
            got = {c_["hash"] for c_ in p[0]["cands"]}
            assert got == hset(D2, want), (fl, len(got))
            cc = [c_ for c_ in p[0]["cands"] if c_["hash"] == D2.sub[C]["hash"]][0]
            assert set(cc["tags"]) == {C, C2}, cc
            ev = rs_evals(D2); q = {x["tag"]: x for x in Host(f"{sub}/conf.json").queued()}
            assert len(ev) == len(want) and hset(D2, ev) == hset(D2, want), ev
            assert all("early_stop" not in q[x] and q[x]["slices"][0]["parts"]["m"] == "/data/q27/w164/slices/golden-s186e" for x in ev)
            assert I not in [t for c_ in p[0]["cands"] for t in c_["tags"]] and A not in [t for c_ in p[0]["cands"] for t in c_["tags"]]


@test
def pool_members_count_and_left_members_stay():
    """K = 3: a member that left the pool after from_line (I, replaced) is still a candidate; 2.0 has 4 models, 3.0 2"""
    with tempfile.TemporaryDirectory() as tmp:
        cp, D, H, m = history(tmp); add_reselect(cp, 1, [2.0, 3.0])
        D2 = driver(cp); quiet(D2.step)
        p = {x["size"]: x for x in D2.rs_plan["plan"]}
        assert {c["hash"] for c in p[2.0]["cands"]} == hset(D2, [m["I"], m["A"], m["B"], m["C"]])
        assert {c["hash"] for c in p[3.0]["cands"]} == hset(D2, [m["J"], m["X"]])
        assert len(rs_evals(D2)) == 6


# ------------------------------------------------------------------------------------------------ z rule, pool, rescore
@test
def switch_pool_top3_and_rescore_on_active():
    """2.0 (incumbent C): on s186e I 0.95, A 0.97, C 1.0, B 1.03 -> I (z < -1) is the new incumbent, pool = top 3 [I, A, C]
    (B out); I is rescored whole on the active slice (a rotation rescore), A and C keep their tags; RULES-pool §5 is not run
    again on the active slice; no decision event; the incumbent event says why "reselect9". 3.0 (incumbent X): J 0.999 ->
    z > -1 -> kept, pool unchanged in membership"""
    with tempfile.TemporaryDirectory() as tmp:
        cp, D, H, m = history(tmp); add_reselect(cp, 1, [2.0, 3.0])
        for k, f in (("I", 0.95), ("A", 0.97), ("C", 1.0), ("B", 1.03), ("X", 1.0), ("J", 0.999)): H.set(D, m[k], s186e=f)
        D2 = driver(cp); n0 = nlines(D2)
        steps(D2, H, 4)
        r = {e["size"]: e for e in events(D2) if e["ev"] == "reselect_size"}
        assert r[2.0]["verdict"] == "switched" and r[3.0]["verdict"] == "kept", {z: g["verdict"] for z, g in r.items()}
        rk = r[2.0]["ranking"]
        assert [x["of"] for x in rk] == [m["I"], m["A"], m["B"]] and rk[0]["pass"] and rk[0]["z"] < -1, [(x["of"], x["da"]) for x in rk]
        assert D2.sub[D2.inc[2.0]]["hash"] == D2.sub[m["I"]]["hash"] and D2.res[D2.inc[2.0]]["slice"] == "s186a" and D2.sub[D2.inc[2.0]]["kind"] == "incumbent"
        assert hset(D2, D2.members(2.0)) == hset(D2, [m["I"], m["A"], m["C"]]) and m["A"] in D2.members(2.0) and m["C"] in D2.members(2.0)
        assert D2.scored_on(2.0, "s186a") and D2.inc[3.0] == m["X"] and hset(D2, D2.members(3.0)) == hset(D2, [m["J"], m["X"]])
        ev = events(D2)[n0:]
        assert not [e for e in ev if e["ev"] == "decision"], "the re-selection wrote a decision event"
        assert not [e for e in ev if e["ev"] == "reselect"], "RULES-pool §5 ran again on the active slice"
        sw = [e for e in ev if e["ev"] == "incumbent" and e.get("why") == "reselect9"]
        assert len(sw) == 1 and sw[0]["size"] == 2.0 and sw[0]["prev"] == m["C"], sw
        assert [e["ev"] for e in ev if e["ev"].startswith("reselect")][-1] == "reselect_done" and D2.rs_end
        S = D2.reg.d["slices"]["s186e"]
        assert S["reselect_decisions"] == {"2.0": 1, "3.0": 1} and not S["decisions"] and not S.get("confirm_decisions") and S["status"] == "staged"


@test
def ranking_rule_matches_the_event():
    """for every size: the recorded verdict / pool = PR.reseed_pick on the recorded ranking with the z rule, the new incumbent in the
    pool; the pure rule: three better-ranked models without z < -1 -> the incumbent stays and is kept as the pool's last member"""
    with tempfile.TemporaryDirectory() as tmp:
        cp, D, H, m = history(tmp); add_reselect(cp, 1, [2.0, 3.0])
        for k, f in (("I", 0.98), ("A", 0.99), ("C", 1.0), ("B", 0.97), ("X", 1.0), ("J", 1.01)): H.set(D, m[k], s186e=f)
        D2 = driver(cp); steps(D2, H, 3)
        for g in (e for e in events(D2) if e["ev"] == "reselect_size"):
            hi = D2.sub[g["prev"]]["hash"]
            new_h, pool_h = PR.reseed_pick(hi, [(x["hash"], x["da"], x["pass"]) for x in g["ranking"]], 3)
            assert D2.sub[g["inc"]]["hash"] == new_h and hset(D2, g["pool"]) == set(pool_h) and g["inc"] in g["pool"], g
            assert (g["verdict"] == "switched") == (new_h != hi)
    assert PR.reseed_pick("inc", [("a", -0.03, False), ("b", -0.02, False), ("c", -0.01, False)], 3) == ("inc", ["a", "b", "inc"])
    assert PR.reseed_pick("inc", [("a", -0.03, False), ("b", -0.02, True), ("c", 0.01, True)], 3) == ("b", ["a", "b", "inc"])


@test
def rotation_during_reselection_holds_section5():
    """a rotation to s186c while the re-selection is pending: the pools are rescored there but RULES-pool §5 does not run for a
    pending size (A would win it on s186c); after the one-shot the new pool is rescored on s186c and §5 is taken as done there
    (the one-shot's incumbent stands)"""
    with tempfile.TemporaryDirectory() as tmp:
        cp, D, H, m = history(tmp); add_reselect(cp, 1, [2.0, 3.0])
        for k, f in (("I", 0.95), ("A", 1.0), ("C", 1.0), ("B", 1.03), ("X", 1.0), ("J", 1.0)): H.set(D, m[k], s186e=f)
        H.set(D, m["A"], s186e=1.0, s186c=0.9); H.set(D, m["X"], s186e=1.0, s186c=1.0); H.set(D, m["J"], s186e=1.0, s186c=0.9)
        D2 = driver(cp); n0 = nlines(D2)
        D2.c["rotate_after"] = 1; quiet(D2.step); D2.c["rotate_after"] = 10 ** 6
        assert D2.reg.active() == "s186c" and D2.rs_plan is not None
        steps(D2, H, 3, only=lambda t: D2.sub[t]["kind"] != "reselect")
        assert D2.scored_on(2.0, "s186c") and D2.scored_on(3.0, "s186c")
        assert not [e for e in events(D2)[n0:] if e["ev"] == "reselect"], "RULES-pool §5 ran for a size with its re-selection pending"
        steps(D2, H, 5)
        assert D2.rs_end and D2.sub[D2.inc[2.0]]["hash"] == D2.sub[m["I"]]["hash"] and D2.inc[3.0] in D2.res and D2.sub[D2.inc[3.0]]["hash"] == D2.sub[m["X"]]["hash"]
        assert D2.scored_on(2.0, "s186c") and D2.reselected == {**D2.reselected, 2.0: "s186c", 3.0: "s186c"}
        assert not [e for e in events(D2)[n0:] if e["ev"] == "reselect"], "RULES-pool §5 ran after the one-shot on the same slice"


# ------------------------------------------------------------------------------------------------ ordering: go-live first, tentatives wait
@test
def waits_for_golive_then_tentatives_wait():
    """with a go-live check pending nothing of the re-selection starts (no reselect_start, no s186e eval; status says it waits);
    it starts in the step the check completes; while a size's re-selection is pending its tentatives wait (their candidate's
    confirmation eval may run); after it, the tentative is confirmed against the NEW incumbent (rule 3)"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp, sizes=(2.0,), confirm=False); D, H = boot(cp)
        L = nlines(D); sw = D.inc[2.0]
        A = accept(D, H, 2.0, 2, 0, s186a=0.9)
        c = json.load(open(cp)); c["confirm"] = {"z": -1.0, "pairs": [["s186a", "s186b"], ["s186c", "s186d"]], "sizes": [2.0], "golive_from_line": L}
        json.dump(c, open(cp, "w")); add_reselect(cp, 1, [2.0])
        D2 = driver(cp); H = Host(cp)
        H.setx(D2.sub[A]["hash"], s186a=0.9, s186b=0.92, s186e=1.0); H.setx(D2.sub[sw]["hash"], s186e=0.95)
        swx = lambda t: D2.sub[t]["kind"] == "confirm" and D2.sub[t]["hash"] == D2.sub[sw]["hash"]
        quiet(D2.step); steps(D2, H, 2, only=lambda t: not swx(t))
        assert D2.golive is not None and not D2.golive_end and D2.rs_plan is None and not rs_evals(D2)
        assert "waits for go-live" in open(f"{D2.R}/status.txt").read()
        steps(D2, H, 1)                                                     # the go-live check completes -> the re-selection starts
        ev = events(D2); ig = line_of(D2, lambda e: e["ev"] == "golive" and e["phase"] == "done"); ir = line_of(D2, lambda e: e["ev"] == "reselect_start")
        assert ig is not None and ir is not None and ig < ir, (ig, ir)
        assert D2.inc[2.0] == A                                             # kept by go-live
        Cc = cand(D2, 2.0, 2, 3); H.setx(D2.sub[Cc]["hash"], s186a=0.85, s186b=0.96)
        steps(D2, H, 3, only=lambda t: D2.sub[t]["kind"] != "reselect")     # the s186e evals withheld
        assert Cc in D2.tent and D2.rs_wait(2.0) and confirm_evals_of(D2, Cc), "a tentative was decided during the re-selection"
        steps(D2, H, 4)
        g = [e for e in events(D2) if e["ev"] == "reselect_size"][0]
        assert g["verdict"] == "switched" and D2.sub[g["inc"]]["hash"] == D2.sub[sw]["hash"], g
        cf = [e for e in events(D2) if e["ev"] == "confirm" and e["tag"] == Cc][0]
        assert D2.sub[cf["ref"]]["hash"] == D2.sub[sw]["hash"] and cf["ref_changed"] is True, cf
        assert line_of(D2, lambda e: e["ev"] == "reselect_size") < line_of(D2, lambda e: e["ev"] == "confirm" and e["tag"] == Cc)


def confirm_evals_of(D, t): return [x for x, e in D.sub.items() if e["kind"] == "confirm" and e["hash"] == D.sub[t]["hash"]]


# ------------------------------------------------------------------------------------------------ restart
@test
def restart_mid_reselection():
    """a restart with s186e evals outstanding: the plan and the eval cache come back from the state (no second submit, no second
    reselect_start); the sizes decided before the restart stay decided; the rest finish; a second restart after the end adds
    no re-selection event and has the same incumbents and pools"""
    with tempfile.TemporaryDirectory() as tmp:
        cp, D, H, m = history(tmp); add_reselect(cp, 1, [2.0, 3.0])
        for k, f in (("I", 0.95), ("A", 1.0), ("C", 1.0), ("B", 1.0), ("X", 1.0), ("J", 0.9)): H.set(D, m[k], s186e=f)
        D2 = driver(cp); quiet(D2.step)
        x3 = set(rs_evals(D2, m["X"]) + rs_evals(D2, m["J"]))
        steps(D2, H, 2, only=x3)                                            # 3.0's set complete, 2.0's not
        assert 3.0 in D2.rs_res and 2.0 not in D2.rs_res and D2.inc[3.0] != m["X"]
        nq = len(H.queued()); D3 = driver(cp)
        nt = lambda e: {k: v for k, v in e.items() if k != "t"}
        assert nt(D3.rs_plan) == nt(D2.rs_plan) and set(D3.rs_res) == {3.0} and D3.cfx == D2.cfx and D3.reselected == D2.reselected
        quiet(D3.step); assert len(H.queued()) == nq, "a restart resubmitted a re-selection eval"
        steps(D3, H, 4)
        assert D3.rs_end and D3.sub[D3.inc[2.0]]["hash"] == D3.sub[m["I"]]["hash"] and D3.sub[D3.inc[3.0]]["hash"] == D3.sub[m["J"]]["hash"]
        assert sum(e["ev"] == "reselect_start" for e in events(D3)) == 1
        n0 = nlines(D3); D4 = driver(cp); quiet(D4.step)
        assert D4.rs_end and D4.inc == D3.inc and {z: D4.members(z) for z in D4.sizes} == {z: D3.members(z) for z in D3.sizes}
        assert not [e for e in events(D4)[n0:] if e["ev"].startswith("reselect") or e["ev"] in ("incumbent", "pool")]


@test
def failed_evals():
    """a candidate whose s186e eval fails twice is left out of the ranking (listed as failed); the incumbent's failing twice ->
    verdict error, incumbent and pool kept"""
    with tempfile.TemporaryDirectory() as tmp:
        cp, D, H, m = history(tmp); add_reselect(cp, 1, [2.0, 3.0])
        H.set(D, m["I"], s186e=0.9)
        D2 = driver(cp); quiet(D2.step)
        for _ in range(2):
            for x in rs_evals(D2, m["I"]) + rs_evals(D2, m["X"]): H.fail.add(x)
            steps(D2, H, 1, only=lambda t: D2.sub[t]["kind"] == "reselect" and D2.sub[t]["hash"] in hset(D2, [m["I"], m["X"]]))
        steps(D2, H, 3)
        r = {e["size"]: e for e in events(D2) if e["ev"] == "reselect_size"}
        assert r[2.0]["n_failed"] == 1 and m["I"] not in [x["of"] for x in r[2.0]["ranking"] if not x.get("failed")], r[2.0]["ranking"]
        assert r[3.0]["verdict"] == "error" and D2.inc[3.0] == m["X"], r[3.0]
        assert len(rs_evals(D2, m["I"])) == 2


# ------------------------------------------------------------------------------------------------ registry
@test
def registry_role_rotation_and_refusals():
    """a reselect slice is never activated (even when ready); the driver refuses CONF reselect without the role, after the slice
    was active, decided or confirmed, or when it is in a confirm pair; it accepts pairs a/b, c/d while s186f keeps role confirm;
    draw_merged never draws its calib docs; slices.py role sets it"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp + "/a", statuses={"s186c": "staged"}); rp = json.load(open(cp))["registry"]
        R = SL.Registry(rp); R.d["slices"]["s186e"]["status"] = "ready"; R.save()
        assert SL.Registry(rp).ready() == [] and SL.Registry(rp).activate_next() is None
        add_reselect(cp, 1, [2.0, 3.0]); D = driver(cp)                    # pairs a/b, c/d; s186f role confirm, unpaired: starts
        assert D.rs and D.cf_pairs == {"s186a": "s186b", "s186c": "s186d"}
        D.c["rotate_after"] = 1; H = Host(cp); steps(D, H, 3); A = cand(D, 2.0, 2, 0); steps(D, H, 3)
        assert D.reg.active() == "s186a" and not any(e["ev"] == "rotate" and e.get("from") for e in events(D))

        def refuses(name, edit=None, conf=None, role="reselect"):
            cp = make_env(f"{tmp}/{name}", rs_role=role); add_reselect(cp, 1, [2.0])
            if edit:
                R = SL.Registry(json.load(open(cp))["registry"]); edit(R.d["slices"]["s186e"]); R.save()
            if conf:
                c = json.load(open(cp)); conf(c); json.dump(c, open(cp, "w"))
            try: driver(cp)
            except SystemExit as e: return str(e)
            return None
        assert "role" in (refuses("norole", role=None) or "")
        assert "role" in (refuses("confrole", role="confirm") or "")
        assert "was active" in (refuses("active", edit=lambda s: s.update(status="retired", activated="x")) or "")
        assert "decided" in (refuses("decided", edit=lambda s: s.update(decisions={"2.0": 1})) or "")
        assert "decided" in (refuses("confirmed", edit=lambda s: s.update(confirm_decisions={"2.0": 1})) or "")
        assert "s186e" in (refuses("paired", conf=lambda c: c["confirm"]["pairs"].append(["s186c", "s186e"])) or "")   # w167's check first
        assert "s186e: a search slice with role reselect" in (refuses("paired2", conf=lambda c: c["confirm"]["pairs"].append(["s186e", "s186f"])) or "")
        assert "from_line" in (refuses("noline", conf=lambda c: c["reselect"].pop("from_line")) or "")
        assert refuses("ok") is None
        quiet(SL.main, ["role", rp, "s186c", "reselect"]); assert SL.Registry(rp).role("s186c") == "reselect"
        R = SL.Registry(rp); R.d["slices"]["s186e"]["docs"][0]["id"] = "c:3"; R.d["slices"]["s186e"]["status"] = "staged"
        for n in ("s186b", "s186d", "s186f"): R.d["slices"][n]["docs"][0]["id"] = "c:9"
        open(f"{tmp}/led.jsonl", "w").close()
        pick, _, _ = SL.draw_merged(R, f"{tmp}/led.jsonl", ["search"] * 8, 8, 0)
        assert 3 not in pick and 5 in pick, pick


# ------------------------------------------------------------------------------------------------ the plot
def load_cp():
    sys.path.insert(0, SEARCH)
    try: import matplotlib  # noqa: F401
    except ImportError:
        import types; mp = types.ModuleType("matplotlib"); mp.use = lambda *a, **k: None; mp.pyplot = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = mp; sys.modules["matplotlib.pyplot"] = mp.pyplot
    import convplot4 as CP
    return CP


@test
def plot_level_reset_nothing_chained():
    """convplot4 on a state with a re-selection switch: no decision event, the chain is the same before and after the switch
    (nothing of the re-selection chained); with a read of the new incumbent's model (any tag, same hash) the size is re-anchored
    there from the switch step (estimate = that read, then later accepts chain on it); without one the level is held"""
    CP = load_cp()
    with tempfile.TemporaryDirectory() as tmp:
        cp, D, H, m = history(tmp); add_reselect(cp, 1, [2.0, 3.0])
        for k, f in (("I", 0.9), ("A", 1.0), ("C", 1.0), ("B", 1.0), ("X", 1.0), ("J", 1.0)): H.set(D, m[k], s186e=f)
        D2 = driver(cp); steps(D2, H, 4)
        assert D2.sub[D2.inc[2.0]]["hash"] == D2.sub[m["I"]]["hash"]
        W = CP.walk(f"{D2.R}/state.jsonl"); k = (2.0, "f16")
        assert len(W["resets"]) == 1 and W["resets"][0]["size"] == 2.0
        x = W["resets"][0]; ch = W["accepts"][k]
        before = [c for s, c in ch if s < x["step"]][-1]
        assert all(abs(c - before) < 1e-12 for s, c in ch if s >= x["step"]), ch       # nothing chained at or after the reset
        assert not [d for d in W["decs"] if d["step"] >= x["step"] and d["size"] == 2.0]
        an0 = {"kld": 0.01, "cof": 0.0}
        W["reads"] = []; segs, held = CP.reanchor(W, {k: an0})
        assert held == [x] and CP.anchor_at(segs, k, x["step"] + 5) is an0                # no read: level held
        W["reads"] = [dict(step=1, size=2.0, mode="f16", kld=0.0123, bpw=2.0, of=m["I"], cof=0.0)]  # I's read under its old tag
        segs, held = CP.reanchor(W, {k: an0})
        assert not held and CP.anchor_at(segs, k, x["step"] - 1) is an0
        a1 = CP.anchor_at(segs, k, x["step"]); assert a1["kld"] == 0.0123 and abs(a1["cof"] - x["C"]) < 1e-12
        est = lambda st: CP.anchor_at(segs, k, st)["kld"] * math.exp([c for s, c in ch if s <= st][-1] - CP.anchor_at(segs, k, st)["cof"])
        assert abs(est(x["step"]) - 0.0123) < 1e-12 and abs(est(x["step"] - 1) - 0.01 * math.exp(before)) < 1e-12


# ------------------------------------------------------------------------------------------------ identity (CONF reselect absent)
def tree_base(tmp):
    return TC.baseline_tree(tmp)


@test
def identity_reselect_absent_live_on_result():
    """CONF reselect absent: HEAD and this tree emit the same events on the on_result replay of the live (confirm-mode) state"""
    if not os.path.exists(f"{SNAP}/drv/state.jsonl"): print(f"   (SKIP: no snapshot at {SNAP})"); return
    with tempfile.TemporaryDirectory() as tmp:
        base = tree_base(tmp); outs = []
        if base is None: print("   (SKIP: no baseline tree: set RXM_POOL_BASE)"); return
        for tree, name in ((base, "head"), (TC.TREE, "work")):
            r = subprocess.run([sys.executable, "-I", f"{V3}/pool188/replay_events.py", "--tree", tree, "--snap", SNAP, "--out", f"{tmp}/{name}.json"],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(json.load(open(f"{tmp}/{name}.json")))
        assert outs[0]["captured"] == outs[1]["captured"], "CONF reselect absent: events differ from HEAD"
        assert not any(e["ev"].startswith("reselect_") for e in outs[1]["captured"])
        print(f"   {len(outs[1]['captured'])} events identical ({sum(e['ev'] == 'decision' for e in outs[1]['captured'])} decisions, "
              f"{sum(e['ev'] == 'tentative' for e in outs[1]['captured'])} tentatives; state sha256 {outs[1]['snap_state_sha256'][:16]})")


@test
def identity_reselect_absent_live_steps():
    """the same for Driver.step() on a copy of the live state (go-live, confirmation, pool rescore, §5, report): events, queue, log,
    incumbents, pools and the status line (after its time stamp) identical over 3 steps"""
    if not os.path.exists(f"{SNAP}/drv/state.jsonl"): print(f"   (SKIP: no snapshot at {SNAP})"); return
    with tempfile.TemporaryDirectory() as tmp:
        base = tree_base(tmp); outs = []
        if base is None: print("   (SKIP: no baseline tree: set RXM_POOL_BASE)"); return
        for tree, name in ((base, "head"), (TC.TREE, "work")):
            r = subprocess.run([sys.executable, "-I", f"{HERE}/step_events.py", "--tree", tree, "--snap", SNAP, "--out", f"{tmp}/{name}.json"],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(json.load(open(f"{tmp}/{name}.json")))
        h, w = outs
        for k in ("events", "queued", "log", "inc", "pool"): assert h[k] == w[k], f"CONF reselect absent: {k} differ from HEAD"
        assert h["status"].split(" | ", 1)[1] == w["status"].split(" | ", 1)[1]
        print(f"   {len(w['events'])} events, {len(w['queued'])} queued identical over 3 steps")


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
