# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_pool188.py -- the elite pool of the #183 search driver (#188, RULES-pool.md): the pure rules (poolrule.py),
the driver's join / rotation / re-seed / re-selection on synthetic per-document sums, the crossover move (moves.cross),
the identity of K = 1 with the HEAD driver on the recorded #183 replay (replay_events.py), cap = null with strata
(lead note 11-14 7:20 PM CST), and the early-stop reference of a full pool (the worst member).

  python3 -I search/v3/pool188/test_pool188.py     (exit 0 iff every test passes; CPU, about a minute)

The identity test needs the frozen #183 snapshot (env RXM_POOL_SNAP, default /workspace/rad-local/workers/w165/snap) and a
baseline tree (env RXM_POOL_BASE, default: `git archive HEAD` of the repository). Without the snapshot it is SKIPPED."""
import io, json, math, os, subprocess, sys, tarfile, tempfile, types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
TREE = os.path.dirname(os.path.dirname(V3))             # the tree under test: search/ eval/ pipeline/
sys.path.insert(0, V3)
import menu as MN          # noqa: E402
import moves as MV         # noqa: E402
import poolrule as PR      # noqa: E402
import seqrule as SR       # noqa: E402,F401

SNAP = os.environ.get("RXM_POOL_SNAP", "/workspace/rad-local/workers/w165/snap")
T = []


def test(f): T.append(f); return f


# ------------------------------------------------------------------------------------------------ pure rules
@test
def join_table():
    """RULES-pool §3: not full -> add iff it beats the incumbent; full -> replace iff it beats the worst or the incumbent"""
    assert PR.join_action(False, True, False) == ("add", True)
    assert PR.join_action(False, False, True) == (None, False)            # no worst while the pool is not full
    assert PR.join_action(True, False, True) == ("replace", False)        # beats the worst only: joins, incumbent stays
    assert PR.join_action(True, True, False) == ("replace", True)         # beats the incumbent: joins and becomes it
    assert PR.join_action(True, False, False) == (None, False)            # beats neither: rejected


@test
def worst_reseed_reselect_rules():
    """worst = largest size-adjusted statistic (tie -> non-incumbent); reseed = the first accepted in ranking order is the
    incumbent, the pool is the top K of the ranking with the incumbent kept in it; reselect = lowest accepted member"""
    assert PR.worst_of("I", {"I": 0.0, "A": -0.1, "B": 0.3}) == "B"
    assert PR.worst_of("I", {"I": 0.0, "A": 0.0}) == "A"
    new, pool = PR.reseed_pick("I", [("A", 0.2, False), ("B", -0.05, True), ("C", -0.01, True), ("D", -0.3, False)], 3)
    assert new == "B" and set(pool) == {"D", "B", "C"} and pool[0] == "D", (new, pool)     # D ranks first but fails the guard
    new, pool = PR.reseed_pick("I", [("A", -0.3, False), ("B", -0.1, True)], 2)
    assert new == "B" and set(pool) == {"A", "B"}, (new, pool)
    new, pool = PR.reseed_pick("I", [("A", -0.3, False), ("B", -0.2, False), ("C", -0.1, False)], 2)
    assert new == "I" and "I" in pool and len(pool) == 2, (new, pool)     # nothing accepted: the incumbent stays, in the pool
    assert PR.reselect_pick("I", [("A", -0.1, False), ("B", -0.05, True), ("C", -0.2, True)]) == "C"
    assert PR.reselect_pick("I", [("A", 0.1, False)]) is None


@test
def parent_draws():
    """RULES-pool §4: the incumbent w.p. 0.6, else uniform over the others; no RNG draw when there is no other member"""
    rng = np.random.default_rng(3); st = rng.bit_generator.state
    assert PR.choose_parent("I", ["I"], rng) == "I" and rng.bit_generator.state == st
    assert PR.choose_other("I", ["I"], rng) is None and rng.bit_generator.state == st
    rng = np.random.default_rng(4)
    picks = [PR.choose_parent("I", ["I", "A", "B"], rng) for _ in range(6000)]
    assert abs(picks.count("I") / 6000 - 0.6) < 0.03, picks.count("I")
    assert abs(picks.count("A") / 6000 - 0.2) < 0.03 and abs(picks.count("B") / 6000 - 0.2) < 0.03


# ------------------------------------------------------------------------------------------------ crossover (moves.cross)
class FakePred:
    """the move's view of a prediction: a bytes table, parent assignments o<k>, a gain table with no upgrade gains (so a
    wide window needs no repair) and the attributes cross / repair read"""

    def __init__(self, n_layers=5, per=3, nopt=4):
        self.W = [f"blk.{i // per}.t{i % per}" for i in range(n_layers * per)]
        self.D = types.SimpleNamespace(W=self.W)
        self.L = np.array([i // per for i in range(len(self.W))])
        self.B = [np.array([100.0 * (k + 1) * (1 + 0.1 * i) for k in range(nopt)]) for i in range(len(self.W))]
        self.M = types.SimpleNamespace(bytes=self.bytes)

    def bytes(self, a): return sum(self.B[i][int(a[w][1:])] for i, w in enumerate(self.W))

    def idx(self, a): return [int(a[w][1:]) for w in self.W]

    def assign(self, J, a0):
        a = dict(a0)
        for i, w in enumerate(self.W): a[w] = f"o{J[i]}"
        return a

    def lin(self, a):
        J = self.idx(a)
        # d log KL of option k vs the current one: positive for every k != current, so no move is an upgrade gain
        dl = [np.array([0.5 * abs(k - J[i]) * (1 + 0.01 * i) for k in range(len(self.B[i]))]) for i in range(len(self.W))]
        return J, dl


def fake_assign(pred, k): return {w: f"o{k}" for w in pred.W}


@test
def cross_units_and_byte_window():
    """RULES-pool §4: in a wide window each layer's tensors come from one parent; in a finite window every returned child is
    inside it; the move without a second parent returns None"""
    pred = FakePred(); a = fake_assign(pred, 1); b = fake_assign(pred, 2)
    c = MV.cross(pred, a, b, 0.0, 1e12, np.random.default_rng(5))
    J = pred.idx(c)
    assert set(J) == {1, 2}, J                                             # both parents contributed
    for layer in set(pred.L.tolist()):
        vals = {J[i] for i in range(len(J)) if pred.L[i] == layer}
        assert len(vals) == 1, (layer, vals)                               # one parent per layer
    lo_w, hi_w = pred.bytes(a) * 0.9, pred.bytes(b) * 1.1                 # a window both parents sit in
    got = 0
    for seed in range(40):
        ch = MV.generate_cross(pred, a, b, lo_w, hi_w, set(), np.random.default_rng(seed))
        if ch is not None: got += 1; assert lo_w <= pred.bytes(ch) <= hi_w
    assert got >= 20, got
    lo_n, hi_n = pred.bytes(a) * 0.999, pred.bytes(a) * 1.001             # a narrow window: children inside it, or none
    for seed in range(40):
        ch = MV.generate_cross(pred, a, b, lo_n, hi_n, set(), np.random.default_rng(seed))
        if ch is not None: assert lo_n <= pred.bytes(ch) <= hi_n
    assert MV.generate_cross(pred, a, None, lo_w, hi_w, set(), np.random.default_rng(1)) is None


# ------------------------------------------------------------------------------------------------ driver helpers
def make_driver(tmp, K=3, sizes=(2.0, 3.0), cap=0.2, reseed=None, act="s186a", strata=True, nd=12, cap_none=False):
    import driver as DR                                   # the tree under test (imported after sys.path is set above)
    os.makedirs(tmp, exist_ok=True)
    menu = MN.synth(n_layers=4)
    json.dump(menu, open(f"{tmp}/menu.json", "w"))
    parts = ["c"] * (nd // 2) + ["x"] * (nd - nd // 2)
    st = (["L", "N"] * nd)[:nd] if strata else None
    docs = [dict({"part": p, "src": i, "id": f"{p}:{i}", "tokens": 4096, "rows": 4095}, **({"stratum": st[i]} if st else {}))
            for i, p in enumerate(parts)]
    reg = {"slices": {act: {"parts": {}, "mix": "", "host_dirs": {}, "docs": docs, "status": "active", "decisions": {}, "built": "2026-11-14T00:00:00Z"},
                      "s186b": {"parts": {}, "mix": "", "host_dirs": {}, "docs": docs, "status": "ready", "decisions": {}, "built": "2026-11-14T01:00:00Z"}},
           "doc_decisions": {}}
    json.dump(reg, open(f"{tmp}/registry.json", "w"))
    conf = {"root": f"{tmp}/drv", "menu": f"{tmp}/menu.json", "registry": f"{tmp}/registry.json", "base": "V3-T4", "sizes": list(sizes),
            "evals": f"{tmp}/evals.jsonl", "results": f"{tmp}/results.jsonl", "docs": f"{tmp}/docs", "tag_prefix": "t188",
            "rule": {"kind": "full"}, "cap": None if cap_none else cap, "guard": "docmean", "size_adjust": False, "pool": K,
            "queue_depth": 4, "go": True, "bytes_offset0": 0, "monitor": None}
    if reseed: conf["reseed"] = reseed
    json.dump(conf, open(f"{tmp}/conf.json", "w"))
    return DR.Driver(f"{tmp}/conf.json")


def assign_for(D, k):
    """the k-th distinct uniform assignment of the synth menu"""
    lv = D.M.levels(); return D.M.uniform(lv[k % len(lv)], None, None)


def add_cand(D, tag, size, kind, slice_name, factor, k, base=None, ref=None, b=0.0):
    """a scored whole-slice candidate: per-document KL sums = factor x base (default 1.0 each), 1000 positions a document;
    written to the state like the live driver's submit and result events"""
    a = assign_for(D, k)
    json.dump(a, open(f"{D.R}/assign/{tag}.json", "w"))
    N = len(D.reg.d["slices"][slice_name]["docs"]); base = base or [1.0] * N
    e = {"ev": "submit", "tag": tag, "size": size, "kind": kind, "slice": slice_name, "hash": MV.ahash(a), "early": bool(ref), "ref": ref,
         "bpw": D.M.bpw(a), "b": b}
    D.rec(e); D.sub[tag] = e; D.seen.add(e["hash"])
    D.res[tag] = {"slice": slice_name, "docs": [base[i] * factor for i in range(N)], "pos": [1000.0] * N, "stopped": None,
                  "bytes": None, "bpw": None, "kld": factor}
    return tag


def score(D, tag):
    """the result event, then the decision path (as ingest does)"""
    D.rec({"ev": "result", "tag": tag, "r": D.res[tag]}); D.on_result(tag, D.res[tag])


def state_events(D):
    return [json.loads(l) for l in open(f"{D.R}/state.jsonl")]


@test
def pool_add_then_reject_then_add():
    """K = 3: while the pool is not full a candidate beating the incumbent joins (the old incumbent stays); a candidate
    beating neither is rejected; a full pool's worst is the member with the largest statistic"""
    with tempfile.TemporaryDirectory() as tmp:
        D = make_driver(tmp, K=3); act = "s186a"
        I = add_cand(D, "I", 2.0, "incumbent", act, 1.0, k=0); D.inc[2.0] = I
        add_cand(D, "A", 2.0, "partial", act, 0.9, k=1, ref=I); score(D, "A")
        assert D.inc[2.0] == "A" and sorted(D.members(2.0)) == ["A", "I"], (D.inc, D.members(2.0))
        add_cand(D, "B", 2.0, "pair", act, 1.2, k=2, ref="A"); score(D, "B")          # beats neither
        assert D.inc[2.0] == "A" and sorted(D.members(2.0)) == ["A", "I"]
        add_cand(D, "C", 2.0, "solve", act, 0.95, k=3, ref="A"); score(D, "C")       # not better than A: rejected
        assert D.inc[2.0] == "A" and sorted(D.members(2.0)) == ["A", "I"]
        add_cand(D, "E", 2.0, "pair", act, 0.8, k=4, ref="A"); score(D, "E")         # beats A while not full: joins, incumbent
        assert D.inc[2.0] == "E" and sorted(D.members(2.0)) == ["A", "E", "I"], D.members(2.0)
        add_cand(D, "F", 2.0, "block", act, 1.05, k=5, ref="E"); score(D, "F")       # full: worst = I (1.0); F beats neither
        assert D.inc[2.0] == "E" and sorted(D.members(2.0)) == ["A", "E", "I"]
        pe = [e for e in state_events(D) if e["ev"] == "pool"]
        assert len(pe) == 2 and pe[0]["join"] == "add" and sorted(pe[0]["members"]) == ["A", "I"], pe


@test
def pool_full_replace_worst_then_incumbent():
    """a full pool: a candidate beating the worst (not the incumbent) replaces it and the incumbent stays; a candidate beating
    the incumbent replaces the worst and becomes the incumbent"""
    with tempfile.TemporaryDirectory() as tmp:
        D = make_driver(tmp, K=3); act = "s186a"
        I = add_cand(D, "I", 2.0, "incumbent", act, 1.0, k=0); D.inc[2.0] = I
        add_cand(D, "A", 2.0, "partial", act, 0.9, k=1, ref=I); score(D, "A")
        add_cand(D, "E", 2.0, "pair", act, 0.8, k=2, ref="A"); score(D, "E")          # pool [E, A, I], incumbent E
        assert D.inc[2.0] == "E" and sorted(D.members(2.0)) == ["A", "E", "I"]
        add_cand(D, "Y", 2.0, "block", act, 0.95, k=3, ref="I"); score(D, "Y")         # worst I (1.0): Y beats I, not E
        assert D.inc[2.0] == "E" and sorted(D.members(2.0)) == ["A", "E", "Y"], D.members(2.0)
        add_cand(D, "Z", 2.0, "explore", act, 0.7, k=4, ref="Y"); score(D, "Z")        # beats E: replaces the worst
        assert D.inc[2.0] == "Z" and sorted(D.members(2.0)) == ["A", "E", "Z"], (D.inc[2.0], D.members(2.0))
        pe = [e for e in state_events(D) if e["ev"] == "pool"]
        assert pe[-1]["join"] == "replace" and pe[-1]["replaced"] == "Y", pe[-1]


@test
def early_stop_reference_is_the_worst_member():
    """a full pool's new candidates are submitted with the worst member as early-stop reference (the hosts' block), with the
    weights of that reference"""
    with tempfile.TemporaryDirectory() as tmp:
        D = make_driver(tmp, K=3); act = "s186a"
        I = add_cand(D, "I", 2.0, "incumbent", act, 1.0, k=0); D.inc[2.0] = I
        add_cand(D, "A", 2.0, "partial", act, 0.9, k=1, ref=I); score(D, "A")
        add_cand(D, "W", 2.0, "block", act, 1.3, k=2, ref="A")
        D.pool[2.0] = ["A", "I", "W"]; D.res["W"]["slice"] = act                       # a full pool (W is the worst)
        assert D.worst(2.0) == "W" and D.reference(2.0) == "W"
        D.submit("t188-2-solve-x", assign_for(D, 7), 2.0, "solve", act, early=True, ref_tag=D.reference(2.0), extra={"audit": False})
        line = [json.loads(l) for l in open(D.c["evals"])][-1]
        assert line["early_stop"]["ref_tag"] == "W"
        assert line["early_stop"]["ref"] == [[b, n] for b, n in zip(D.res["W"]["docs"], D.res["W"]["pos"])]
        assert line["early_stop"]["rule"].get("w") is not None                        # cap 0.2: the capped weights of W


@test
def rotation_rescore_then_reselect():
    """at a rotation every member is rescored on the new slice (nothing decided); the pool waits (scored_on) until all are;
    reselect then makes the accepted member with the lowest statistic the incumbent"""
    with tempfile.TemporaryDirectory() as tmp:
        D = make_driver(tmp, K=3); act = "s186a"
        I = add_cand(D, "I", 2.0, "incumbent", act, 1.0, k=0); D.inc[2.0] = I
        add_cand(D, "A", 2.0, "partial", act, 0.9, k=1, ref=I); score(D, "A")          # pool [I, A], incumbent A
        D.reg.d["slices"][act]["status"] = "retired"; D.reg.d["slices"]["s186b"]["status"] = "active"
        D.ensure_pool("s186b")
        mem = D.members(2.0)
        assert len(mem) == 2 and not set(mem) & {"I", "A"}, mem                        # rescoring tags replace the members
        assert not D.scored_on(2.0, "s186b")
        tA = [m for m in mem if D.sub[m]["hash"] == MV.ahash(D.assign("A"))][0]
        tI = [m for m in mem if m != tA][0]
        for m, f in ((tA, 0.9), (tI, 0.8)):                                              # on the new slice I is now the better one
            D.res[m] = {"slice": "s186b", "docs": [f] * 12, "pos": [1000.0] * 12, "stopped": None, "bytes": None, "bpw": D.sub[m]["bpw"], "kld": f}
            D.rec({"ev": "result", "tag": m, "r": D.res[m]})
            D.on_result(m, D.res[m])
        assert D.scored_on(2.0, "s186b")
        D.reselect("s186b")
        assert D.inc[2.0] == tI, (D.inc[2.0], tI)
        assert D.reselected.get(2.0) == "s186b"
        assert not any(e["ev"] == "decision" and e.get("tag") in (tA, tI) for e in state_events(D)), "rescoring decided something"


@test
def rotation_incumbent_rescore_survives_restart():
    """a rotation that rescores the incumbent swaps its tag in the pool: the pool event is written, so a restarted driver
    rebuilds the same pool and incumbent (regression: the swap was not in the state)"""
    with tempfile.TemporaryDirectory() as tmp:
        D = make_driver(tmp, K=3); act = "s186a"
        I = add_cand(D, "I", 2.0, "incumbent", act, 1.0, k=0); D.inc[2.0] = I; D.rec({"ev": "incumbent", "size": 2.0, "tag": I, "prev": None})
        add_cand(D, "A", 2.0, "partial", act, 0.9, k=1, ref=I); score(D, "A")
        D.reg.d["slices"][act]["status"] = "retired"; D.reg.d["slices"]["s186b"]["status"] = "active"
        D.ensure_pool("s186b")
        before = (D.inc[2.0], sorted(D.members(2.0)))
        assert D.inc[2.0] not in ("I", "A") and D.inc[2.0] in D.members(2.0), before
        D2 = type(D)(f"{tmp}/conf.json")
        assert (D2.inc.get(2.0), sorted(D2.members(2.0))) == before, ((D2.inc.get(2.0), sorted(D2.members(2.0))), before)


@test
def reseed_set_ranking_and_pool():
    """the re-seed: the frontier's top assignments (distinct, not the incumbent's) are scored in full on the slice; then the
    first accepted in ranking order is the incumbent and the pool is the top K with it in"""
    with tempfile.TemporaryDirectory() as tmp:
        act = "s186a"
        D = make_driver(tmp, K=3, sizes=(3.0,), act=act)
        I = add_cand(D, "I", 3.0, "incumbent", act, 1.0, k=0); D.inc[3.0] = I
        P0 = add_cand(D, "P0", 3.0, "solve", act, 0.97, k=1)
        P1 = add_cand(D, "P1", 3.0, "solve", act, 1.02, k=2)
        P2 = add_cand(D, "P2", 3.0, "solve", act, 0.9, k=3)
        json.dump({"3.0": {"top": [{"tags": [P0], "bpw": 3.0, "score": -0.1, "n": 1}, {"tags": [P1], "bpw": 3.0, "score": 0.0, "n": 1},
                                  {"tags": [P2], "bpw": 3.0, "score": 0.1, "n": 1}]}}, open(f"{tmp}/frontier.json", "w"))
        D.reseed_c = {"slice": act, "sizes": [3.0], "top": 2, "source": f"{tmp}/frontier.json"}
        D.reseed_step(act)
        plan = D.reseed_plan[3.0]
        assert len(plan) == 2 and D.reseed_wait(3.0, act) and not D.reseed_wait(3.0, "s186b"), plan
        # the plan's tags are new submits of P0 and P1 (the top two): score them with the sums of their sources
        for t, src, f in ((plan[0], P0, 0.97), (plan[1], P1, 1.02)):
            assert D.sub[t]["kind"] == "reseed" and D.assign(t) == D.assign(src)
            D.res[t] = {"slice": act, "docs": [f] * 12, "pos": [1000.0] * 12, "stopped": None, "bytes": None, "bpw": None, "kld": f}
            D.on_result(t, D.res[t])
        D.reseed_step(act)
        assert not D.reseed_wait(3.0, act) and 3.0 in D.reseed_done
        assert D.inc[3.0] == plan[0], (D.inc[3.0], plan)                              # P0 ranks first and passes the accept rule
        assert len(D.members(3.0)) == 3 and D.inc[3.0] in D.members(3.0), D.members(3.0)
        rs = [e for e in state_events(D) if e["ev"] == "reseed"]
        assert rs and rs[-1]["inc"] == plan[0] and rs[-1]["ranking"][0][0] == plan[0], rs[-1:]


@test
def cap_null_with_strata_is_plain_pooled_sums():
    """lead note 11-14 7:20 PM CST: cap = null + strata gives the plain per-(part, stratum) sums (no weights), and the
    early-stop rule carries no "w" but says "strata": true; cap 0.2 keeps the weights"""
    with tempfile.TemporaryDirectory() as tmp:
        act = "s186a"
        D = make_driver(tmp + "/n", K=1, cap_none=True); base = [1.0 + 0.1 * i for i in range(12)]
        add_cand(D, "I", 2.0, "incumbent", act, 1.0, k=0, base=base); D.inc[2.0] = "I"
        add_cand(D, "C", 2.0, "partial", act, 1.0, k=1, base=base)
        D.res["C"]["docs"] = [base[i] * (0.8 if i % 3 == 0 else 1.1) for i in range(12)]
        st = D.reg.strata_of(act); parts = D.reg.parts_of(act)
        rule = D.prule({"kind": "full"}, 0.0, D.res["I"]["docs"], parts, st)
        assert "w" not in rule and rule.get("strata") is True, rule
        P = D.paired(D.res["I"], D.res["C"], parts, strata=st)
        assert P.w is None and P.summary()["capped"] is False
        for (p, s), v in P.ps.items():
            idx = [i for i in range(12) if parts[i] == p and st[i] == s]
            assert math.isclose(v[0], sum(D.res["I"]["docs"][i] for i in idx)) and math.isclose(v[1], sum(D.res["C"]["docs"][i] for i in idx))
        D.submit("t188-cap-solve", assign_for(D, 7), 2.0, "solve", act, early=True, ref_tag="I", extra={})
        line = [json.loads(l) for l in open(D.c["evals"])][-1]
        assert "w" not in line["early_stop"]["rule"] and line["early_stop"]["rule"].get("strata") is True, line["early_stop"]["rule"]
        D2 = make_driver(tmp + "/c", K=1, cap=0.2)
        r2 = D2.prule({"kind": "full"}, 0.0, [1.0] * 12, D2.reg.parts_of(act), D2.reg.strata_of(act))
        assert r2.get("w") is not None and len(r2["w"]) == 12


@test
def pool_state_roundtrip():
    """a restarted driver rebuilds the pool and the incumbent from the state (submit, result, incumbent, pool events)"""
    with tempfile.TemporaryDirectory() as tmp:
        D = make_driver(tmp, K=3); act = "s186a"
        I = add_cand(D, "I", 2.0, "incumbent", act, 1.0, k=0); D.inc[2.0] = I; D.rec({"ev": "incumbent", "size": 2.0, "tag": I, "prev": None})
        add_cand(D, "A", 2.0, "partial", act, 0.9, k=1, ref=I); score(D, "A")
        add_cand(D, "Bq", 2.0, "pair", act, 0.85, k=2, ref="A"); score(D, "Bq")
        before = (D.inc[2.0], sorted(D.members(2.0)))
        D2 = type(D)(f"{tmp}/conf.json")
        assert (D2.inc.get(2.0), sorted(D2.members(2.0))) == before, ((D2.inc.get(2.0), sorted(D2.members(2.0))), before)


# ------------------------------------------------------------------------------------------------ identity with HEAD (K = 1)
def repo_root():
    """the git repository containing this file (walking up), or None (a scratch tree: set RXM_POOL_BASE)"""
    d = HERE
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".git")): return d
        d = os.path.dirname(d)
    return None


def baseline_tree(tmp):
    """the HEAD tree (search/ eval/ pipeline/) to compare with: env RXM_POOL_BASE, else `git archive HEAD` of the repository"""
    base = os.environ.get("RXM_POOL_BASE")
    if base: return base
    root = repo_root()
    if root is None: return None
    data = subprocess.run(["git", "-C", root, "archive", "HEAD", "search", "eval", "pipeline"], capture_output=True, check=True).stdout
    out = os.path.join(tmp, "head"); os.makedirs(out)
    tarfile.open(fileobj=io.BytesIO(data)).extractall(out)
    return out


@test
def k1_identity_on_the_recorded_replay():
    """K = 1 is today's driver: on the recorded #183 state the HEAD tree and this tree emit the same events, and the replayed
    decisions equal the recorded ones (the replay's own check)"""
    if not os.path.exists(f"{SNAP}/drv/state.jsonl"):
        print(f"   (SKIP: no snapshot at {SNAP})"); return
    with tempfile.TemporaryDirectory() as tmp:
        base = baseline_tree(tmp)
        if base is None: print("   (SKIP: no baseline tree: set RXM_POOL_BASE)"); return
        outs = []
        for tree, name in ((base, "head"), (TREE, "work")):
            out = f"{tmp}/{name}.json"
            r = subprocess.run([sys.executable, "-I", f"{HERE}/replay_events.py", "--tree", tree, "--snap", SNAP, "--out", out],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(json.load(open(out)))
        h, w = outs
        assert h["captured"] == w["captured"], "K = 1 events differ from HEAD"
        assert h["recorded_decisions"] == w["recorded_decisions"]
        rec = {e["tag"]: e for e in h["recorded_decisions"]}; n = 0
        for e in h["captured"]:
            if e["ev"] != "decision": continue
            r0 = rec[e["tag"]]; n += 1
            assert e["decision"] == r0["decision"], (e["tag"], e["decision"], r0["decision"])
            if e["decision"] != "stale":
                assert abs(e["logr"] - r0["logr"]) <= 1e-9 * max(1.0, abs(r0["logr"])), e["tag"]
        assert n > 200, n
        assert not any(e["ev"] in ("pool", "reseed", "reselect") for e in h["captured"] + w["captured"])


def main():
    bad = 0
    for f in T:
        try: f(); print(f"PASS {f.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1; print(f"FAIL {f.__name__}: {e!r}")
    print(f"{len(T) - bad} / {len(T)} PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
