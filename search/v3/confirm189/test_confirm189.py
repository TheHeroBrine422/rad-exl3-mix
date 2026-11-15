# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_confirm189.py -- the two-stage accept of the #183 search driver (#189, RULES-confirm.md): tentative -> confirmation
on the paired slice -> accept / join / reject, the incumbent changing mid-confirmation (rule 3), a restart mid-confirmation,
the go-live check (rule 7), rotation never landing on a confirmation slice (rule 5), paused sizes untouched (rule 6), the
plot chaining the stage-2 log r (rule 4), and CONF "confirm" absent = the HEAD driver event for event on recorded replays.

  python3 -I search/v3/confirm189/test_confirm189.py      (exit 0 iff every test passes; CPU, ~30 s)

The driver runs its real loop (Driver.step: ingest, rotate, pool, go-live, confirmation; no fits, so no generation) against a
fake host that scores every queued eval whole from a per-model factor table (deterministic per-document noise per model and
slice, so a rotation's retag of a model gets the same sums). Identity: env RXM_POOL_SNAP (the #183 snapshot, default
/workspace/rad-local/workers/w165/snap) and RXM_CONF_SNAP (a copy of the live K = 3 run after the s186 switch, default
/workspace/rad-local/workers/w167/snap189); baseline tree RXM_POOL_BASE or `git archive HEAD`. A missing snapshot is SKIPPED."""
import io, json, math, os, random, subprocess, sys, tarfile, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
SEARCH = os.path.dirname(V3)
TREE = os.path.dirname(SEARCH)
sys.path.insert(0, V3)
import menu as MN          # noqa: E402
import moves as MV         # noqa: E402
import slices as SL        # noqa: E402

SNAP183 = os.environ.get("RXM_POOL_SNAP", "/workspace/rad-local/workers/w165/snap")
SNAP189 = os.environ.get("RXM_CONF_SNAP", "/workspace/rad-local/workers/w167/snap189")
T = []
QUIET = os.environ.get("RXM_TEST_VERBOSE") is None


def test(f): T.append(f); return f


# ------------------------------------------------------------------------------------------------ environment
def slice_docs(tag, nd, seed):
    out = []
    for i in range(nd):
        p = "c" if i % 2 == 0 else "x"
        out.append({"part": p, "src": i, "id": f"c:{1000 * (seed + 1) + i}" if p == "c" else f"x:{tag}:{i}", "tokens": 4096, "rows": 4095, "stratum": "N" if i % 6 == 5 else "L"})
    return out


def make_env(tmp, K=3, sizes=(2.0, 3.0), confirm=True, golive=None, cf_sizes=None, nd=24, roles=True, statuses=None):
    import driver as DR
    os.makedirs(tmp, exist_ok=True)
    json.dump(MN.synth(n_layers=4), open(f"{tmp}/menu.json", "w"))
    st = dict({"s186a": "active", "s186b": "staged", "s186c": "ready", "s186d": "staged"}, **(statuses or {}))
    reg = {"slices": {}, "doc_decisions": {}}
    for i, n in enumerate(("s186a", "s186b", "s186c", "s186d")):
        reg["slices"][n] = {"parts": {}, "mix": f"/gold/{n}", "alias": f"/data/q27/w164/slices/golden-{n}", "host_dirs": {}, "docs": slice_docs(n, nd, i),
                            "status": st[n], "decisions": {}, "built": f"2026-11-14T00:0{i}:00Z"}
        if roles and n in ("s186b", "s186d"): reg["slices"][n]["role"] = "confirm"
    json.dump(reg, open(f"{tmp}/registry.json", "w"))
    conf = {"root": f"{tmp}/drv", "menu": f"{tmp}/menu.json", "registry": f"{tmp}/registry.json", "base": "V3-T4", "sizes": list(sizes),
            "evals": f"{tmp}/evals.jsonl", "results": f"{tmp}/results.jsonl", "docs": f"{tmp}/docs", "tag_prefix": "t189",
            "rule": {"kind": "full"}, "cap": None, "guard": None, "size_adjust": False, "pool": K, "queue_depth": 4, "go": True,
            "bytes_offset0": 0, "monitor": None, "min_rows_fit": 10 ** 9, "refit_every": 10 ** 9, "rotate_after": 10 ** 6, "poll_s": 0}
    if confirm:
        conf["confirm"] = {"z": -1.0, "pairs": [["s186a", "s186b"], ["s186c", "s186d"]], "sizes": list(cf_sizes or sizes)}
        if golive: conf["confirm"]["golive_from_line"] = golive
    os.makedirs(f"{tmp}/docs", exist_ok=True)
    json.dump(conf, open(f"{tmp}/conf.json", "w"))
    return f"{tmp}/conf.json"


def driver(cp):
    import driver as DR
    if QUIET:
        so = sys.stdout; sys.stdout = open(os.devnull, "w")
        try: return DR.Driver(cp)
        finally: sys.stdout.close(); sys.stdout = so
    return DR.Driver(cp)


def quiet(f, *a):
    if not QUIET: return f(*a)
    so = sys.stdout; sys.stdout = open(os.devnull, "w")
    try: return f(*a)
    finally: sys.stdout.close(); sys.stdout = so


class Host:
    """the eval queue's fake hosts: every queued eval is scored whole; a model (assignment hash) has a KL factor per slice
    (default 1.0; key "*" = every slice); per-document sums = slice base x factor x exp(noise(model, slice, doc))"""

    def __init__(self, cp):
        self.c = json.load(open(cp)); self.f = {}; self.fail = set()

    def set(self, D, tag, **fs): self.f[D.sub[tag]["hash"]] = fs

    def setx(self, h, **fs): self.f[h] = fs

    def answered(self):
        p = self.c["results"]
        return {json.loads(l)["tag"] for l in open(p)} if os.path.exists(p) else set()

    def queued(self):
        p = self.c["evals"]
        return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

    def run(self, D, only=None, skip=()):
        """score the queued, unanswered evals (only: tags or a predicate) -> the tags scored"""
        done = self.answered(); out = []
        for q in self.queued():
            t = q["tag"]
            if t in done or t in skip or (only is not None and not (only(t) if callable(only) else t in only)): continue
            sl = q["slices"][0]["name"]; h = D.sub[t]["hash"]
            with open(self.c["results"], "a") as f:
                if t in self.fail:
                    self.fail.discard(t); f.write(json.dumps({"tag": t, "slice": sl, "fail": True, "err": "test"}) + "\n"); out.append(t); continue
                fs = self.f.get(h, {}); fac = fs.get(sl, fs.get("*", 1.0))
                N = len(D.reg.d["slices"][sl]["docs"]); rb = random.Random(f"base|{sl}"); rn = random.Random(f"{h}|{sl}")
                base = [math.exp(rb.gauss(0, 0.6)) for _ in range(N)]
                docs = [[b * fac * math.exp(rn.gauss(0, 0.02)), 1000.0] for b in base]
                json.dump({"m": docs}, open(f"{self.c['docs']}/{t}-{sl}.json", "w"))
                f.write(json.dumps({"tag": t, "slice": sl, "kld": sum(d[0] for d in docs) / (1000.0 * N), "bytes": None, "bpw": D.sub[t]["bpw"]}) + "\n")
            out.append(t)
        return out


def variant(D, k, j):
    """a distinct assignment: uniform level k with trunk tensor j one level up (or down at the top)"""
    lv = D.M.levels(); a = D.M.uniform(lv[k], None, None)
    w = [t["w"] for t in D.M.T if t["w"] in a and t["w"] not in (MN.HEAD, MN.EMB)][j]
    a[w] = lv[k + 1] if k + 1 < len(lv) else lv[k - 1]
    return a


def cand(D, size, k, j, kind="partial"):
    """a stage-1 candidate on the active slice (submitted like generate(): early-stop block vs the reference)"""
    a = variant(D, k, j); tag = D.newtag(size, kind)
    quiet(D.submit, tag, a, size, kind, D.reg.active(), True, D.reference(size), {"audit": False})
    return tag


def steps(D, H, n=1, **kw):
    for _ in range(n):
        H.run(D, **kw); quiet(D.step)


def events(D): return [json.loads(l) for l in open(f"{D.R}/state.jsonl")]


def boot(cp, K=3):
    """a driver with every size's incumbent scored on the active slice (init assignments, factor 1)"""
    D = driver(cp); H = Host(cp)
    quiet(D.step); steps(D, H, 2)
    for z in D.sizes: assert D.inc.get(z) in D.res and D.scored_on(z, D.reg.active()), (z, D.inc)
    return D, H


def confirm_evals(D, tag=None):
    return [t for t, e in D.sub.items() if e["kind"] == "confirm" and (tag is None or D.sub[t]["hash"] == D.sub[tag]["hash"])]


# ------------------------------------------------------------------------------------------------ paths
@test
def tentative_then_confirmed_accept():
    """K = 3: a stage-1 accept is tentative (incumbent and pool unchanged, the search slice counts the decision); its candidate
    and the incumbent are scored whole on s186b (kind confirm, no early stop); z < -1 -> accepted: incumbent + pool join;
    the stage-2 decision carries the stage-2 log r (logr) and stage 1's (logr1); stage 2 never counts towards rotate_after;
    both stages are fit rows"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp); D, H = boot(cp); I = D.inc[2.0]
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.90, s186b=0.94)
        steps(D, H)                                                         # stage 1
        assert A in D.tent and D.inc[2.0] == I and D.members(2.0) == [I], (D.tent, D.inc, D.members(2.0))
        assert D.reg.d["slices"]["s186a"]["decisions"] == {"2.0": 1}
        ce = confirm_evals(D)
        assert len(ce) == 2 and {D.sub[x]["of"] for x in ce} == {A, I} and all(D.sub[x]["slice"] == "s186b" for x in ce), ce
        q = {x["tag"]: x for x in H.queued()}
        assert all("early_stop" not in q[x] and q[x]["slices"][0]["parts"]["m"] == "/data/q27/w164/slices/golden-s186b" for x in ce)
        steps(D, H)                                                         # stage 2
        assert not D.tent and D.inc[2.0] == A and sorted(D.members(2.0)) == sorted([A, I]), (D.inc, D.members(2.0))
        ev = events(D)
        d2 = [e for e in ev if e["ev"] == "decision" and e.get("stage") == 2]
        cf = [e for e in ev if e["ev"] == "confirm"]
        assert len(d2) == 1 and d2[0]["decision"] == "accept" and d2[0]["z"] < -1 and abs(d2[0]["logr"] - math.log(0.94)) < 0.02, d2
        assert abs(d2[0]["logr1"] - math.log(0.90)) < 0.02 and cf[0]["verdict"] == "accept" and cf[0]["ref"] == I
        assert not [e for e in ev if e["ev"] == "decision" and e.get("tag") == A and e.get("stage") != 2], "stage 1 wrote a decision event"
        assert D.reg.d["slices"]["s186a"]["decisions"] == {"2.0": 1} and D.reg.d["slices"]["s186b"]["decisions"] == {}
        assert D.reg.d["slices"]["s186b"]["confirm_decisions"] == {"2.0": 1}
        assert any(k[1].startswith("s186b|") for k in D.obs) and any(k[1].startswith("s186a|") for k in D.obs)     # rule 4: fit rows
        assert D.kind_stats["partial"] == [1, 1], D.kind_stats["partial"]
        pe = [e for e in ev if e["ev"] == "pool" and e.get("why") == "join"]
        assert len(pe) == 1 and pe[0]["join"] == "add" and pe[0]["tag"] == A


@test
def tentative_then_rejected():
    """z >= -1 on the confirmation slice -> "reject" with why "confirm"; nothing changes; kind stats count a reject"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp); D, H = boot(cp); I = D.inc[3.0]
        A = cand(D, 3.0, 4, 1); H.set(D, A, s186a=0.92, s186b=1.03)
        steps(D, H, 3)
        assert not D.tent and D.inc[3.0] == I and D.members(3.0) == [I]
        d2 = [e for e in events(D) if e["ev"] == "decision" and e.get("stage") == 2]
        assert d2[0]["decision"] == "reject" and d2[0]["why"] == "confirm" and d2[0]["z"] > -1 and d2[0]["logr1"] < 0, d2
        assert D.kind_stats["partial"] == [0, 1]


@test
def k1_accept_and_reject():
    """K = 1 (no pool): the same two paths, no pool events"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp, K=1); D, H = boot(cp); I = D.inc[2.0]
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.9, s186b=0.95)
        B = cand(D, 2.0, 2, 1); H.set(D, B, s186a=0.85, s186b=1.02)
        steps(D, H, 4)
        assert D.inc[2.0] == A and not D.tent
        v = {e["tag"]: e["verdict"] for e in events(D) if e["ev"] == "confirm"}
        assert v == {A: "accept", B: "reject"}, v
        assert not any(e["ev"] == "pool" for e in events(D))


@test
def pool_join_confirmed_as_join():
    """a full pool: a candidate that beats the worst member only is tentative with target "worst"; confirmed against the worst
    on s186b -> decision "join" (not "accept": the plot does not chain it), the worst is replaced, the incumbent stays"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp); D, H = boot(cp); I = D.inc[2.0]
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.90, s186b=0.90)
        steps(D, H, 3)
        Bt = cand(D, 2.0, 2, 1); H.set(D, Bt, s186a=0.85, s186b=0.85)
        steps(D, H, 3)
        assert D.inc[2.0] == Bt and len(D.members(2.0)) == 3, D.members(2.0)
        C = cand(D, 2.0, 2, 2); H.set(D, C, s186a=0.95, s186b=0.96)       # beats I (worst, 1.0), not Bt (0.85)
        steps(D, H)
        assert D.tent[C]["target"] == "worst" and D.tent[C]["vs"] == I, D.tent[C]
        steps(D, H, 2)
        assert D.inc[2.0] == Bt and sorted(D.members(2.0)) == sorted([A, Bt, C]), D.members(2.0)
        d2 = [e for e in events(D) if e["ev"] == "decision" and e.get("stage") == 2 and e["tag"] == C]
        assert d2[0]["decision"] == "join" and d2[0]["vs"] == I, d2


@test
def incumbent_changed_mid_confirmation():
    """rule 3: A and B tentative together; A is confirmed first and becomes incumbent; B is then compared with A on s186b,
    from A's cached confirmation eval (no new eval of A); a rotation's retag of the incumbent is the same model (cached)"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp); D, H = boot(cp); I = D.inc[2.0]
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.90, s186b=0.90)
        B = cand(D, 2.0, 2, 1); H.set(D, B, s186a=0.92, s186b=0.93)        # beats I, not A, on s186b
        steps(D, H)
        assert A in D.tent and B in D.tent
        ceB = [x for x in confirm_evals(D, B)]
        steps(D, H, skip=set(ceB))                                         # A's evals scored, B's withheld
        assert D.inc[2.0] == A and B in D.tent
        steps(D, H, 2)
        assert not D.tent and D.inc[2.0] == A
        cf = {e["tag"]: e for e in events(D) if e["ev"] == "confirm"}
        assert cf[B]["ref"] == A and cf[B]["ref_changed"] is True and cf[B]["verdict"] in ("join", "reject"), cf[B]
        assert len(confirm_evals(D, A)) == 1 and len(confirm_evals(D, I)) == 1, (confirm_evals(D, A), confirm_evals(D, I))
        assert any(e["ev"] == "confirm_submit" and e["why"] == "reference changed" for e in events(D)) is False   # cached, no submit


@test
def rotation_mid_confirmation_uses_the_cached_retag():
    """a tentative on s186a whose confirmation spans a rotation to s186c: still confirmed on s186b (its pair), the incumbent's
    rescore tag on s186c has the same model hash (its s186b eval is reused), and K = 3 waits for the pool's rescore first"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp); D, H = boot(cp); I = D.inc[3.0]
        A = cand(D, 3.0, 4, 0); H.set(D, A, s186a=0.9, s186b=0.92)
        steps(D, H)
        ceA = confirm_evals(D, A); ceI = confirm_evals(D, I)
        assert len(ceA) == 1 and len(ceI) == 1
        D.c["rotate_after"] = 1; quiet(D.step); D.c["rotate_after"] = 10 ** 6
        assert D.reg.active() == "s186c" and D.inc[3.0] != I and D.sub[D.inc[3.0]]["hash"] == D.sub[I]["hash"], D.inc
        H.run(D, only=set(ceA + ceI)); quiet(D.step)                        # confirmation evals in, the pool not yet rescored
        assert A in D.tent, "decided before the pool was scored on the new slice"
        steps(D, H, 3)
        assert not D.tent and D.sub[D.inc[3.0]]["hash"] == D.sub[A]["hash"], (D.tent, D.inc)
        assert len(confirm_evals(D, I)) == 1 and all(D.sub[x]["slice"] == "s186b" for x in confirm_evals(D))
        cf = [e for e in events(D) if e["ev"] == "confirm"][0]
        assert cf["slice"] == "s186b" and cf["ref_changed"] is False and cf["verdict"] == "accept", cf


@test
def restart_mid_confirmation():
    """a driver restarted with confirmations outstanding rebuilds the tentatives and the eval cache from the state (no second
    submit), decides them when the evals arrive; a second restart has the same incumbent, pool, counts and no pending"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp); D, H = boot(cp); I = D.inc[2.0]
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.9, s186b=0.93)
        B = cand(D, 3.0, 4, 0); H.set(D, B, s186a=0.9, s186b=1.05)
        H.run(D); quiet(D.step)
        assert set(D.tent) == {A, B}
        nq = len(H.queued())
        D2 = driver(cp)
        assert set(D2.tent) == {A, B} and D2.cfx == D.cfx and D2.dec == D.dec, (D2.tent.keys(), D2.dec, D.dec)
        quiet(D2.step); assert len(H.queued()) == nq, "a restart resubmitted a confirmation eval"
        steps(D2, H, 2)
        assert not D2.tent and D2.inc[2.0] == A and D2.inc[3.0] == D.inc[3.0]
        D3 = driver(cp)
        assert not D3.tent and D3.inc == D2.inc and {z: sorted(D3.members(z)) for z in D3.sizes} == {z: sorted(D2.members(z)) for z in D2.sizes}
        assert dict(D3.dec) == dict(D2.dec) and {k: list(v) for k, v in D3.kind_stats.items()} == {k: list(v) for k, v in D2.kind_stats.items()}


@test
def failed_confirmation_eval():
    """a failed confirmation eval is resubmitted once (CONF tries 2); two failures -> reject (confirm (eval failed))"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp, K=1); D, H = boot(cp)
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.9, s186b=0.9)
        steps(D, H)
        x1 = confirm_evals(D, A)[0]; H.fail.add(x1)
        steps(D, H)
        x2 = [x for x in confirm_evals(D, A) if x != x1]
        assert len(x2) == 1 and A in D.tent, confirm_evals(D, A)
        H.fail.add(x2[0]); steps(D, H, 2)
        assert not D.tent
        d2 = [e for e in events(D) if e["ev"] == "decision" and e.get("stage") == 2][0]
        assert d2["decision"] == "reject" and "eval failed" in d2["why"] and d2["logr"] is None, d2


@test
def paused_size_untouched():
    """rule 6: a size outside CONF confirm.sizes accepts at stage 1 as today (decision accept, no tentative, no confirm eval)"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp, K=1, cf_sizes=[3.0]); D, H = boot(cp)
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.9)
        steps(D, H)
        assert D.inc[2.0] == A and not D.tent and not confirm_evals(D)
        assert [e["decision"] for e in events(D) if e["ev"] == "decision" and e["tag"] == A] == ["accept"]


@test
def golive_revert_and_keep():
    """rule 7: started with CONF golive_from_line = the line after the switch rescore: 2.0's current incumbent (accepted after the
    switch) holds on s186b -> kept; 3.0's does not -> back to the switch-time model, pool = that model; an unchanged size ->
    "same" (no eval). A tentative of 3.0 waits for the check, then is confirmed against the reverted incumbent from the cached
    eval. One-shot: a restart does not check again"""
    with tempfile.TemporaryDirectory() as tmp:
        cp0 = make_env(tmp, sizes=(2.0, 2.5, 3.0), confirm=False); D, H = boot(cp0)
        L = len(open(f"{D.R}/state.jsonl").readlines())
        sw = dict(D.inc)
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.9, s186b=0.92)
        B = cand(D, 3.0, 4, 0); H.set(D, B, s186a=0.9, s186b=1.04)
        steps(D, H, 2)
        assert D.inc[2.0] == A and D.inc[3.0] == B                          # accepted without confirmation (before go-live)
        c = json.load(open(cp0)); c["confirm"] = {"z": -1.0, "pairs": [["s186a", "s186b"], ["s186c", "s186d"]], "sizes": [2.0, 2.5, 3.0],
                                                  "golive_from_line": L}
        json.dump(c, open(cp0, "w"))
        D2 = driver(cp0); H = Host(cp0); H.f = {}
        for t, fs in ((A, {"s186a": 0.9, "s186b": 0.92}), (B, {"s186a": 0.9, "s186b": 1.04})): H.setx(D2.sub[t]["hash"], **fs)
        C = cand(D2, 3.0, 4, 1); H.setx(D2.sub[C]["hash"], s186a=0.85, s186b=0.95)
        quiet(D2.step)                                                      # go-live starts; C's stage 1 is not scored yet
        g = [e for e in events(D2) if e["ev"] == "golive"]
        assert len(g) == 1 and {p["size"] for p in g[0]["plan"]} == {2.0, 3.0} and [x["size"] for x in g[0]["same"]] == [2.5], g
        sw3 = lambda t: D2.sub[t]["kind"] == "confirm" and D2.sub[t]["hash"] == D2.sub[sw[3.0]]["hash"]
        steps(D2, H, 3, only=lambda t: not sw3(t))                          # everything but the switch-time 3.0 model's eval
        assert C in D2.tent and D2.golive_wait(3.0) and D2.inc[3.0] == B, "a tentative was decided before its size's go-live check"
        assert all(x in D2.res for x in confirm_evals(D2, C) + confirm_evals(D2, B))
        steps(D2, H, 3)
        r = {e["size"]: e for e in events(D2) if e["ev"] == "golive_size"}
        assert r[2.0]["verdict"] == "kept" and r[3.0]["verdict"] == "reverted" and r[2.5]["verdict"] == "same", {z: v["verdict"] for z, v in r.items()}
        assert D2.inc[2.0] == A and D2.golive_end
        cfC = [e for e in events(D2) if e["ev"] == "confirm" and e["tag"] == C][0]
        assert cfC["ref"] == sw[3.0] and cfC["verdict"] == "accept", cfC
        assert D2.inc[3.0] == C and sorted(D2.members(3.0)) == sorted([sw[3.0], C]), (D2.inc[3.0], D2.members(3.0))
        assert len(confirm_evals(D2, sw[3.0])) == 1 and len(confirm_evals(D2, sw[2.5])) == 0
        pe = [e for e in events(D2) if e["ev"] == "pool" and e.get("why") == "golive"]
        assert len(pe) == 1 and pe[0]["members"] == [sw[3.0]]
        n0 = len(events(D2)); D3 = driver(cp0); quiet(D3.step)
        assert D3.golive_end and not any(e["ev"] == "golive" for e in events(D3)[n0:]) and D3.inc == D2.inc


@test
def golive_restart_mid_check():
    """a restart while the go-live evals are outstanding: the plan is replayed, nothing is resubmitted, the check completes"""
    with tempfile.TemporaryDirectory() as tmp:
        cp0 = make_env(tmp, sizes=(2.0,), confirm=False); D, H = boot(cp0)
        L = len(open(f"{D.R}/state.jsonl").readlines()); I = D.inc[2.0]
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.9, s186b=1.05); steps(D, H, 2)
        c = json.load(open(cp0)); c["confirm"] = {"z": -1.0, "pairs": [["s186a", "s186b"]], "sizes": [2.0], "golive_from_line": L}
        json.dump(c, open(cp0, "w"))
        D2 = driver(cp0); quiet(D2.step); nq = len(H.queued())
        D3 = driver(cp0); quiet(D3.step)
        assert len(H.queued()) == nq and D3.golive is not None and not D3.golive_end
        H.setx(D3.sub[A]["hash"], s186a=0.9, s186b=1.05); steps(D3, H, 2)
        assert D3.golive_end and D3.inc[2.0] == I


# ------------------------------------------------------------------------------------------------ registry / rotation
@test
def rotation_skips_confirm_slices():
    """rule 5: a slice with role confirm is never activated, even if marked ready; with only such a slice ready no rotation
    happens (and no rotate event to None); the driver refuses CONF confirm without the roles; slices.py role sets / clears it;
    draw_merged never draws a confirmation slice's calib docs"""
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp + "/a", statuses={"s186b": "ready", "s186c": "ready"})
        R = SL.Registry(json.load(open(cp))["registry"])
        assert R.ready() == ["s186c"] and R.activate_next() == "s186c" and R.activate_next() is None
        cp = make_env(tmp + "/b", K=1, statuses={"s186b": "ready", "s186c": "staged"})
        D, H = boot(cp); D.c["rotate_after"] = 1
        A = cand(D, 2.0, 2, 0); steps(D, H, 2)
        assert D.reg.active() == "s186a" and not any(e["ev"] == "rotate" and e.get("from") for e in events(D))
        try: driver(make_env(tmp + "/c", roles=False)); raise AssertionError("started without roles")
        except SystemExit as e: assert "role" in str(e), e
        reg = json.load(open(cp))["registry"]
        quiet(SL.main, ["role", reg, "s186c", "confirm"]); assert SL.Registry(reg).role("s186c") == "confirm"
        quiet(SL.main, ["role", reg, "s186c", "-"]); assert SL.Registry(reg).role("s186c") == "search"
        R = SL.Registry(reg)
        R.d["slices"]["s186b"]["docs"][0]["id"] = "c:3"; R.d["slices"]["s186d"]["docs"][0]["id"] = "c:5"
        R.d["slices"]["s186d"].pop("role"); R.d["slices"]["s186d"]["status"] = "retired"
        open(f"{tmp}/led.jsonl", "w").close()
        pick, _, _ = SL.draw_merged(R, f"{tmp}/led.jsonl", ["search"] * 8, 8, 0)
        assert 3 not in pick and 5 in pick, pick


# ------------------------------------------------------------------------------------------------ the plot (convplot4, unchanged)
@test
def plot_chains_stage2_logr():
    """convplot4.walk on a confirm-mode state: the training estimate chains each confirmed accept's stage-2 log r (not stage 1's);
    joins and confirmation rejects are not chained; the tentatives are not plotted"""
    sys.path.insert(0, SEARCH)
    try: import matplotlib  # noqa: F401
    except ImportError:                                                     # walk() needs none of it (python -I hides user site)
        import types; mp = types.ModuleType("matplotlib"); mp.use = lambda *a, **k: None; mp.pyplot = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = mp; sys.modules["matplotlib.pyplot"] = mp.pyplot
    import convplot4 as CP
    with tempfile.TemporaryDirectory() as tmp:
        cp = make_env(tmp); D, H = boot(cp)
        A = cand(D, 2.0, 2, 0); H.set(D, A, s186a=0.90, s186b=0.95); steps(D, H, 3)
        B = cand(D, 2.0, 2, 1); H.set(D, B, s186a=0.85, s186b=1.02); steps(D, H, 3)
        Cc = cand(D, 2.0, 2, 2); H.set(D, Cc, s186a=0.80, s186b=0.90); steps(D, H, 3)
        W = CP.walk(f"{D.R}/state.jsonl")
        d2 = {e["tag"]: e for e in events(D) if e["ev"] == "decision" and e.get("stage") == 2}
        acc = [d2[t]["logr"] for t in (A, Cc) if d2[t]["decision"] == "accept"]
        assert len(acc) == 2 and d2[B]["decision"] == "reject", {t: e["decision"] for t, e in d2.items()}
        chain = W["accepts"][(2.0, "f16")]
        assert abs(chain[-1][1] - sum(acc)) < 1e-12, (chain, acc)
        assert abs(sum(acc) - (d2[A]["logr1"] + d2[Cc]["logr1"])) > 0.02     # stage 1 would give another chain
        assert {x["logr"] for x in W["decs"] if x["size"] == 2.0 and x["kind"] != "incumbent"} >= set(acc)


# ------------------------------------------------------------------------------------------------ identity (CONF confirm absent)
def baseline_tree(tmp):
    base = os.environ.get("RXM_POOL_BASE")
    if base: return base
    d = HERE
    while d != os.path.dirname(d) and not os.path.exists(os.path.join(d, ".git")): d = os.path.dirname(d)
    if not os.path.exists(os.path.join(d, ".git")): return None
    data = subprocess.run(["git", "-C", d, "archive", "HEAD", "search", "eval", "pipeline"], capture_output=True, check=True).stdout
    out = os.path.join(tmp, "head"); os.makedirs(out); tarfile.open(fileobj=io.BytesIO(data)).extractall(out, filter="data")
    return out


def identity(snap, need_recorded):
    if not os.path.exists(f"{snap}/drv/state.jsonl"): print(f"   (SKIP: no snapshot at {snap})"); return
    with tempfile.TemporaryDirectory() as tmp:
        base = baseline_tree(tmp)
        if base is None: print("   (SKIP: no baseline tree: set RXM_POOL_BASE)"); return
        outs = []
        for tree, name in ((base, "head"), (TREE, "work")):
            out = f"{tmp}/{name}.json"
            r = subprocess.run([sys.executable, "-I", f"{V3}/pool188/replay_events.py", "--tree", tree, "--snap", snap, "--out", out],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(json.load(open(out)))
        h, w = outs
        assert h["captured"] == w["captured"], "CONF confirm absent: events differ from HEAD"
        assert not any(e["ev"] in ("tentative", "confirm", "confirm_submit", "golive", "golive_size") for e in w["captured"])
        nd = sum(e["ev"] == "decision" for e in w["captured"])
        assert nd >= need_recorded, nd
        if need_recorded:                                                   # the #183 snapshot: the replay reproduces the record
            rec = {e["tag"]: e for e in h["recorded_decisions"]}
            for e in w["captured"]:
                if e["ev"] == "decision": assert e["decision"] == rec[e["tag"]]["decision"], e["tag"]
        print(f"   {len(w['captured'])} events identical ({nd} decisions; state sha256 {w['snap_state_sha256'][:16]})")


@test
def identity_confirm_absent_183():
    """CONF confirm absent: HEAD and this tree emit the same events on the recorded #183 replay (w166's identity test)"""
    identity(SNAP183, 200)


@test
def identity_confirm_absent_live_k3():
    """the same on a copy of the live K = 3 run after the s186 switch (pool, cross, re-seed state)"""
    identity(SNAP189, 0)


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
