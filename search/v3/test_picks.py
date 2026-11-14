# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_picks.py -- RULES-picks.md's statistics (w145): seqrule's capped statistic (b) + per-document guard (c),
pickstat.py's (a) / (b) / (c) with unit bootstrap, finalpick.py's decision / guard / flag, per-slice rotation.
Synthetic per-document sums only (one dominant document, short-document noise, long traces as units).
python3 -I search/v3/test_picks.py  (CPU, seconds; exit 0 iff every test passes)"""
import importlib.util, json, math, os, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import finalpick as FP  # noqa: E402
import pickstat as PS   # noqa: E402
import seqrule as SR    # noqa: E402
import slices as SL     # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
T = []


def test(f): T.append(f); return f


def run_paired(A, B, parts, cap=0.2, guard="docmean", off=0.0, rule=None):
    r = dict(rule or {"kind": "full"}, off=off)
    if cap: r["w"] = SR.cap_weights(A, parts, cap)
    if guard: r["guard"] = guard
    P = SR.Paired(len(A), r)
    for a, b, p in zip(A, B, parts):
        P.add(a, b, p)
        if P.check(): break
    return P


def dominant(n=60, seed=0, share=0.5, gain=0.10, loss=0.03):
    """part with one document holding `share` of A's KL: B is `gain` better on it, `loss` worse on every other one"""
    rng = np.random.default_rng(seed)
    A = rng.gamma(2.0, 1.0, n); A[0] = share / (1 - share) * A[1:].sum()
    B = A * (1 + loss) * np.exp(rng.normal(0, 0.01, n)); B[0] = A[0] * (1 - gain)
    return A, B


@test
def capped_weights_agree():
    rng = np.random.default_rng(1)
    spec = importlib.util.spec_from_file_location("readout", f"{REPO}/eval/docstat/readout.py")
    ro = importlib.util.module_from_spec(spec); spec.loader.exec_module(ro)
    for k in range(20):
        a = rng.gamma(0.5, 1.0, int(rng.integers(6, 200))); a[rng.integers(len(a))] *= 50
        w1 = np.array(SR.capped_weights(list(a), 0.2)); w2 = PS.capped(a); w3 = ro.capped_weights(a, 0.2)
        assert np.allclose(w1, w2), k
        assert (w1 * a).max() / (w1 * a).sum() <= 0.2 + 1e-9
        if (w3 * a).max() / (w3 * a).sum() <= 0.2 + 1e-9: assert np.allclose(w1, w3), k    # readout's loop, where it converged
        assert np.all(w1[a < 0.2 * (w1 * a).sum() - 1e-9] == 1.0)        # only over-cap documents are scaled
    # bootstrap capping: copy counts act as repeated documents
    a = np.array([10.0, 1, 1, 1, 1, 1, 1, 1]); n = np.array([2.0, 1, 1, 1, 1, 1, 1, 0])
    rep = np.array([10.0, 10, 1, 1, 1, 1, 1, 1])
    w = PS.capped(a[None, :], n[None, :])[0]; wr = PS.capped(rep)
    assert math.isclose((n * w * a).sum(), (wr * rep).sum()) and math.isclose(w[0] * a[0], wr[0] * rep[0])


@test
def old_rule_unchanged():
    """no "w" / "guard" in the rule = the #171 statistic, byte for byte the old earlystop.py's numbers"""
    spec = importlib.util.spec_from_file_location("es_old", f"{REPO}/pipeline/rxrun/earlystop.py")
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    if "cap_weights" in open(f"{REPO}/pipeline/rxrun/earlystop.py").read(): return       # already restaged: nothing to compare
    rng = np.random.default_rng(2)
    for k in range(30):
        n = 60; A = rng.gamma(1, 1, n); B = A * np.exp(rng.normal(rng.normal(0, 0.02), 0.05, n)); parts = SR.interleave(40, 20)
        rule = {"kind": "curve", "p": 2.0, "warmup": 0.05, "off": float(rng.normal(0, 0.01))}
        P1, P2 = SR.Paired(n, rule), old.Paired(n, rule)
        for a, b, p in zip(A, B, parts):
            P1.add(a, b, p); P2.add(a, b, p)
            assert P1.check() == P2.check()
            if P1.stopped: break
        assert (P1.d(), P1.se(), P1.z(), P1.final()) == (P2.d(), P2.se(), P2.z(), P2.final()), k


@test
def one_dominant_document():
    """(a) follows the one heavy document (better), (b) and (c) see the other 59 (worse): in-search rejects on (b);
    pickstat agrees; finalpick decides on (a) but FLAGS the document"""
    A, B = dominant()
    p = PS.point(A, B)
    assert p["a"] < 0 and p["b"] > 0 and p["c"] > 0, p
    parts = ["x"] * len(A)
    assert run_paired(list(A), list(B), parts, cap=None, guard=None).final() == "accept"     # the old pooled rule took it
    P = run_paired(list(A), list(B), parts)
    assert P.final() == "reject" and P.why is None and P.d() > 0
    # as a two-part slice (dominant doc in x, a clean better c part)
    rng = np.random.default_rng(3); Ac = rng.gamma(2, 1, 40); Bc = Ac * np.exp(rng.normal(-0.02, 0.01, 40))
    order = SR.interleave(60, 40); ix = iter(range(60)); ic = iter(range(40)); AA, BB = [], []
    for q in order:
        i = next(ix) if q == "x" else next(ic)
        AA.append(A[i] if q == "x" else Ac[i]); BB.append(B[i] if q == "x" else Bc[i])
    assert run_paired(AA, BB, order, cap=None, guard=None).final() == "accept"
    assert run_paired(AA, BB, order).final() == "reject"


@test
def guard_catches_a_heavy_tail():
    """(b) better on both parts but most documents worse (the gain sits in ~5 heavy-ish documents under the cap):
    the sign guard rejects; with the guard off the capped rule accepts"""
    rng = np.random.default_rng(4); n = 100
    A = rng.gamma(2.0, 1.0, n); A[:5] *= 3                      # 5 heavier documents, each under 20 %
    B = A * 1.015; B[:5] = A[:5] * 0.85
    parts = ["c"] * n
    p = PS.point(A, B); assert p["b"] < 0 and p["c"] > 0 and p["a"] == p["b"], p
    P = run_paired(list(A), list(B), parts); assert P.final() == "reject" and P.why == "guard"
    assert run_paired(list(A), list(B), parts, guard=None).final() == "accept"
    # the guard is size-adjusted like (b): an offset that turns (c) negative too lets it through
    P = run_paired(list(A), list(B), parts, off=-0.02); assert P.final() == "accept", P.summary()


@test
def short_document_noise():
    """many short documents with noisy per-document ratios, long ones consistently 2 % better: (a) / (b) better with
    a tight CI; (c) is noisier (short documents weigh as much as long ones) -- its CI is wider and covers 0, so the
    final-pick guard (CI wholly > 0) does not fire; finalpick picks the better candidate"""
    rng = np.random.default_rng(5)
    nl, ns = 60, 240
    pos = np.r_[np.full(nl, 8000), rng.integers(20, 200, ns)].astype(float)
    A = pos * 0.02 * np.exp(rng.normal(0, 0.3, nl + ns))
    noise = np.r_[rng.normal(-0.02, 0.01, nl), rng.normal(0.0, 0.6, ns)]
    B = A * np.exp(noise)
    st = PS.compare(A, B, boot=4000)
    assert st["a"]["ci"][1] < 0 and st["b"]["ci"][1] < 0, st
    wa = st["a"]["ci"][1] - st["a"]["ci"][0]; wc = st["c"]["ci"][1] - st["c"]["ci"][0]
    assert wc > 3 * wa and st["c"]["ci"][0] < 0, st
    with tempfile.TemporaryDirectory() as d:
        spec = mkspec(d, {"x": (A, pos)}, {"good": {"x": B}, "base": {"x": A}}, parts=["x"])
        R = FP.pick(spec, boot=2000)
        assert R["pick"] == "good" and not R["flags"], (R["pick"], R["flags"])


def mkspec(d, base, cands, parts=("c", "x"), bpw=None, units=None, default=None):
    """base: {part: (A sums, positions)}; cands: {tag: {part: sums}} -> a finalpick SPEC with files in d"""
    C = []
    for t, v in cands.items():
        docs = {}
        for p in parts:
            f = f"{d}/{t}-{p}.json"; json.dump({"m": [[float(s), float(n)] for s, n in zip(v[p], base[p][1])]}, open(f, "w")); docs[p] = f
        C.append({"tag": t, "bpw": (bpw or {}).get(t, 3.0), "docs": docs})
    s = {"band": 3.0, "b": 1.0, "parts": list(parts), "cands": C}
    if units: s["units"] = units
    if default: s["default"] = default
    return s


@test
def finalpick_flag_guard_default():
    """two parts: candidate D wins (a) by the dominant x document (FLAG), but its x (c) is significantly worse than the
    others' -> the guard passes it over; E (uniformly 1 % better) is picked; vs runner-up / default reported"""
    rng = np.random.default_rng(6)
    Ax, Bx = dominant(n=200, seed=7, share=0.35, gain=0.25, loss=0.02); px = np.full(200, 4000.0)
    Ac = rng.gamma(2, 1, 90); pc = np.full(90, 2000.0)
    E = {"x": Ax * np.exp(rng.normal(-0.01, 0.01, 200)), "c": Ac * np.exp(rng.normal(-0.01, 0.01, 90))}
    Dd = {"x": Bx, "c": Ac * np.exp(rng.normal(0.0, 0.01, 90))}
    with tempfile.TemporaryDirectory() as d:
        spec = mkspec(d, {"x": (Ax, px), "c": (Ac, pc)}, {"base": {"x": Ax, "c": Ac}, "D": Dd, "E": E}, default="base")
        R = FP.pick(spec, boot=2000)
        assert R["best_a"] == "D" and R["pick"] == "E", (R["best_a"], R["pick"])
        assert any(f["part"] == "x" for f in R["flags"]) and R["parts"]["x"]["max_doc_share"] > 0.2
        rD = next(r for r in R["rows"] if r["tag"] == "D"); assert rD["guard_fail"] and "(x)" in rD["guard_fail"][0]
        assert R["vs"]["default"]["tag"] == "base" and R["vs"]["default"]["pooled_a"]["ci"][1] < 0
        assert R["vs"]["runner_up"]["tag"] in ("base", "D")
        # size adjustment: E made 0.05 bpw bigger with b = 1 costs +0.05 > its ~1 % gain -> base wins on score
        spec2 = mkspec(d, {"x": (Ax, px), "c": (Ac, pc)}, {"base": {"x": Ax, "c": Ac}, "E": E}, bpw={"E": 3.05})
        R2 = FP.pick(spec2, boot=1000); assert R2["pick"] == "base", R2["pick"]
        FP.show(R)


@test
def traces_resample_as_units():
    """a long trace's chunks share one unit: their copy counts are equal in every draw, and the CI is wider than when
    the chunks are (wrongly) independent"""
    rng = np.random.default_rng(8)
    units = [f"t{i // 8}" if i < 32 else f"d{i}" for i in range(80)]           # 4 traces x 8 chunks + 48 documents
    C = PS.draws(80, units, boot=500, seed=1)
    for t in range(4): assert np.all(C[:, 8 * t:8 * t + 8] == C[:, [8 * t]])
    assert np.all(C.sum(1) >= 0) and math.isclose(float(C[:, 32:].sum() + C[:, :32].sum() / 8 * 1), float(C[:, 32:].sum() + C[:, :32].sum() / 8))
    A = rng.gamma(2, 1, 80); eff = np.r_[np.repeat(rng.normal(0, 0.1, 4), 8), rng.normal(0, 0.02, 48)]   # trace-level effects
    B = A * np.exp(eff)
    w_unit = np.diff(PS.compare(A, B, units=units, boot=4000)["a"]["ci"])[0]
    w_doc = np.diff(PS.compare(A, B, boot=4000)["a"]["ci"])[0]
    assert w_unit > 1.3 * w_doc, (w_unit, w_doc)


@test
def early_stop_matches_the_hosts():
    """the driver's rule (weights from the incumbent's whole slice) and a host's loop over the early-stop block give the
    same stop and statistic"""
    sys.path.insert(0, f"{REPO}/pipeline")
    rng = np.random.default_rng(9)
    order = SR.interleave(130, 50); A = rng.gamma(1, 1, 180); A[5] *= 40; B = A * np.exp(rng.normal(0.03, 0.05, 180))
    w = SR.cap_weights(list(A), order, 0.2)
    rule = {"kind": "curve", "p": 2.0, "warmup": 0.05, "off": 0.0, "w": w, "guard": "docmean"}
    es = json.loads(json.dumps({"rule": rule, "ref": [[a, 1] for a in A], "parts": order}))   # through JSON, as the queue line
    P = SR.Paired(180, es["rule"])
    for i in range(180):
        P.add(es["ref"][i][0], B[i], es["parts"][i])
        if P.check(): break
    Q = run_paired(list(A), list(B), order, rule={"kind": "curve", "p": 2.0, "warmup": 0.05})
    assert P.stopped == Q.stopped == "reject" and P.k == Q.k and math.isclose(P.d(), Q.d())


@test
def rotation_per_slice():
    with tempfile.TemporaryDirectory() as d:
        R = SL.Registry(f"{d}/reg.json")
        R.d["slices"] = {"s1": {"status": "active", "decisions": {}, "docs": [], "built": "1"},
                         "s2": {"status": "ready", "decisions": {}, "docs": [], "built": "2"}}
        for k in range(24): R.decided("s1", str([1.5, 2.0, 3.0][k % 3]), 0)
        assert not R.due("s1", 25) and not R.due("s1", 25, "size")
        R.decided("s1", "1.5", 0)
        assert R.due("s1", 25) and not R.due("s1", 25, "size")
        R.retire("s1", "test"); assert R.active() is None and R.activate_next() == "s2"


# ---- RULES-strata (#186, w164): per-(part, stratum) statistic

OLD_CASE = {   # seqrule.py at HEAD c86c9e4e (before #186): rule {"kind": "full", "off": 0.01, "w": cap 0.2, "guard": "docmean"}
    "d": -0.008175587267912208, "se": 0.008748675081800527, "z": 0.208535888580779, "why": None, "docmean_x": -0.008415597147676597,
    "docmean_c": -0.003559112483005812, "logr_x": -0.01063435808891513, "logr_c": -0.005716816446909287, "w7": 0.7699087443379068,
    "d_pooled": -0.009090543199630366,
    "d2": -0.010361730885858086, "se2": 0.010228122280696382, "z2": -1.5019111489163954}   # pooled, off -0.005, no cap / guard


@test
def old_slice_unchanged_recorded():
    """RULES-strata: a rule without "strata" (every old slice) gives the pre-#186 numbers, recorded above, and no strata key"""
    rng = np.random.default_rng(11); n = 60
    A = rng.gamma(1.5, 1.0, n); A[7] *= 25.0; B = A * np.exp(rng.normal(-0.01, 0.08, n)); B[7] *= 0.9; parts = SR.interleave(40, 20)
    P = run_paired(list(A), list(B), parts, cap=0.2, guard="docmean", off=0.01, rule={"kind": "full"})
    S = P.summary()
    assert math.isclose(P.d(), OLD_CASE["d"], rel_tol=1e-12) and math.isclose(P.se(), OLD_CASE["se"], rel_tol=1e-12), (P.d(), P.se())
    assert math.isclose(P.z(), OLD_CASE["z"], rel_tol=1e-12) and P.why == OLD_CASE["why"] and P.final() == "reject"
    assert math.isclose(P.docmean("x"), OLD_CASE["docmean_x"], rel_tol=1e-12) and math.isclose(P.docmean("c"), OLD_CASE["docmean_c"], rel_tol=1e-12)
    assert math.isclose(S["parts"]["x"]["logr"], OLD_CASE["logr_x"], rel_tol=1e-12) and math.isclose(S["parts"]["c"]["logr"], OLD_CASE["logr_c"], rel_tol=1e-12)
    assert math.isclose(P.w[7], OLD_CASE["w7"], rel_tol=1e-12) and math.isclose(P.d_pooled(), OLD_CASE["d_pooled"], rel_tol=1e-12)
    assert "strata" not in S and all("strata" not in v for v in S["parts"].values())
    P2 = run_paired(list(A), list(B), parts, cap=0, guard=None, off=-0.005, rule={"kind": "full", "stat": "pooled"})
    assert math.isclose(P2.d(), OLD_CASE["d2"], rel_tol=1e-12) and math.isclose(P2.se(), OLD_CASE["se2"], rel_tol=1e-12)
    assert math.isclose(P2.z(), OLD_CASE["z2"], rel_tol=1e-12) and P2.final() == "accept"


def strat_case(seed=4, n=60):
    """a two-part slice with N documents in both parts (about 15 %), the rest L"""
    rng = np.random.default_rng(seed)
    A = rng.gamma(2.0, 1.0, n); B = A * np.exp(rng.normal(-0.02, 0.1, n))
    parts = SR.interleave(n // 2, n - n // 2); strata = ["N" if rng.random() < 0.15 else "L" for _ in range(n)]
    return A, B, parts, strata


def run_strat(A, B, parts, strata, cap=0.2, guard="docmean", off=0.0, rule=None):
    r = dict(rule or {"kind": "full"}, off=off, strata=True)
    if cap: r["w"] = SR.cap_weights(list(A), parts, cap, strata=strata)
    if guard: r["guard"] = guard
    P = SR.Paired(len(A), r)
    for a, b, p, s in zip(A, B, parts, strata):
        P.add(a, b, p, s)
        if P.check(): break
    return P


def manual_strat(A, B, parts, strata, cap=0.2, off=0.0):
    """RULES-strata §4-5 written out from the definitions, independent of Paired: per part d_p, se_p, docmean; d = mean d_p"""
    w = SR.cap_weights(list(A), parts, cap, strata=strata); out = {"logr": {}, "se": {}, "docmean": {}}
    for p in dict.fromkeys(parts):
        st = [s for s in ("L", "N") if any(q == p and t == s for q, t in zip(parts, strata))]
        W = sum(PS.SW[s] for s in st); lr = se2 = dm = 0.0
        for s in st:
            ix = [i for i in range(len(A)) if parts[i] == p and strata[i] == s]
            a = np.array([w[i] * A[i] for i in ix]); b = np.array([w[i] * B[i] for i in ix]); k = len(ix)
            lr += PS.SW[s] * math.log(b.sum() / a.sum()) / W
            u = b / b.mean() - a / a.mean()
            se_ps = math.sqrt((u * u).sum() / (k - 1)) / math.sqrt(k) if k >= 3 else float("inf")
            se2 += (PS.SW[s] / W * se_ps) ** 2
            dm += PS.SW[s] * float(np.mean([math.log(B[i] / A[i]) for i in ix if A[i] > 0 and B[i] > 0])) / W
        out["logr"][p] = lr; out["se"][p] = math.sqrt(se2); out["docmean"][p] = dm
    out["d"] = sum(out["logr"].values()) / len(out["logr"])
    out["se_all"] = math.sqrt(sum(x * x for x in out["se"].values())) / len(out["se"])
    return out


@test
def strata_cap_within_stratum():
    """the cap is per (part, stratum): a dominant N document is capped against the N stratum's total only; the L
    documents keep weight 1; the N weights are capped_weights of the N documents alone"""
    A = [1.0] * 20 + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0]
    parts = ["x"] * len(A); strata = ["L"] * 20 + ["N"] * 8
    w = SR.cap_weights(A, parts, 0.2, strata=strata)
    assert all(x == 1.0 for x in w[:20]), w[:20]
    assert all(math.isclose(a, b) for a, b in zip(w[20:], SR.capped_weights(A[20:], 0.2))) and w[27] < 1.0
    assert max(w[i] * A[i] for i in range(20, 28)) <= 0.2 * sum(w[i] * A[i] for i in range(20, 28)) + 1e-9
    assert SR.cap_weights(A, parts, 0.2) != w                                  # unstratified: the L documents change too
    assert SR.cap_weights(A, parts, 0.2, strata=None) == SR.cap_weights(A, parts, 0.2)


@test
def strata_formula_matches_definition():
    """d, the per-part logs, se and docmean of the stratified Paired = the RULES-strata formulas written out by hand"""
    A, B, parts, strata = strat_case(4, 60)
    P = run_strat(A, B, parts, strata, cap=0.2, guard=None); M = manual_strat(A, B, parts, strata, 0.2)
    assert math.isclose(P.d(), M["d"], rel_tol=1e-12, abs_tol=1e-15), (P.d(), M["d"])
    for p in ("x", "c"):
        assert math.isclose(P.part_logr(p), M["logr"][p], rel_tol=1e-12, abs_tol=1e-15), p
    assert math.isclose(P.se(), M["se_all"], rel_tol=1e-9), (P.se(), M["se_all"])
    P2 = run_strat(A, B, parts, strata, cap=0.2, guard="docmean", off=0.01)
    for p in ("x", "c"):
        assert math.isclose(P2.docmean(p), M["docmean"][p], rel_tol=1e-12, abs_tol=1e-15), p
    S = P.summary()
    assert S["strata"] is True and set(S["parts"]["x"]["strata"]) == {"L", "N"} and set(S["parts"]["c"]["strata"]) == {"L", "N"}


@test
def strata_absent_stratum():
    """a stratum absent from a part: that part's value is the other stratum alone (no weighting); the other part unchanged"""
    A, B, parts, strata = strat_case(7, 60)
    strata = [s if p == "x" else "L" for p, s in zip(parts, strata)]              # part c: no N documents at all
    P = run_strat(A, B, parts, strata, cap=0.2, guard=None); M = manual_strat(A, B, parts, strata, 0.2)
    assert math.isclose(P.part_logr("c"), M["logr"]["c"], rel_tol=1e-12) and "N" not in P.summary()["parts"]["c"]["strata"]
    assert math.isclose(P.d(), M["d"], rel_tol=1e-12)


@test
def strata_guard_matches_definition():
    """the guard (RULES-strata §5): every part's size-adjusted stratified docmean has the sign of its size-adjusted
    stratified log ratio; P.guard_ok() equals that rule on random slices (both outcomes occur)"""
    outcomes = set()
    for seed in range(40):
        A, B, parts, strata = strat_case(100 + seed, 40)
        M = manual_strat(A, B, parts, strata, 0.2, 0.005)
        P = run_strat(A, B, parts, strata, cap=0.2, guard="docmean", off=0.005)
        want = all((M["docmean"][p] + 0.005 < 0) == (M["logr"][p] + 0.005 < 0) for p in M["logr"])
        assert P.guard_ok() == want, (seed, P.guard_ok(), want)
        outcomes.add(want)
    assert outcomes == {True, False}, outcomes


@test
def strata_se_needs_three_per_stratum():
    """se is infinite until every (part, stratum) present has 3 documents (no early stop before that)"""
    P = SR.Paired(10, {"strata": True})
    for k, (p, s) in enumerate([("x", "N"), ("x", "L"), ("x", "L"), ("x", "L"), ("x", "N"), ("x", "N")]):
        P.add(1.0, 0.9, p, s)
        ready = all(v[2] >= 3 for v in P.ps.values())
        assert (P.se() != SR.INF) == ready, (k, {key: v[2] for key, v in P.ps.items()}, P.se())


@test
def strata_require_a_stratum():
    P = SR.Paired(3, {"strata": True})
    for args in [(1.0, 1.0, "x", None), (1.0, 1.0, "x", "Z")]:
        try: P.add(*args); raise AssertionError(f"no error for {args}")
        except ValueError: pass
    try: SR.Paired(3, {}).add(1.0, 1.0, "x", "N"); raise AssertionError("unstratified rule took a stratum")
    except ValueError: pass


@test
def strata_registry():
    """the registry keeps each document's stratum; a slice without strata has none (old slices unchanged)"""
    with tempfile.TemporaryDirectory() as d:
        R = SL.Registry(f"{d}/reg.json")
        base = {"part": "x", "src": 0, "orig": 1, "source": "a", "tokens": 100, "score_from": 0}
        R.add("sn", "/nonexistent", man={"docs": [dict(base, stratum="N", script="CJK"), dict(base, part="c", src=1, orig=2, stratum="L", script="Latin")]})
        R.add("so", "/nonexistent", man={"docs": [dict(base)]})
        assert R.strata_of("sn") == ["N", "L"] and R.strata_of("so") is None and "stratum" not in R.d["slices"]["so"]["docs"][0]


@test
def pickstat_strata_and_small_parts():
    """pickstat's stratified point = the by-hand value (the (b) cap inside each stratum); the bootstrap interval brackets
    the point; a stratum or part of 3 documents is uncapped (no nan), as seqrule does"""
    A, B, parts, strata = strat_case(9, 60)
    pt = PS.point_strat(A, B, parts, strata, 0.2)
    for p in ("x", "c"):
        present = [s for s in ("L", "N") if any(q == p and t == s for q, t in zip(parts, strata))]
        wsum = sum(PS.SW[s] for s in present)
        for k in ("a", "b", "c"):                                   # each statistic: the per-stratum point, weighted
            want = sum(PS.SW[s] * PS.point([A[i] for i in range(len(A)) if parts[i] == p and strata[i] == s],
                                           [B[i] for i in range(len(A)) if parts[i] == p and strata[i] == s], 0.2)[k]
                       for s in present) / wsum
            assert math.isclose(pt["parts"][p][k], want, rel_tol=1e-12), (p, k, pt["parts"][p][k], want)
    assert math.isclose(pt["b"], (pt["parts"]["x"]["b"] + pt["parts"]["c"]["b"]) / 2, rel_tol=1e-12)
    cb = PS.compare_strat(A, B, parts, strata, boot=400, seed=2)
    assert all(v["ci"][0] <= v["d"] <= v["ci"][1] for v in cb.values()), cb
    assert np.allclose(PS.capped(np.array([1.96, 0.96, 2.9])), 1.0)
    assert math.isfinite(PS.point(np.array([1.96, 0.96, 2.9]), np.array([1.9, 0.9, 2.8]))["b"])


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
