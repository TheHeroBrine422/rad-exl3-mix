#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""testbranch231.py -- w231 branch training (CPU, testhf225's tiny real Qwen3.5 stack). Two "sessions" of real history (~1k tokens), three cuts each
(cut = the history up to the cut + a generated turn, scored only on the turn, a golden doc of its own). Trained two ways in one optimizer step on two streams:
  separate: each cut as its own full-prefix document;  branch: the history walked once (no grad), each cut trained from its branch point (the cut's
  start rounded down to Q, so its segments are the separate document's) on a copy of the history's state, the history continuing from the snapshot.
Checks: KL sums / rows equal and the reduced gradient within 1e-6 (bit for bit expected) for one card, the two-card naive split, the pipeline (host
transfers) and 2 data-parallel ranks; a lost snapshot (a resume) = the existing resume path (prefix rebuilt from the item's own tokens) bit for bit;
the history really runs without grad (no graph: fewer saved tensors). Run like testhf231. Exit 0 = all pass."""
import json, os, sys, tempfile, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad225"))
import testhf225 as H                                          # noqa: E402
D, Q_ = H.D, H.Q
QT = 128                                                       # the stream quota (H.targs Q)


def build_gold(d, seed=7):
    """2 sessions x 3 cuts: docs [s0c0, s0c1, s0c2, s1c0, ...]; cut c of session s = hist_s[:P] + gen; score_from = P"""
    rg = np.random.default_rng(seed); os.makedirs(d, exist_ok=True)
    docs, toks, cuts = [], [], []
    for s_ in range(2):
        hist = rg.integers(0, H.V_, 1000)
        for P in ((250, 520, 800) if s_ == 0 else (180, 450, 700)):
            g = rg.integers(0, H.V_, int(rg.integers(60, 150)))
            t = np.concatenate([hist[:P], g]); docs.append((len(t), P)); toks.append(t); cuts.append((s_, P))
    R = sum(n - 1 - P for n, P in docs); K = 4
    top = np.empty((R, K), D.rec(K)); rows = np.zeros((R, 3), np.float32)
    for r in range(R):
        pr = rg.dirichlet(np.ones(H.V_) * 0.3); ix = np.argsort(-pr)[:K]; top["id"][r] = ix; top["lp"][r] = np.log(pr[ix]); rows[r, 2] = 1 - pr[ix].sum()
    top.tofile(f"{d}/top.bin"); rows.tofile(f"{d}/rows.f32"); np.concatenate(toks).astype(np.int32).tofile(f"{d}/tokens.i32")
    r0, js = 0, []
    for n, P in docs: js.append({"tokens": int(n), "score_from": int(P), "row0": r0}); r0 += n - 1 - P
    json.dump({"docs": js, "top": K, "rows": R}, open(f"{d}/kld.json", "w")); np.ones(R, np.uint8).tofile(f"{d}/gen.u8")
    return D.Gold(d, "sess"), cuts


def segs(it, a, b):
    out, x = [], a
    while x < b: y = min(b, (x // QT + 1) * QT); out.append((it, x, y)); x = y
    return out


def arrangement(gold, cuts, branch):
    """C and one step's segments per stream (stream = session)"""
    items = [(0, i, gold.ntok(i), gold.nrows(i, 1 << 30)) for i in range(len(cuts))]
    C = types.SimpleNamespace(g=[gold], tags=["sess"], items=list(items))
    st = [[], []]
    if not branch:
        for i, (s_, P) in enumerate(cuts): st[s_] += segs(i, 0, gold.ntok(i))
        return C, [st]
    C.branch, C.starts = {}, {}
    for s_ in range(2):
        mine = [i for i, (q, _) in enumerate(cuts) if q == s_]; hist = mine[-1]; prev = 0     # the history = the last cut's prefix
        for k, i in enumerate(mine):
            a0 = cuts[i][1] // QT * QT
            assert np.array_equal(gold.ids(i, 0, a0), gold.ids(hist, 0, a0))
            C.items.append((0, hist, a0, 0)); h = len(C.items) - 1
            C.branch[h] = ("hist", prev, 0, hist)
            if prev: C.starts[h] = prev
            st[s_] += segs(h, prev, a0)
            C.branch[i] = ("cut_last" if k == len(mine) - 1 else "cut", a0, 0, hist); C.starts[i] = a0
            st[s_] += segs(i, a0, gold.ntok(i)); prev = a0
    return C, [st]


def grads(T): return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double() for p in T.P])


def train(C, steps, split=0, pipeline=False, lose_snaps=False):
    stu = H.Tiny(0, split=split); a = H.targs(pipeline=pipeline, pipe_xfer="host", V=2, Q=QT)
    stu.kvbuf = bool(split); stu.pos_off = 0; stu.kv_cap = 0; stu.mask = None; Q_.set_first_train(stu, lambda s: None)
    rows_of, cum = Q_.rows_cum(C, steps)
    T = Q_.Trainer(a, stu, C, steps, rows_of, Q_.Dist(), lambda s: None)
    if lose_snaps:                                             # a resume between the history and its cut: no snapshot, the prefix is rebuilt
        T.branch_in = lambda sid, it, sa: None
    accs, w = T.run_group([0], a.lr)
    return T, (float(accs[0]), float(accs[1]), float(accs[-1])), grads(T)


def rank_main(r, port, out, gdir, cuts):
    os.environ.update(RANK=str(r), WORLD_SIZE="2", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    gold = D.Gold(gdir, "sess"); z = {}
    for br in (False, True):
        C, st = arrangement(gold, cuts, br); T, o, g = train(C, st, split=4, pipeline=True); z[br] = (o, g)
    torch.save(z, f"{out}/r{r}.pt"); T.d.dist.destroy_process_group()


def main():
    gdir = os.path.join(tempfile.mkdtemp(), "sess"); gold, cuts = build_gold(gdir)
    res = {}
    for nm, kw in (("one card", {}), ("naive split", {"split": 4}), ("pipeline", {"split": 4, "pipeline": True})):
        Cs, ss = arrangement(gold, cuts, False); Cb, sb = arrangement(gold, cuts, True)
        Ts, os_, gs = train(Cs, ss, **kw); Tb, ob, gb = train(Cb, sb, **kw)
        gr = float((gb - gs).norm() / gs.norm())
        H.check(f"branch = separate ({nm}): KL sum / rows equal, reduced gradient within 1e-6 (bit for bit: {torch.equal(gb, gs)}); no snapshot left",
                abs(ob[0] - os_[0]) <= 1e-9 * abs(os_[0]) and ob[1] == os_[1] and gr < 1e-6 and not Tb.snaps,
                f"KL {ob[0]!r} vs {os_[0]!r}, rows {ob[1]} vs {os_[1]}, grad rel {gr:.2e}, prefix-rebuild tokens {ob[2]:.0f} vs {os_[2]:.0f}")
        res[nm] = (ob, gb)
    Cb, sb = arrangement(gold, cuts, True); Tl, ol, gl = train(Cb, sb, lose_snaps=True)
    Cr, sr = arrangement(gold, cuts, False)                    # the old resume path: each cut from its branch point, the prefix rebuilt by prefix_state
    sr = [[[x for x in s_ if x[1] >= cuts[x[0]][1] // QT * QT] for s_ in sr[0]]]
    Tr, orr, grr = train(Cr, sr)
    H.check("lost snapshots (a resume): = the existing resume path (each cut's prefix rebuilt from its own tokens) bit for bit",
            ol[:2] == orr[:2] and torch.equal(gl, grr), f"KL {ol[0]!r} vs {orr[0]!r}; rebuilt {ol[2]:.0f} vs {orr[2]:.0f} tokens (vs the walked branch: KL rel {abs(ol[0] / res['one card'][0][0] - 1):.1e}, grad rel {float((gl - res['one card'][1]).norm() / res['one card'][1].norm()):.1e}: the rebuild's {H.targs().len}-token segments round differently on this tiny bf16 model)")
    saved = []                                                  # the history runs with grad off: the layers save nothing for backward
    def hook(t): saved.append(1); return t
    with torch.autograd.graph.saved_tensors_hooks(hook, lambda t: t):
        Cb, sb = arrangement(gold, cuts, True); hist_only = [[[x for x in s_ if Cb.branch.get(x[0], ("",))[0] == "hist"] for s_ in sb[0]]]
        train(Cb, hist_only)
    H.check("history pieces alone: no tensor saved for backward (no grad)", not saved, f"{len(saved)} saved")
    out = tempfile.mkdtemp(); import torch.multiprocessing as mp
    mp.start_processes(rank_main, args=(29000 + os.getpid() % 1000, out, gdir, cuts), nprocs=2, join=True, start_method="spawn")
    z = torch.load(f"{out}/r0.pt", weights_only=False)
    gr = float((z[True][1] - z[False][1]).norm() / z[False][1].norm()); g1 = float((z[True][1] - res["pipeline"][1]).norm() / res["pipeline"][1].norm())
    H.check("2 data-parallel ranks (pipeline each): branch = separate (gradient within 1e-6), and = the one-rank branch run within 1e-6",
            z[True][0][:2] == z[False][0][:2] and gr < 1e-6 and g1 < 1e-6, f"grad rel {gr:.2e} (vs one rank {g1:.2e}); KL {z[True][0][0]!r} vs {z[False][0][0]!r}")
    print("\n" + ("ALL PASS" if not H.fails else f"FAILED: {H.fails}"))
    sys.exit(1 if H.fails else 0)


if __name__ == "__main__": main()
