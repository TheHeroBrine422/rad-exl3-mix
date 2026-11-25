#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""plan231.py SPEC.json OUT.json [--steps-tsv FILE] -- w231: a mixed-length QAD plan for the finals (qad220 --plan231 OUT.json).

One plan mixes sources of very different lengths (8k calib docs, teacher cuts = prefix + generation of 8k-160k scored only on the generation,
xgap long docs up to 128k, deep traces of ~250k) over qad220's V streams x Q tokens a step. qad220 trains stream s's queue in order, Q tokens a
step (data181.plan), so the plan is the per-stream queues. This builder fills them step by step (a simulation of data181.plan that picks a stream's
next document when the stream needs one; the trainer re-runs data181.plan on the queues and gets the same steps, checked at the end):
  - single epoch: every document at most once; the plan stops taking documents once the assigned gen rows reach budget_rows x (1 + margin)
  - source shares (`share`, by gen rows; the lead sets them): the next document comes from the source furthest below its share of the rows so far
    (an exhausted source's share goes to the others; the step it ran out is reported)
  - deep traces (every document of a `deep: true` source, or its `ids`): their `starts` (step numbers) or spread evenly over [deep_first, end - longest trace]: each trace starts at
    its step on the first stream that needs a document, on the rank with the fewest deep streams, so the depth bands of the traces fall in every lr
    phase (reported per band and phase); a trace waits while the carried budget is full
  - bounded steps: every rank's carried state (the sum of its streams' depths, 64 KB a token; predicted at each admission: long documents grow Q a
    step until they end, short ones count whole) stays <= max_carried_tokens; at most max_long documents longer than long_tokens per rank at once. Per-step cost (forward +
    backward of scored segments, attention ~ 1 + depth / 100k) is reported per rank (max / mean): the data-parallel balance and the step's bound.
SPEC (JSON):
  {"V": 24, "Q": 1024, "ranks": 2, "seed": 0, "budget_rows": 1e7, "margin": 0.02, "max_carried_tokens": 400000, "long_tokens": 32768, "max_long": 4,
   "phases_rows": [1.5e6, 9e6],                       # lr phase boundaries (report only)
   "exclude_calib": ["ids.json", ...],                # calib doc ids never to train (judges: j217, jc228, golden-jlong, a run's fixed-eval-docs.json, ...)
   "fixed_eval": {"pools": "pools214.json", "pool": "train214", "seed": 2181, "rows": 20000, "len": 2048},   # qad220's fixed-eval docs (excluded too)
   "sources": {
     "calib":  {"golden": "calib", "dir": "/data/q27/challenge/golden/calib", "pools": "pools-v3.json", "pool": ["qad", "fit", "search"], "cap": 8192, "share": 0.4},
     "tset1":  {"golden": "tset1", "dir": "QROOT/ext/gold/tset1", "genmask": "QROOT/ext/genmask/tset1.u8", "share": 0.3},
     "xgap":   {"golden": "xgl1", "dir": ..., "cap": 131072, "share": 0.3, "ids": "optional ids.json"},
     "traces": {"golden": "trdeep-b1", "dir": ..., "genmask": ..., "deep": true, "ids": "8 trace ids.json"}}, "deep_first": 0}
  dir / genmask: the golden's directory and gen mask (default DIR/gen.u8 if it exists, else every row is gen); cap: tokens a document is cut at
  (rows recounted); min_rows (16): documents with fewer gen rows are skipped. The plan stores each golden's name with dir / genmask: qad220 loads
  them (the calib golden named "calib" is the trainer's own --calib).
OUT.json: {"format": "plan231", V, Q, ranks, goldens {name: {dir, genmask}}, items [[golden, doc, tokens, rows]], queues [[item, ..] per stream],
  budget_rows, spec, summary}. It lists per-document ids (the user's traces may be among them): keep it on our machines."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D                                      # noqa: E402

KV_BYTES = 65536                                         # 16 attention layers x 4 KV heads x 256 x bf16 K + V, a token


class Src:
    def __init__(self, name, sp, excl):
        self.name, self.sp = name, sp
        self.golden = sp.get("golden", name); d = sp["dir"]
        gm = sp.get("genmask") or (os.path.join(d, "gen.u8") if os.path.exists(os.path.join(d, "gen.u8")) else None)
        self.dir, self.genmask = d, gm
        self.g = D.Gold(d, self.golden, genfile=gm)
        self.cap = int(sp.get("cap", 1 << 30)); self.deep = bool(sp.get("deep", False)); self.share = float(sp.get("share", 0.0))
        ids = range(len(self.g.docs))
        if "ids" in sp: ids = json.load(open(sp["ids"]))
        if "pools" in sp:
            pl = json.load(open(sp["pools"]))["pool"]; want = set(sp["pool"] if isinstance(sp["pool"], list) else [sp["pool"]])
            ids = [i for i in ids if pl[i] in want]
        ids = [int(i) for i in ids if not (self.golden == "calib" and int(i) in excl)]
        self.items = []
        for i in ids:
            n = min(self.cap, self.g.ntok(i)); nr = self.g.nrows(i, n)
            if nr >= int(sp.get("min_rows", 16)): self.items.append((i, n, nr))


def fixed_eval_docs(fe, calib):
    """qad220 build_data's fixed-eval selection (the docs it holds out of training)"""
    pl = json.load(open(fe["pools"]))["pool"]
    cands = [i for i, p in enumerate(pl) if p == fe.get("pool", "train214")]
    rng = np.random.default_rng(fe.get("seed", 2181)); rng.shuffle(cands); out, tot = [], 0
    for i in cands:
        nr = len(calib.scored(i, 0, min(fe.get("len", 2048), calib.ntok(i)))[0])
        if nr >= 200: out.append(i); tot += nr
        if tot >= fe.get("rows", 20000): break
    return out


def seg_cost(a, b, m):
    """relative cost of a segment of tokens a..b with m scored rows: forward 1, backward + recompute 2 more when scored; attention ~ 1 + depth / 100k"""
    return (b - a) * (1 + (a + b) / 2 / 100000) * (3 if m else 1) / 3


def build(spec):
    V, Q, R = int(spec["V"]), int(spec["Q"]), int(spec.get("ranks", 1))
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    excl = set()
    for f in spec.get("exclude_calib", []): excl |= {int(x) for x in json.load(open(f))}
    srcs = {}
    for nm, sp in spec["sources"].items():
        if sp.get("golden", nm) == "calib" and "fixed_eval" in spec:
            excl |= set(fixed_eval_docs(spec["fixed_eval"], D.Gold(sp["dir"], "calib")))
        srcs[nm] = Src(nm, sp, excl)
    for s in srcs.values(): rng.shuffle(s.items)
    deep = [s for s in srcs.values() if s.deep]; mix = [s for s in srcs.values() if not s.deep]
    tot_share = sum(s.share for s in mix); assert tot_share > 0, "no share > 0 among the non-deep sources"
    budget = float(spec["budget_rows"]); stop_rows = budget * (1 + float(spec.get("margin", 0.02)))
    maxc = int(spec.get("max_carried_tokens", 400000)); LT = int(spec.get("long_tokens", 32768)); ML = int(spec.get("max_long", 4))
    seen = set()
    for s in srcs.values():
        for (i, n, nr) in s.items:
            assert (s.golden, i) not in seen, f"{s.golden}:{i} in two sources"; seen.add((s.golden, i))
    have = sum(nr for s in srcs.values() for (_, _, nr) in s.items)
    assert have >= stop_rows, f"the sources hold {have} gen rows < budget_rows x (1 + margin) = {stop_rows:.0f} (single epoch): lower the budget or add data"
    # deep trace starts: two passes (the end step is known after the first)
    traces = [(s, it) for s in deep for it in s.items]
    end_est = None; plan = None
    for _ in range(3):
        plan = simulate(spec, V, Q, R, srcs, mix, deep, traces, budget, stop_rows, maxc, LT, ML, end_est)
        e2 = plan["last"]
        if end_est is not None and abs(e2 - end_est) <= 2: break
        end_est = e2
    return plan


def trace_starts(spec, traces, end_est, Q):
    if not traces: return []
    longest = max(n for _, (_, n, _) in traces)
    hi = max(0, (end_est or 10 ** 9) - -(-longest // Q) - 2); lo = min(hi, int(spec.get("deep_first", 0)))
    given = None
    for s, _ in traces:
        if "starts" in s.sp: given = s.sp["starts"]
    if given is not None: return sorted(int(x) for x in given)[:len(traces)] + [10 ** 9] * max(0, len(traces) - len(given))
    k = len(traces); return [int(round(lo + (hi - lo) * j / max(1, k - 1))) for j in range(k)] if end_est is not None else [lo] * k


def simulate(spec, V, Q, R, srcs, mix, deep, traces, budget, stop_rows, maxc, LT, ML, end_est, min_tail=128):
    starts = trace_starts(spec, traces, end_est, Q)
    pend = list(zip(starts, traces))                       # (start step, (source, (doc, n, rows))) in start order
    nxt = {s.name: 0 for s in mix}                         # next unused item per source (single epoch)
    items, queues = [], [[] for _ in range(V)]
    cur_item = [None] * V; pos = [0] * V; cur = [0] * V; done = [False] * V
    rows_by = {s.name: 0 for s in srcs.values()}; out_at = {}; assigned = 0
    trace_log = []                                         # (item, stream, start step)
    steps = []
    rank = lambda s: s % R

    def active(r):
        return [queues[s][pos[s]] for s in range(V) if rank(s) == r and pos[s] < len(queues[s])]

    def fits(r, n):
        """the rank's carried state stays <= maxc from now until the new document ends, predicted: a long document's state grows Q a step
        from its current depth and is gone when it ends (the stream's next document is checked when it is taken), a short one counts whole"""
        act = [(items[queues[s][pos[s]]][2], cur[s]) for s in range(V) if rank(s) == r and pos[s] < len(queues[s])]
        if n > LT and sum(1 for m, _ in act if m > LT) >= ML: return False
        k = np.arange(-(-n // Q) + 1) * Q
        tot = np.minimum(k, n) if n > LT else np.full(len(k), n)
        for m, d in act:
            if m <= LT: tot = tot + m
            else: dk = d + k; tot = tot + np.where(dk < m, dk, 0)
        return int(tot.max()) <= maxc

    def take(s, t):
        nonlocal assigned
        r = rank(s)
        if assigned >= stop_rows and not pend: return None
        if pend and pend[0][0] <= t:                       # a trace is due: the rank with the fewest deep streams takes it
            dcnt = [sum(1 for it in active(q) if items[it][4]) for q in range(R)]
            src, (i, n, nr) = pend[0][1]
            if dcnt[r] == min(dcnt) and fits(r, n):
                pend.pop(0); items.append((src.golden, i, n, nr, True, src.name)); rows_by[src.name] += nr; assigned += nr
                trace_log.append((len(items) - 1, s, t)); return len(items) - 1
        if assigned >= stop_rows: return None if not pend else take_short(s, r)
        tot = sum(rows_by[m.name] for m in mix) or 1
        live = [m for m in mix if nxt[m.name] < len(m.items)]
        for m in mix:
            if nxt[m.name] >= len(m.items) and m.name not in out_at: out_at[m.name] = t
        if not live: return None if not pend else take_short(s, r)
        sh = sum(m.share for m in live)
        order = sorted(live, key=lambda m: -(m.share / sh * tot - rows_by[m.name]))
        for m in order:
            L = m.items
            for k in range(nxt[m.name], min(len(L), nxt[m.name] + 64)):
                i, n, nr = L[k]
                if fits(r, n):
                    L[nxt[m.name]], L[k] = L[k], L[nxt[m.name]]; nxt[m.name] += 1
                    items.append((m.golden, i, n, nr, False, m.name)); rows_by[m.name] += nr; assigned += nr; return len(items) - 1
        return take_short(s, r, force=True)

    def take_short(s, r, force=False):
        """a stream that must not idle (a trace is still pending, or nothing fits): the shortest unused document of any mixed source"""
        nonlocal assigned
        best = None
        for m in mix:
            for k in range(nxt[m.name], len(m.items)):
                if best is None or m.items[k][1] < best[2][1]: best = (m, k, m.items[k])
        if best is None: return None
        m, k, (i, n, nr) = best; L = m.items
        L[nxt[m.name]], L[k] = L[k], L[nxt[m.name]]; nxt[m.name] += 1
        items.append((m.golden, i, n, nr, False, m.name)); rows_by[m.name] += nr; assigned += nr
        if force: over.append((len(steps), s))
        return len(items) - 1

    over = []
    t = 0
    while not all(done):
        st = []
        for s in range(V):
            segs, q = [], Q
            while q > 0 and not done[s]:
                if pos[s] == len(queues[s]):
                    it = take(s, t)
                    if it is None: done[s] = True; break
                    queues[s].append(it)
                it = queues[s][pos[s]]; n = items[it][2]; a = cur[s]
                b = min(n, a + q)
                if n - b < min_tail: b = n
                segs.append((it, a, b)); q -= b - a
                if b >= n: pos[s] += 1; cur[s] = 0
                else: cur[s] = b
                if q < min_tail: break
            st.append(segs)
        if any(st): steps.append(st)
        t += 1
        assert t < 10 ** 6, "plan does not end"
    return dict(items=items, queues=queues, steps=steps, rows_by=rows_by, out_at=out_at, trace_log=trace_log, starts=starts, over=over, srcs=srcs,
                last=last_step(items, steps, budget, srcs))


def seg_rows(srcs_by_golden, items, it, a, b):
    g = srcs_by_golden[items[it][0]].g; return len(g.scored(items[it][1], a, b)[0])


def last_step(items, steps, budget, srcs):
    byg = {s.golden: s for s in srcs.values()}; tot = 0
    for k, st in enumerate(steps):
        for segs in st:
            for (it, a, b) in segs: tot += seg_rows(byg, items, it, a, b)
        if tot >= budget: return k + 1
    return len(steps)


def report(spec, plan, tsv=None):
    V, Q, R = int(spec["V"]), int(spec["Q"]), int(spec.get("ranks", 1))
    items, steps, srcs = plan["items"], plan["steps"], plan["srcs"]
    byg = {s.golden: s for s in srcs.values()}
    budget = float(spec["budget_rows"]); last = plan["last"]
    phases = [float(x) for x in spec.get("phases_rows", [])]
    cum = 0; rows_src = {}; ph_src = {}; band = {}; cost = []; carried = []; deepn = []
    for k, st in enumerate(steps):
        c = [0.0] * R; car = [0] * R; dn = 0
        for s, segs in enumerate(st):
            for (it, a, b) in segs:
                m = seg_rows(byg, items, it, a, b); src = items[it][5]
                c[s % R] += seg_cost(a, b, m)
                if k < last:
                    rows_src[src] = rows_src.get(src, 0) + m
                    p = int(np.searchsorted(phases, cum, side="right")); ph_src.setdefault(src, [0] * (len(phases) + 1))[p] += m
                    if items[it][4]: band.setdefault(min(a // 65536, 3), [0] * (len(phases) + 1))[p] += m
                cum += m
            if segs:
                it, a, b = segs[-1]; car[s % R] += b if b < items[it][2] else 0
                dn += any(items[it_][4] for it_, _, _ in segs)
        cost.append(c); carried.append(car); deepn.append(dn)
    cs = np.array(cost[:last]); cr = np.array(carried[:last])
    tok_src, cost_src = {}, {}
    for k, st in enumerate(steps[:last]):
        for segs in st:
            for (it, a, b) in segs:
                src = items[it][5]; tok_src[src] = tok_src.get(src, 0) + b - a; cost_src[src] = cost_src.get(src, 0.0) + seg_cost(a, b, seg_rows(byg, items, it, a, b))
    pool = {}
    for s in srcs.values():                                 # the whole source (all its usable documents): rows a token, and the cost of training all of it
        t_ = sum(n for _, n, _ in s.items); r_ = sum(nr for _, _, nr in s.items)
        c_ = sum(seg_cost(a_, min(n, a_ + Q), len(s.g.scored(i, a_, min(n, a_ + Q))[0])) for i, n, _ in s.items for a_ in range(0, n, Q))
        pool[s.name] = {"docs": len(s.items), "tokens": t_, "rows": r_, "rows_per_token": round(r_ / max(1, t_), 3), "cost_all": round(c_)}
    tot = sum(rows_src.values())
    out = {"steps": len(steps), "last": last, "rows_to_last": int(sum(rows_src.values())),
           "share_rows": {k: round(v / tot, 4) for k, v in sorted(rows_src.items())},
           "share_tokens": {k: round(v / max(1, sum(tok_src.values())), 4) for k, v in sorted(tok_src.items())},
           "share_cost": {k: round(v / max(1e-9, sum(cost_src.values())), 4) for k, v in sorted(cost_src.items())},
           "run_cost_to_last": round(float(cs.sum())), "sources": pool,
           "rows_by_phase": ph_src, "trace_rows_by_depth_band_64k_and_phase": {f"{b * 64}k+": v for b, v in sorted(band.items())},
           "trace_starts": [(items[it][0], items[it][1], items[it][2], s, t) for it, s, t in plan["trace_log"]],
           "trace_ends_by_last": all(trace_end(steps, it) < last for it, _, _ in plan["trace_log"]) and len(plan["trace_log"]) == len(plan["starts"]),
           "planned_starts": plan["starts"], "source_exhausted_at_step": plan["out_at"],
           "step_cost_per_rank": {"mean": round(float(cs.mean()), 1), "max": round(float(cs.max()), 1), "p95": round(float(np.percentile(cs.max(1), 95)), 1),
                                  "rank_imbalance_mean": round(float((cs.max(1) / np.maximum(cs.mean(1), 1e-9)).mean()), 3)},
           "carried_gib_per_rank": {"max": round(float(cr.max()) * KV_BYTES / 2**30, 1), "p95": round(float(np.percentile(cr.max(1), 95)) * KV_BYTES / 2**30, 1)},
           "max_deep_streams": int(max(deepn[:last] or [0])), "forced_over_bound": len(plan["over"]),
           "unused_docs": {s.name: len(s.items) - sum(1 for x in items if x[5] == s.name) for s in srcs.values()}}
    if tsv:
        with open(tsv, "w") as f:
            f.write("step\t" + "\t".join(f"cost_r{r}" for r in range(R)) + "\t" + "\t".join(f"carried_tok_r{r}" for r in range(R)) + "\tdeep_streams\n")
            for k in range(len(steps)): f.write(f"{k + 1}\t" + "\t".join(f"{x:.0f}" for x in cost[k]) + "\t" + "\t".join(str(x) for x in carried[k]) + f"\t{deepn[k]}\n")
    return out


def trace_end(steps, it):
    e = -1
    for k, st in enumerate(steps):
        for segs in st:
            for (i, a, b) in segs:
                if i == it: e = k
    return e


def main():
    a = sys.argv[1:]
    spec = json.load(open(a[0])); outp = a[1]; tsv = a[a.index("--steps-tsv") + 1] if "--steps-tsv" in a else None
    plan = build(spec)
    items = [[x[0], x[1], x[2], x[3]] for x in plan["items"]]
    chk = D.plan([(0, x[1], x[2], x[3]) for x in items], None, spec["V"], spec["Q"], queues=plan["queues"])
    assert chk == plan["steps"], "data181.plan on the queues differs from the simulation"
    summ = report(spec, plan, tsv)
    gold = {s.golden: {"dir": s.dir, "genmask": s.genmask} for s in plan["srcs"].values()}
    out = {"format": "plan231", "V": spec["V"], "Q": spec["Q"], "ranks": spec.get("ranks", 1), "goldens": gold, "items": items, "queues": plan["queues"],
           "budget_rows": spec["budget_rows"], "spec": spec, "summary": summ}
    json.dump(out, open(outp + ".part", "w")); os.replace(outp + ".part", outp)
    print(json.dumps(summ, indent=1, default=str))
    if not summ["trace_ends_by_last"]: print("WARNING: not every deep trace trains whole by the budget's step (too many / too long for the carried bound and max_long, or starts too late): fewer traces, a larger bound or earlier starts")
    if summ["forced_over_bound"]: print(f"WARNING: {summ['forced_over_bound']} documents taken over the carried bound (nothing else fitted)")


if __name__ == "__main__": main()
