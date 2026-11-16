#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""convplot4.py STATE.jsonl OUT.png --refs R.jsonl [--refs R2.jsonl ...] [--results RESULTS.jsonl] [--monitor ver3-8k]
-- the V3 search's convergence plot in V2's layout (search/convplot.py, v2-convergence-final.png; #171, w134).

Inputs: the search driver's state (search/v3/driver.py drv/state.jsonl: submit / result / decision / incumbent /
monitor / rotate / act lines), reference rows (rxrun.multi format: tag ref-V2-<size> and ref-IQR4-<name>, slice, kld,
bpw; ref171.py / ref134.py) and optionally the coordinator's results.jsonl (the incumbents' own search-slice scores).

Scale: KLD ÷ V2 at the same bpw on the same slice (V2 log-linear in log KLD between its sizes; sizes outside V2's
scored range are left out). 1.0 = V2, lower is better. x = eval step (result lines of the state, in order).
Top, per bpw band (< 2.75, 2.75-4.5, 4.5-6, >= 6), geo-mean over the band's sizes:
  line   training estimate of the incumbents: each size's first monitor read x exp(sum of the accepted decisions'
         log r (the accept statistic: equal-weight mean of the slice parts' log ratios)) since that read's pick, ÷ V2
         at the incumbent's bpw. Anchored once per size, so it starts at the held-out value and the training /
         held-out gap shows as the line leaving the circles.
  dots / x   every decided eval (rejected / accepted): its incumbent's estimate x exp(its log r), per size.
  circles    held-out monitor reads (ver3-8k) ÷ V2 measured on ver3-8k.
  diamonds   (--results) the incumbents' full scores on the search slice ÷ V2 scored on that slice, where it is: with
             --docs and --slice-kld the equal-weight geo-mean of the parts' (c / x) ratios, as the accept statistic.
  dotted     IQR-V4 on ver3-8k ÷ V2 on ver3-8k (both measured), geo-mean over the band's sizes inside IQR's range.
Bottom: validation loss V2-style: per band the geo-mean of each size's latest held-out read (from the step every
size of the band has one), black = all sizes, dashed = the same picks' training estimates.
Modes: submits with "act" (A8, w132) and without (f16) are separate series: anchors, chains and lines restart at the
"act" state event (red line); f16 and A8 numbers are never in one line.
Re-selection (#189-02, RULES-confirm rule 9; incumbent events with "why": "reselect9"): a level reset, nothing chained. From
that step the size's line is anchored on the new incumbent's own read (a monitor / --anchor-track read of the same assignment
hash, any step); without such a read the level is held (the chain goes on from where it was) until the plot is redrawn with one."""
import argparse, collections, json, math, os, re, sys, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BN = ["b1 ≤ 2.5 bpw", "b2 3–4 bpw", "b3 4.5–6 bpw", "b4 (unused)"]; COL = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]   # = the #183 fit bands (user 11-14)
DROP = {7.0}                                                    # (#183, user 11-14: 7.0 stopped, not plotted)
band = lambda b: 0 if b < 2.75 else 1 if b < 4.25 else 2 if b < 6.5 else 3
gmean = lambda v: math.exp(sum(map(math.log, v)) / len(v))
AMARK = ["*", "P", "D", "^", "v", "s", "p", "h", "<", ">", "X", "8"]   # inbox allocators


def loglin(pts, b, extrap=False):
    """log-linear in KLD between neighbouring (bpw, kld) points (end segments within 0.03 bpw); None further out unless extrap"""
    pts = sorted(pts)
    if not pts or (not extrap and not pts[0][0] - 0.03 <= b <= pts[-1][0] + 0.03): return None   # 0.03: menu vs file bpw
    if len(pts) == 1: return pts[0][1]
    i = 0 if b <= pts[0][0] else len(pts) - 2 if b >= pts[-1][0] else max(j for j in range(len(pts) - 1) if pts[j][0] <= b)
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return math.exp(math.log(y0) + (b - x0) / (x1 - x0) * (math.log(y1) - math.log(y0)))


def load_refs(paths):
    """-> {slice: {"V2": [(bpw, kld)], "IQR4": [...]}}; one row per (tag, slice), the last one wins"""
    rows = {}
    for p in paths:
        if not p or not os.path.exists(p): continue
        for l in open(p):
            if not l.strip(): continue
            r = json.loads(l); m = re.match(r"^ref-(V2|IQR4)-", r.get("tag", ""))
            if m and r.get("kld"):
                rows[(r["tag"], r["slice"])] = (m.group(1), r["slice"], r["bpw"], r["kld"])
                for pn, pv in (r.get("parts") or {}).items():
                    if pn != "m" and pv.get("kld"): rows[(r["tag"], f"{r['slice']}|{pn}")] = (m.group(1), f"{r['slice']}|{pn}", r["bpw"], pv["kld"])
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, sl, b, k in rows.values(): out[sl][fam].append((b, k))
    return out


def walk(state_path):
    """replay the state -> per (size, mode) chains, monitor reads, decisions, events"""
    st = [json.loads(l) for l in open(state_path) if l.strip()]
    st = [e for e in st if e.get("size") is None or float(e["size"]) not in DROP]     # (#183) sizes left off the plot
    sub = {e["tag"]: e for e in st if e["ev"] == "submit"}
    mode = lambda tag: "A8" if (sub.get(tag) or {}).get("act") else "f16"
    step = 0; C = collections.defaultdict(float); inc = {}; cat = {}
    became = {e["tag"] for e in st if e["ev"] == "incumbent"}   # (#188/#189) a pool join that never became incumbent was
                                                                 # paired against the worst member: not chained (w167)
    accepts = collections.defaultdict(list)            # (size, mode) -> [(step, C)]
    reads, decs, events, incbpw, resets = [], [], [], {}, []
    for e in st:
        ev = e["ev"]
        if ev == "result": step += 1
        elif ev == "incumbent":
            s = float(e["size"]); m = mode(e["tag"]); inc[s] = e["tag"]
            if e["tag"] not in cat: cat[e["tag"]] = C[(s, m)]
            if not accepts[(s, m)]: accepts[(s, m)].append((step, C[(s, m)]))
            elif e["tag"] in cat and cat[e["tag"]] != C[(s, m)] and e.get("why") != "reselect9":   # (#189 go-live revert, lead
                C[(s, m)] = cat[e["tag"]]; accepts[(s, m)].append((step, C[(s, m)]))      # 11-14) back to an earlier model's chain value
            elif e.get("why") == "reselect9":            # (#189-02) a level reset: no log r, a grid point for the re-anchor
                resets.append(dict(step=step, size=s, mode=m, tag=e["tag"], C=C[(s, m)])); accepts[(s, m)].append((step, C[(s, m)]))
        elif ev == "decision" and e.get("decision") in ("accept", "reject") and e.get("logr") is not None:
            s = float(e["size"]); m = mode(e["tag"]); acc = e["decision"] == "accept" and e["tag"] in became
            ref = (sub.get(e["tag"]) or {}).get("ref")
            decs.append(dict(step=step, size=s, mode=m, logr=e["logr"], acc=acc, C=C[(s, m)], kind=e.get("kind"), tag=e["tag"], stage=e.get("stage"),
                             early=(sub.get(e["tag"]) or {}).get("early"), ref=ref))
            if acc:
                C[(s, m)] += e["logr"]; cat[e["tag"]] = C[(s, m)]; accepts[(s, m)].append((step, C[(s, m)]))
        elif ev == "monitor" and e.get("kld"):
            s = float(e["size"]); of = e.get("of")
            reads.append(dict(step=step, size=s, mode=mode(e["tag"]) if e["tag"] in sub else mode(of), kld=e["kld"],
                              bpw=e["bpw"], of=of, cof=cat.get(of)))
        elif ev == "rotate": events.append((step, "rotate", f"→ {e.get('to')}"))
        elif ev == "act": events.append((step, "act", f"A8 switch ({e.get('act', 'a8g128')})"))
    for t, e in sub.items():
        incbpw[t] = e.get("bpw")
    return dict(st=st, sub=sub, accepts=accepts, reads=reads, decs=decs, events=events, steps=step, cat=cat,
                incbpw=incbpw, inc_hist=[e for e in st if e["ev"] == "incumbent"], resets=resets)


def reanchor(W, anchor):
    """(#189-02) per (size, mode) the anchors in step order: [(from step, {"kld", "cof"})], the first one = `anchor`; a re-selection
    reset adds the new incumbent's own read (same assignment hash, same mode) with cof = the chain value at the reset. -> (segments,
    the resets held for want of a read)"""
    hof = lambda t: (W["sub"].get(t) or {}).get("hash")
    segs = {k: [(-1, r)] for k, r in anchor.items()}; held = []
    for x in W.get("resets", []):
        k = (x["size"], x["mode"]); h = hof(x["tag"])
        rd = [r for r in W["reads"] if r["mode"] == x["mode"] and h is not None and hof(r["of"]) == h]
        if k not in segs or not rd: held.append(x); continue
        segs[k].append((x["step"], {"kld": rd[0]["kld"], "cof": x["C"], "reset": x["tag"]}))
    return segs, held


def anchor_at(segs, k, step):
    """the anchor in force for (size, mode) at `step` (None before the size's first anchor)"""
    an = None
    for s0, r in segs.get(k, []):
        if s0 <= step: an = r
        else: break
    return an


def _tleg(axT, a):
    """the tracking panel's legend: in the gap above the panel (user 11-15) when the panels are stacked training-on-top"""
    if a.track and a.no_val and not a.track_top:
        axT.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=7, frameon=False)
    else:
        axT.legend(fontsize=7.5, loc="upper right")


def track_panel(axT, W, a, sizes):
    """(#183) the fixed tracking slice: each incumbent's full score there (one measurement each, no chaining), plotted
    at the step it became incumbent, ÷ V2 at the same bpw on the same slice (log-linear between V2's measured sizes, as
    the other panels); IQR-V4 ÷ V2 there as dotted lines per size. Raw KL until two V2 sizes are measured."""
    step_of, step = {}, 0
    for e in W["st"]:
        if e["ev"] == "result": step += 1
        elif e["ev"] == "incumbent": step_of.setdefault(e["tag"], (step, float(e["size"])))
    T = [json.loads(l) for l in open(a.track) if l.strip() and '"trk183-' in l]
    hof = lambda t: (W["sub"].get(t) or {}).get("hash")             # (#189) a restored / re-selected incumbent has a new tag:
    byh = {hof(r["tag"][len("trk183-"):]): r for r in T if hof(r["tag"][len("trk183-"):])}   # use the read of the same model
    have = {r["tag"][len("trk183-"):] for r in T}
    for t in step_of:
        if t not in have and hof(t) in byh: T.append(dict(byh[hof(t)], tag="trk183-" + t))
    refs = [json.loads(l) for l in open(a.track_refs)] if a.track_refs and os.path.exists(a.track_refs) else []
    iq = sorted((r["bpw"], r["kld"]) for r in refs if r["name"].startswith("iq"))
    v2 = sorted((r["bpw"], r["kld"]) for r in refs if r["name"].startswith("v2"))
    rel = len(v2) >= 2
    ser = collections.defaultdict(list)
    for r in T:
        t = r["tag"][len("trk183-"):]
        if t not in step_of or r.get("kld") is None: continue
        st_, size = step_of[t]; y = r["kld"]
        if size in DROP: continue
        if rel:
            q = loglin(v2, r["bpw"])
            if q is None: continue
            y = y / q
        ser[size].append((st_, y))
    # scatter: every tracking point, coloured by band; lines: per band the geo-mean of each size's latest point, from
    # the step every size of the band has one (as the validation panel); IQR-V4 ÷ V2 per band dotted
    for size in sorted(ser):
        for x, y in ser[size]: axT.scatter([x], [y], s=16, color=COL[band(size)], alpha=0.55, lw=0, zorder=3)
    ev = sorted({x for v in ser.values() for x, _ in v})
    for bd in range(len(BN)):
        bs = [z for z in ser if band(z) == bd]
        if not bs: continue
        pts = []
        for x in ev:
            last = [max((p for p in ser[z] if p[0] <= x), default=None) for z in bs]
            if any(l is None for l in last): continue
            pts.append((x, math.exp(sum(math.log(l[1]) for l in last) / len(last))))
        if pts:
            pts.append((W["steps"], pts[-1][1]))
            axT.step(*zip(*pts), where="post", color=COL[bd], lw=2.2, label=f"{BN[bd]} {sorted(bs)}: {pts[-1][1]:.3f}")
        if rel and len(iq) >= 2:
            r = [loglin(iq, z) / loglin(v2, z) for z in bs if loglin(iq, z) and loglin(v2, z)]
            if r: axT.axhline(math.exp(sum(map(math.log, r)) / len(r)), color=COL[bd], ls=":", lw=1.5,
                              label=f"IQR-V4 {BN[bd]}: {math.exp(sum(map(math.log, r)) / len(r)):.3f}")
    if rel:
        axT.axhline(1.0, color="k", lw=1); axT.set_ylim(0.3, 2.0)
        axT.set_ylabel(f"KLD ÷ V2 at the same bpw ({a.track_name})")
    else:
        axT.set_yscale("log"); axT.set_ylabel(f"raw KL on {a.track_name} (V2 refs pending: {len(v2)} of 2+)")
    axT.set_yscale("log"); axT.grid(alpha=.3, which="both"); _tleg(axT, a)
    axT.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); axT.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if rel:                                                     # log scale, range fitted to the data (user 11-14)
        ys = [y for v in ser.values() for _, y in v] + [1.0]
        lo, hi = max(0.2, min(ys) * 0.9), min(3.0, max(max(ys) * 1.1, 1.05))
        axT._fit_ylim = (lo, hi); axT.set_ylim(lo, hi)
        axT.set_yticks([t for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0] if lo <= t <= hi])
    axT.set_title(f"Fixed tracking slice {a.track_name} (never decides): dots = incumbents' full scores, lines = band geo-means, "
                  f"dotted = IQR-V4 (refs: V2 {len(v2)}, IQR {len(iq)})", fontsize=10)
    if getattr(a, "track_strata", None) or getattr(a, "track_n", None):
        strata_overlay(axT, a, step_of)


def strata_overlay(axT, a, step_of):
    """(#186, RULES-strata §7, w164) the non-Latin supplement on the tracking panel, as ratios to V2 at the same bpw
    (log-linear between V2's measured sizes, as above): IQR-V4 references on t186n's N documents and their board
    combination 0.961 L + 0.039 N, per band (geo-mean over the band's sizes, dotted N / dash-dot board, as the band lines
    above; --track-strata: refs186.jsonl, whose board uses t183's L-only reads); the incumbents' t186n reads (--track-n:
    the trk186n-<tag> eval rows at their step; t186n is all N, so the eval's kld is the N-only KL). The t183 band lines
    above are the full t183 score (3.8 % of its positions are N). No V2 curve: nothing drawn until two V2 sizes have t186n
    readings."""
    S = [json.loads(l) for l in open(a.track_strata)] if a.track_strata and os.path.exists(a.track_strata) else []
    S = [r for r in S if r.get("t186n_N") is not None and r.get("bpw")]
    V2 = [r for r in S if r["name"].startswith("v2")]; IQ = [r for r in S if r["name"].startswith("iq")]
    if len(V2) < 2 and not (a.track_n and os.path.exists(a.track_n)): return
    vs = {"L": sorted((r["bpw"], r["t183_L"]) for r in V2 if r.get("t183_L")),
          "N": sorted((r["bpw"], r["t186n_N"]) for r in V2),
          "B": sorted((r["bpw"], r["board"]) for r in V2 if r.get("board"))}
    ys = []
    if len(vs["N"]) >= 2:
        for bd in range(len(BN)):
            for k, key, ls, lab in (("N", "t186n_N", ":", "N"), ("B", "board", "-.", "board")):
                if len(vs[k]) < 2: continue
                rs = [r[key] / loglin(vs[k], r["bpw"]) for r in IQ if band(r["bpw"]) == bd and r.get(key) is not None
                      and loglin(vs[k], r["bpw"])]
                if rs:
                    g = math.exp(sum(map(math.log, rs)) / len(rs)); ys.append(g)
                    axT.axhline(g, color=COL[bd], ls=ls, lw=1.3, alpha=.9, label=f"IQR-V4 {lab} {BN[bd]}: {g:.3f}")
    if a.track_n and os.path.exists(a.track_n) and len(vs["N"]) >= 2:
        pts = []
        for l in open(a.track_n):
            if '"trk186n-' not in l: continue
            r = json.loads(l); t = r["tag"][len("trk186n-"):]
            if r.get("kld") is None or t not in step_of: continue
            q = loglin(vs["N"], r["bpw"])
            if q: pts.append((step_of[t][0], r["kld"] / q, step_of[t][1]))
        if pts:
            axT.scatter([x for x, _, _ in pts], [y for _, y, _ in pts], marker="s", s=16, color="#a33", alpha=0.7, lw=0, zorder=3,
                        label="incumbents on t186n N (s)"); ys += [y for _, y, _ in pts]
    if ys and getattr(axT, "_fit_ylim", None):              # keep every N / board line on the panel (the fitted range is applied last)
        lo, hi = axT._fit_ylim; axT._fit_ylim = (min(lo, min(ys) * 0.95), max(hi, max(ys) * 1.05)); axT.set_ylim(*axT._fit_ylim)
    if axT.get_legend_handles_labels()[0]: _tleg(axT, a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="convplot4.py")
    ap.add_argument("state"); ap.add_argument("out")
    ap.add_argument("--refs", action="append", default=[]); ap.add_argument("--results")
    ap.add_argument("--monitor", default="ver3-8k"); ap.add_argument("--docs", help="the coordinator's per-document sums dir")
    ap.add_argument("--slice-kld", action="append", default=[], help="SLICE=kld.json: the slice's documents' parts (c / x)")
    ap.add_argument("--track", help="(#183) results.jsonl holding the tracking slice's trk183-<incumbent tag> rows: a middle panel")
    ap.add_argument("--track-refs", help="(#183) {name, bpw, kld} reference rows on the tracking slice (iq* / v2-*)")
    ap.add_argument("--track-name", default="t183")
    ap.add_argument("--track-strata", help="(#186) refs186.jsonl: the references' L (t183) / N (t186n) / board readings")
    ap.add_argument("--track-n", help="(#186) results.jsonl with trk186n-<incumbent tag> rows (the N supplement's incumbent reads)")
    ap.add_argument("--dpi", type=int, default=100, help="output resolution (the status loop uses 160)")
    ap.add_argument("--anchor-track", action="store_true", help="(#186, user 11-14) training lines anchored on each size's first --track read, ÷ V2 on the tracking slice; IQR lines there; no monitor circles")
    ap.add_argument("--track-top", action="store_true", help="(#186) the tracking panel on top, the training estimate below")
    ap.add_argument("--no-val", action="store_true", help="(#183) drop the held-out validation panel")
    ap.add_argument("--note", action="append", default=[], help="extra footnote line"); ap.add_argument("--title", default="EXL3-V3 search (#171)")
    ap.add_argument("--drop", default="7", help="(#199) comma list of sizes left off the plot (band geo-means keep a fixed size set)")
    ap.add_argument("--bands", default="", help="(11-16) fixed size sets, one line each: '2,2.5;3,3.5,4;...' (labels = the sizes)")
    ap.add_argument("--xmin", type=float, default=None, help="(#199) left edge of the step axis")
    ap.add_argument("--only", default="", help="(#199) comma list: plot only these sizes (e.g. the in-between sizes)")
    a = ap.parse_args(argv)
    global DROP, band, BN, COL
    DROP = {float(x) for x in a.drop.split(",") if x}
    if a.bands:                                                                   # (11-16) one figure, fixed size sets per line
        B = [[float(x) for x in g.split(",")] for g in a.bands.split(";")]
        allb = {x for g in B for x in g}
        DROP = {float(json.loads(l)["size"]) for l in open(a.state) if '"size"' in l and json.loads(l).get("size") is not None} - allb
        band = lambda b, B=B: min(range(len(B)), key=lambda i: min(abs(x - b) for x in B[i]))
        BN = [(f"{g[0]:g}–{g[-1]:g} bpw" if len(g) > 1 else f"{g[0]:g} bpw") for g in B]
        COL = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"][:len(B)]
    if a.only:
        keep = {float(x) for x in a.only.split(",")}
        ks = sorted(keep)                                                         # (#199, user) one line per size on this plot
        band = lambda b, ks=ks: min(range(len(ks)), key=lambda i: abs(ks[i] - b))
        BN = [f"{k:g} bpw" for k in ks]
        COL = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"][:len(ks)]
        DROP = {float(json.loads(l)["size"]) for l in open(a.state) if '"size"' in l and json.loads(l).get("size") is not None} - keep
    R = load_refs(a.refs + ([a.results] if a.results else []))
    V2m = R.get(a.monitor, {}).get("V2", []); IQm = R.get(a.monitor, {}).get("IQR4", [])
    v2 = lambda b, sl=a.monitor: loglin(R.get(sl, {}).get("V2", []), b)
    W = walk(a.state); sub = W["sub"]
    parts_of = {}
    for sk in a.slice_kld:
        sl, f = sk.split("=", 1)
        if os.path.exists(f): parts_of[sl] = [d.get("part") or "m" for d in json.load(open(f))["docs"]]

    vA = v2; ylab_sl = a.monitor
    if a.anchor_track:                                  # (user 11-14) anchor + scale on the tracking slice, not ver3-8k
        tr = [json.loads(l) for l in open(a.track_refs)] if a.track_refs and os.path.exists(a.track_refs) else []
        V2t = sorted((r["bpw"], r["kld"]) for r in tr if r["name"].startswith("v2"))
        IQm = sorted((r["bpw"], r["kld"]) for r in tr if r["name"].startswith("iq"))
        vA = lambda b: loglin(V2t, b); ylab_sl = a.track_name
        so, step = {}, 0
        for e in W["st"]:
            if e["ev"] == "result": step += 1
            elif e["ev"] == "incumbent": so.setdefault(e["tag"], (step, float(e["size"])))
        rd = []
        for l in open(a.track):
            if '"trk183-' not in l: continue
            r = json.loads(l); t = r["tag"][len("trk183-"):]
            if r.get("kld") is None or t not in so or so[t][1] in DROP: continue
            rd.append(dict(step=so[t][0], size=so[t][1], mode="A8" if (sub.get(t) or {}).get("act") else "f16",
                           kld=r["kld"], bpw=r["bpw"], of=t, cof=W["cat"].get(t)))
        W["reads"] = sorted(rd, key=lambda r: r["step"])
    # anchors: the first monitor read per (size, mode) whose pick's chain value is known
    anchor = {}
    for r in W["reads"]:
        k = (r["size"], r["mode"])
        if k not in anchor and r["cof"] is not None: anchor[k] = r
    def chain_at(k, step):                              # cumulative accepted log r of (size, mode) at step
        c = None
        for s_, c_ in W["accepts"].get(k, []):
            if s_ <= step: c = c_
            else: break
        return c
    # incumbent bpw over time per size (menu bpw of the submit)
    incb = collections.defaultdict(list)
    step = 0
    for e in W["st"]:
        if e["ev"] == "result": step += 1
        elif e["ev"] == "incumbent": incb[float(e["size"])].append((step, W["incbpw"].get(e["tag"]) or float(e["size"])))
    def bpw_at(s, step):
        b = float(s)
        for s_, b_ in incb.get(s, []):
            if s_ <= step: b = b_
        return b
    segs_a, held = reanchor(W, anchor)                 # (#189-02) re-selection = level reset on the new incumbent's own read
    def est(k, step, extra=0.0, b=None):               # training estimate ÷ V2 for (size, mode) at step
        an = anchor_at(segs_a, k, step); c = chain_at(k, step)
        if an is None or c is None: return None
        b = b or bpw_at(k[0], step); v = vA(b)
        return None if v is None else an["kld"] * math.exp(c - an["cof"] + extra) / v

    # mode segments: [start, end) steps
    act = [s for s, kind, _ in W["events"] if kind == "act"]
    segs = [("f16", 0, act[0] if act else W["steps"] + 1)] + ([("A8", act[0], W["steps"] + 1)] if act else [])
    sizes = sorted({s for s, _ in W["accepts"]})
    alloc_of = {}                                       # inbox candidate tag -> allocator label (file name / note)
    for e in W["st"]:
        if e["ev"] == "inbox" and e.get("mode") == "decide":
            f = e.get("file", "")                       # (user 11-16) three groups only
            if f.startswith(("w172-", "w181-")): lab = "worker allocators (#193)"
            elif "cons" in f or "xavg" in f or "consensus" in (e.get("note") or "") or "probe" in f: lab = "averaging (consensus / xavg)"
            else: lab = "other hand-made"
            for t in e.get("tags") or [e.get("tag")]: alloc_of[t] = lab
    if a.track:
        fig = plt.figure(figsize=(13.5, 16)); gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.1, 1])
        if a.no_val:                                        # (#183, user 11-14: no in-search held-out panel)
            fig = plt.figure(figsize=(15, 10.5)); gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.1], hspace=0.2)
            if a.track_top:                                 # (user 11-14) held-out tracking on top, training estimate below
                axT = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=axT)
                ax.set_xlabel("eval step (search results, in order)")
            else:
                ax = fig.add_subplot(gs[0]); axT = fig.add_subplot(gs[1], sharex=ax)
                axT.set_xlabel("eval step (search results, in order)")
            ax2 = plt.figure().add_subplot()                # drawn into a throwaway figure, never saved
        else:
            ax = fig.add_subplot(gs[0]); axT = fig.add_subplot(gs[1], sharex=ax); ax2 = fig.add_subplot(gs[2], sharex=ax)
        track_panel(axT, W, a, sizes)
    else:
        fig = plt.figure(figsize=(13.5, 11.5)); gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1])
        ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax)
    skipped = set(); summary = []
    for m, s0, s1 in segs:
        ks = [(s, m) for s in sizes if (s, m) in anchor and v2(bpw_at(s, s0)) is not None]
        if m == segs[-1][0]: skipped = {s for s in sizes if (s, m) not in anchor or v2(bpw_at(s, s0)) is None}
        grid = sorted({s0} | {st for k in ks for st, _ in W["accepts"][k] if s0 <= st < s1} | {min(s1, W["steps"])} )
        for bd in range(len(BN)):
            kb = [k for k in ks if band(k[0]) == bd]
            if not kb: continue
            ys = []
            for st in grid:
                v = [est(k, st) for k in kb]
                if all(x is not None for x in v): ys.append((st, gmean(v)))
            if ys:
                lab = f"{BN[bd]} [{', '.join(f'{k[0]:g}' for k in kb)}]: training est. ({m}) {ys[-1][1]:.3f}"
                ax.step(*zip(*ys), where="post", color=COL[bd], lw=2.2, ls="-" if m == "f16" else (0, (6, 1.5)), label=lab)
                summary.append(f"{m} {BN[bd]}: training est. {ys[0][1]:.3f} -> {ys[-1][1]:.3f}")
        # evals (inbox candidates drawn separately below, by allocator)
        for d in W["decs"]:
            if d["mode"] != m or not (s0 <= d["step"] < s1) or d["tag"] in alloc_of: continue
            k = (d["size"], m); e0 = est(k, d["step"])
            if e0 is None: continue
            y = e0 * math.exp(d["logr"]); c = COL[band(d["size"])]
            if d["acc"]: ax.scatter([d["step"]], [y], marker="x", s=30, color=c, lw=1.4, zorder=3)
            else: ax.scatter([d["step"]], [y], s=7 if d["early"] else 10, alpha=0.25, color=c, lw=0)
        # (#192/#193, user 11-15) inbox candidates: marker shape = allocator, fill = outcome (hollow: rejected at stage 1;
        # grey: failed confirmation; band colour + black edge: confirmed accept); one legend entry per allocator with counts
        last = {}
        for d in W["decs"]:
            if d["tag"] in alloc_of and d["mode"] == m and s0 <= d["step"] < s1: last[d["tag"]] = d
        for t, d in last.items():
            k = (d["size"], m); e0 = est(k, d["step"])
            if e0 is None: continue
            al = alloc_of[t]; mk = AMARK[sorted(set(alloc_of.values())).index(al) % len(AMARK)]
            y = e0 * math.exp(d["logr"]); c = COL[band(d["size"])]
            if d["acc"]: ax.scatter([d["step"]], [y], marker=mk, s=30, color=c, edgecolors="k", lw=0.8, zorder=5)
            elif d["stage"] == 2: ax.scatter([d["step"]], [y], marker=mk, s=14, color="0.6", edgecolors=c, lw=0.5, zorder=4)
            else: ax.scatter([d["step"]], [y], marker=mk, s=12, facecolors="none", edgecolors=c, lw=0.5, alpha=0.6, zorder=4)
        # held-out reads
        for r in W["reads"]:
            if r["mode"] != m or a.anchor_track: continue   # (anchor-track: the reads are the anchors, drawn on the track panel)
            v = vA(r["bpw"])
            if v is None: continue
            ax.scatter([r["step"]], [r["kld"] / v], s=34, facecolors="none", edgecolors=COL[band(r["size"])], lw=1.2, zorder=4)
    # incumbents' own full scores on the search slice ÷ V2 on that slice
    nd = 0
    if a.results and os.path.exists(a.results):
        incs = {e["tag"] for e in W["inc_hist"]}; full = collections.defaultdict(int); rows = []
        for l in open(a.results):
            r = json.loads(l) if l.strip() else {}
            if r.get("tag") in incs and r.get("kld") and not (r.get("fail") or r.get("flag")):
                rows.append(r); full[r["slice"]] = max(full[r["slice"]], r.get("positions") or 0)
        rstep = {}; step = 0
        for e in W["st"]:
            if e["ev"] == "result": step += 1; rstep.setdefault(e["tag"], step)
        for r in rows:
            if (r.get("positions") or 0) < 0.98 * full[r["slice"]] or r["tag"] not in rstep: continue
            b = sub[r["tag"]].get("bpw") or r["bpw"]; y = None
            dp = f"{a.docs}/{r['tag']}-{r['slice']}.json" if a.docs else None
            if r["slice"] in parts_of and dp and os.path.exists(dp):     # equal-weight mean of the parts' log ratios
                dd = json.load(open(dp))["m"]; acc = collections.defaultdict(lambda: [0.0, 0])
                for (sm, n), pn in zip(dd, parts_of[r["slice"]]):
                    if sm is not None: acc[pn][0] += sm; acc[pn][1] += n
                lr = [math.log(acc[pn][0] / acc[pn][1] / v2(b, f"{r['slice']}|{pn}")) for pn in acc
                      if acc[pn][1] and v2(b, f"{r['slice']}|{pn}")]
                if lr and len(lr) == len(acc): y = math.exp(sum(lr) / len(lr))
            elif v2(b, r["slice"]): y = r["kld"] / v2(b, r["slice"])
            if y is None: continue
            ax.scatter([rstep[r["tag"]]], [y], marker="D", s=16, color=COL[band(b)], alpha=0.7, lw=0); nd += 1
    # IQR-V4 lines (measured on the monitor slice)
    for bd in range(len(BN)):
        v = [loglin(IQm, s) / vA(s) for s in sizes if band(s) == bd and loglin(IQm, s) and vA(s)]
        if v:
            y = gmean(v)
            ax.axhline(y, color=COL[bd], ls=":", lw=1.6, label=f"IQR-V4 {BN[bd]} [{', '.join(f'{s:g}' for s in sizes if band(s) == bd and loglin(IQm, s) and vA(s))}], measured: {y:.3f}")
    for st, kind, lab in W["events"]:
        for x_ in (ax, ax2):
            x_.axvline(st, color="red" if kind == "act" else "gray", ls="--" if kind == "act" else ":", lw=1.6 if kind == "act" else 1)
        ax.text(st, 0.995, " " + lab, fontsize=8, color="red" if kind == "act" else "gray", rotation=90, va="top",
                transform=ax.get_xaxis_transform())
    ax.axhline(1.0, color="k", lw=0.9)
    for x_ in (ax, ax2):
        x_.set_yscale("log"); x_.grid(alpha=0.3, which="both")
        x_.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); x_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if a.anchor_track:                                  # (user 11-14) range fitted to the lines (training + IQR), not 0.3–2
        ys = [y for ln in ax.get_lines() for y in ln.get_ydata() if isinstance(y, (int, float)) and 0 < y < 5]
        lo, hi = (max(0.2, min(ys) * 0.9), max(1.05, min(2.0, max(ys) * 1.05))) if ys else (0.3, 2.0)
        ax.set_ylim(lo, hi)
        ax.set_yticks([t for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0] if lo <= t <= hi])
    else:
        ax.set_ylim(0.3, 2.0); ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_ylabel(f"KLD ÷ V2 at the same bpw (V2 measured on {ylab_sl})")
    if alloc_of:
        res = {}
        for d in W["decs"]:
            if d["tag"] in alloc_of: res[d["tag"]] = d
        for i, al in enumerate(sorted(set(alloc_of.values()))):
            ts = [t for t, v in alloc_of.items() if v == al]; dn = [res[t] for t in ts if t in res]
            ax.scatter([], [], marker=AMARK[i % len(AMARK)], s=45, facecolors="none", edgecolors="k",
                       label=f"inbox {al}: {len(ts)} tried, {len(dn)} decided, {sum(1 for d in dn if d['acc'])} confirmed")
    out_leg = a.track and a.no_val and not a.track_top      # (user 11-15) legends in the gap between the panels
    if out_leg:
        h, l = ax.get_legend_handles_labels()
        if alloc_of:                                    # the inbox key as a legend entry (no marker)
            h.append(matplotlib.lines.Line2D([], [], lw=0)); l.append("inbox key: hollow = rejected at stage 1\n· grey = failed "
                                                                    "confirmation · filled, black edge = confirmed accept")
        ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=7, frameon=False)
    else:
        ax.legend(loc="upper right", fontsize=7.5)
        if alloc_of:
            ax.text(0.005, 0.01, "inbox candidates (shape = allocator): hollow = rejected at stage 1 · grey = passed stage 1, failed "
                    "confirmation · filled with black edge = confirmed accept", transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", boxstyle="round,pad=0.25"))
    ax.set_title(f"{a.title} convergence (lines: training estimate; dots: evals; x: accepted; " + (f"each size anchored on its first "
                 f"{ylab_sl} read" if a.anchor_track else f"open circles: held-out {a.monitor} reads") + ";\n◆: incumbents' full scores on their search slice ÷ V2 there (geo-mean of the c / x parts' ratios); dotted: IQR-V4 measured; lower is better)", fontsize=10)

    # bottom: validation
    for m, s0, s1 in segs:
        rd = [r for r in W["reads"] if r["mode"] == m and v2(r["bpw"]) is not None]
        szs = sorted({r["size"] for r in rd}); last = {}; series = collections.defaultdict(list)
        for r in rd:
            last[r["size"]] = r
            for bd in list(range(len(BN))) + ["all"]:
                kb = [s for s in szs if bd == "all" or band(s) == bd]
                if not kb or not all(s in last for s in kb): continue
                hv = gmean([last[s]["kld"] / v2(last[s]["bpw"]) for s in kb])
                tv = [None if last[s]["cof"] is None or (s, m) not in anchor else
                      anchor_at(segs_a, (s, m), r["step"])["kld"] * math.exp(last[s]["cof"] - anchor_at(segs_a, (s, m), r["step"])["cof"]) / v2(last[s]["bpw"]) for s in kb]
                series[bd].append((r["step"], hv, gmean(tv) if all(x is not None for x in tv) else None))
        for bd in range(len(BN)):
            if series[bd]:
                kb = [s for s in szs if band(s) == bd]
                ax2.plot([x for x, _, _ in series[bd]], [h for _, h, _ in series[bd]], drawstyle="steps-post", marker=".", ms=3,
                         color=COL[bd], lw=1.3, label=f"{a.monitor}, {BN[bd]} ({', '.join(f'{s:g}' for s in kb)}) [{m}] {series[bd][-1][1]:.3f}")
        if series["all"]:
            S = series["all"]
            ax2.plot([x for x, _, _ in S], [h for _, h, _ in S], drawstyle="steps-post", marker=".", ms=4, color="k", lw=2.6,
                     label=f"VALIDATION: all {len(szs)} sizes on held-out {a.monitor} (geo-mean) [{m}] {S[-1][1]:.3f}")
            T = [(x, t) for x, _, t in S if t is not None]
            if T:
                ax2.plot(*zip(*T), drawstyle="steps-post", color="k", ls="--", lw=1.2, alpha=0.7,
                         label=f"same picks, training estimate [{m}] {T[-1][1]:.3f} (gap {S[-1][1] / T[-1][1] - 1:+.1%})")
            summary.append(f"{m} validation (all {len(szs)} sizes): {S[0][1]:.3f} -> {S[-1][1]:.3f}"
                           + (f"; training est. {T[-1][1]:.3f}" if T else ""))
    ax2.axhline(1.0, color="k", lw=0.9); ax2.set_ylim(0.3, 2.0); ax2.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 2.0])
    ax2.set_xlabel("eval step (search results, in order)"); ax2.set_ylabel(f"score of the current picks ÷ V2 ({a.monitor})")
    ax2.set_title(f"Validation loss: the search's current picks scored on held-out {a.monitor}; stop when it stops falling", fontsize=10)
    ax2.legend(loc="upper right", fontsize=7.5)
    v2r = sorted(b for b, _ in V2m); iqr = sorted(b for b, _ in IQm); held_s = ", ".join(f"{x['size']:g}" for x in held)
    if a.anchor_track: v2r = sorted(b for b, _ in V2t)
    note = (f"V2 on {ylab_sl}: {len(v2r)} sizes ({v2r[0]:.2f}–{v2r[-1]:.2f} bpw)" if v2r else "no V2 refs") \
        + (f"; IQR-V4: {len(iqr)} ({iqr[0]:.2f}–{iqr[-1]:.2f})" if iqr else "; IQR-V4: none yet") \
        + f"; sizes without a {ylab_sl} read (not in lines): {', '.join(f'{s:g}' for s in sorted(skipped)) or 'none'}" \
        + f"; bpw = menu bpw (search containers still carry the R16 head in bf16 on disk)" \
        + (f"; re-selection (rule 9) level resets: {sum(len(v) - 1 for v in segs_a.values())} re-anchored"
           + (f", {len(held)} held (no read of the new incumbent yet: {held_s})" if held else "") if W["resets"] else "")
    note = "\n".join(w for n in [note] + a.note for w in textwrap.wrap(n, 230)); nl = note.count("\n") + 1
    fig.text(0.01, 0.003, note, fontsize=7.5, color="dimgray")
    if a.track and getattr(axT, "_fit_ylim", None):        # last word on the tracking panel's range (shared-x panels reset it)
        axT.set_ylim(*axT._fit_ylim)
    if a.track and a.no_val and not a.track_top:      # legends sit between the panels: fixed margins (tight_layout pads them)
        fig.subplots_adjust(left=0.055, right=0.73, top=0.94, bottom=0.05 + 0.012 * nl)
    else:
        fig.tight_layout(rect=(0, 0.012 * nl, 1, 1))
    if a.xmin is not None:                                                        # (#199) zoom on the steps after a start point
        for ax_ in fig.axes:
            if ax_.get_xlabel() or ax_.get_ylabel(): ax_.set_xlim(left=a.xmin)
    fig.savefig(a.out, dpi=a.dpi)
    print("\n".join(summary)); print(note); print(f"{W['steps']} results, {len(W['decs'])} decisions, {len(W['reads'])} held-out reads, {nd} slice diamonds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
