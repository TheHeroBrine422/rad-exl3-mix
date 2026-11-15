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
"act" state event (red line); f16 and A8 numbers are never in one line."""
import argparse, collections, json, math, os, re, sys, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BN = ["b1 ≤ 2.5 bpw", "b2 3–4 bpw", "b3 4.5–6 bpw", "b4 (unused)"]; COL = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]   # = the #183 fit bands (user 11-14)
DROP = {7.0}                                                    # (#183, user 11-14: 7.0 stopped, not plotted)
band = lambda b: 0 if b < 2.75 else 1 if b < 4.25 else 2 if b < 6.5 else 3
gmean = lambda v: math.exp(sum(map(math.log, v)) / len(v))


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
    accepts = collections.defaultdict(list)            # (size, mode) -> [(step, C)]
    reads, decs, events, incbpw = [], [], [], {}
    for e in st:
        ev = e["ev"]
        if ev == "result": step += 1
        elif ev == "incumbent":
            s = float(e["size"]); m = mode(e["tag"]); inc[s] = e["tag"]
            if e["tag"] not in cat: cat[e["tag"]] = C[(s, m)]
            if not accepts[(s, m)]: accepts[(s, m)].append((step, C[(s, m)]))
        elif ev == "decision" and e.get("decision") in ("accept", "reject") and e.get("logr") is not None:
            s = float(e["size"]); m = mode(e["tag"]); acc = e["decision"] == "accept"
            ref = (sub.get(e["tag"]) or {}).get("ref")
            decs.append(dict(step=step, size=s, mode=m, logr=e["logr"], acc=acc, C=C[(s, m)], kind=e.get("kind"),
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
                incbpw=incbpw, inc_hist=[e for e in st if e["ev"] == "incumbent"])


def track_panel(axT, W, a, sizes):
    """(#183) the fixed tracking slice: each incumbent's full score there (one measurement each, no chaining), plotted
    at the step it became incumbent, ÷ V2 at the same bpw on the same slice (log-linear between V2's measured sizes, as
    the other panels); IQR-V4 ÷ V2 there as dotted lines per size. Raw KL until two V2 sizes are measured."""
    step_of, step = {}, 0
    for e in W["st"]:
        if e["ev"] == "result": step += 1
        elif e["ev"] == "incumbent": step_of.setdefault(e["tag"], (step, float(e["size"])))
    T = [json.loads(l) for l in open(a.track) if l.strip() and '"trk183-' in l]
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
    for bd in range(4):
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
    axT.set_yscale("log"); axT.grid(alpha=.3, which="both"); axT.legend(fontsize=7.5, loc="upper right")
    axT.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); axT.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if rel:                                                     # the same scale and ticks as the other panels
        axT.set_ylim(0.3, 2.0); axT.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    axT.set_title(f"Fixed tracking slice {a.track_name} (never decides): dots = incumbents' full scores, lines = band geo-means, "
                  f"dotted = IQR-V4 (refs: V2 {len(v2)}, IQR {len(iq)})", fontsize=10)
    if getattr(a, "track_strata", None) or getattr(a, "track_n", None):
        strata_overlay(axT, a, step_of)


def strata_overlay(axT, a, step_of):
    """(#186, RULES-strata §7, w164) the non-Latin supplement on the tracking panel, as ratios to V2 at the same bpw
    (log-linear between V2's measured sizes, as above): IQR-V4 references on t183's L documents (o), on t186n's N documents
    (s) and their board combination 0.961 L + 0.039 N (^) (--track-strata: refs186.jsonl); and the incumbents' t186n reads
    (--track-n: the trk186n-<tag> eval rows; t186n is all N, so the eval's kld is the N-only KL). No V2 curve: nothing drawn
    until two V2 sizes have t186n readings."""
    S = [json.loads(l) for l in open(a.track_strata)] if a.track_strata and os.path.exists(a.track_strata) else []
    S = [r for r in S if r.get("t186n_N") is not None and r.get("bpw")]
    V2 = [r for r in S if r["name"].startswith("v2")]; IQ = [r for r in S if r["name"].startswith("iq")]
    if len(V2) < 2 and not (a.track_n and os.path.exists(a.track_n)): return
    vs = {"L": sorted((r["bpw"], r["t183_L"]) for r in V2 if r.get("t183_L")),
          "N": sorted((r["bpw"], r["t186n_N"]) for r in V2),
          "B": sorted((r["bpw"], r["board"]) for r in V2 if r.get("board"))}
    style = {"L": ("o", "IQR-V4 t183 L (o)"), "N": ("s", "IQR-V4 t186n N (s)"), "B": ("^", "IQR-V4 board 0.961L+0.039N (^)")}
    ys = []
    if len(vs["N"]) >= 2:
        for k, (m_, lab) in style.items():
            if len(vs[k]) < 2: continue
            xs = [(r["bpw"], r[{"L": "t183_L", "N": "t186n_N", "B": "board"}[k]] / loglin(vs[k], r["bpw"])) for r in IQ
                  if r.get(({"L": "t183_L", "N": "t186n_N", "B": "board"})[k]) is not None and loglin(vs[k], r["bpw"])]
            if xs:
                axT.scatter(*zip(*xs), marker=m_, s=34, facecolors="none", edgecolors="#333", lw=1.1, zorder=4, label=lab); ys += [y for _, y in xs]
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
    if ys:                                                   # keep every N / board point on the panel (the default band is 0.3-2)
        lo, hi = axT.get_ylim(); axT.set_ylim(min(lo, min(ys) * 0.9), max(hi, max(ys) * 1.1))
    if axT.get_legend_handles_labels()[0]: axT.legend(fontsize=7.5, loc="upper right")


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
    ap.add_argument("--no-val", action="store_true", help="(#183) drop the held-out validation panel")
    ap.add_argument("--note", action="append", default=[], help="extra footnote line"); ap.add_argument("--title", default="EXL3-V3 search (#171)")
    a = ap.parse_args(argv)
    R = load_refs(a.refs + ([a.results] if a.results else []))
    V2m = R.get(a.monitor, {}).get("V2", []); IQm = R.get(a.monitor, {}).get("IQR4", [])
    v2 = lambda b, sl=a.monitor: loglin(R.get(sl, {}).get("V2", []), b)
    W = walk(a.state); sub = W["sub"]
    parts_of = {}
    for sk in a.slice_kld:
        sl, f = sk.split("=", 1)
        if os.path.exists(f): parts_of[sl] = [d.get("part") or "m" for d in json.load(open(f))["docs"]]

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
    def est(k, step, extra=0.0, b=None):               # training estimate ÷ V2 for (size, mode) at step
        an = anchor.get(k); c = chain_at(k, step)
        if an is None or c is None: return None
        b = b or bpw_at(k[0], step); v = v2(b)
        return None if v is None else an["kld"] * math.exp(c - an["cof"] + extra) / v

    # mode segments: [start, end) steps
    act = [s for s, kind, _ in W["events"] if kind == "act"]
    segs = [("f16", 0, act[0] if act else W["steps"] + 1)] + ([("A8", act[0], W["steps"] + 1)] if act else [])
    sizes = sorted({s for s, _ in W["accepts"]})
    if a.track:
        fig = plt.figure(figsize=(13.5, 16)); gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.1, 1])
        if a.no_val:                                        # (#183, user 11-14: no in-search held-out panel)
            fig = plt.figure(figsize=(13.5, 11.5)); gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.1])
            ax = fig.add_subplot(gs[0]); axT = fig.add_subplot(gs[1], sharex=ax)
            ax2 = plt.figure().add_subplot()                # drawn into a throwaway figure, never saved
            axT.set_xlabel("eval step (search results, in order)")
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
        for bd in range(4):
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
        # evals
        for d in W["decs"]:
            if d["mode"] != m or not (s0 <= d["step"] < s1): continue
            k = (d["size"], m); e0 = est(k, d["step"])
            if e0 is None: continue
            y = e0 * math.exp(d["logr"]); c = COL[band(d["size"])]
            if d["acc"]: ax.scatter([d["step"]], [y], marker="x", s=30, color=c, lw=1.4, zorder=3)
            else: ax.scatter([d["step"]], [y], s=7 if d["early"] else 10, alpha=0.25, color=c, lw=0)
        # held-out reads
        for r in W["reads"]:
            if r["mode"] != m: continue
            v = v2(r["bpw"])
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
    for bd in range(4):
        v = [loglin(IQm, s) / v2(s) for s in sizes if band(s) == bd and loglin(IQm, s) and v2(s)]
        if v:
            y = gmean(v)
            ax.axhline(y, color=COL[bd], ls=":", lw=1.6, label=f"IQR-V4 {BN[bd]} [{', '.join(f'{s:g}' for s in sizes if band(s) == bd and loglin(IQm, s) and v2(s))}], measured: {y:.3f}")
    for st, kind, lab in W["events"]:
        for x_ in (ax, ax2):
            x_.axvline(st, color="red" if kind == "act" else "gray", ls="--" if kind == "act" else ":", lw=1.6 if kind == "act" else 1)
        ax.text(st, 0.995, " " + lab, fontsize=8, color="red" if kind == "act" else "gray", rotation=90, va="top",
                transform=ax.get_xaxis_transform())
    ax.axhline(1.0, color="k", lw=0.9)
    for x_ in (ax, ax2):
        x_.set_yscale("log"); x_.grid(alpha=0.3, which="both")
        x_.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); x_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.3, 2.0); ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_ylabel(f"KLD ÷ V2 at the same bpw (V2 measured on {a.monitor})")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title(f"{a.title} convergence (lines: training estimate; dots: evals; x: accepted; open circles: held-out "
                 f"{a.monitor} reads;\n◆: incumbents' full scores on their search slice ÷ V2 there (geo-mean of the c / x parts' ratios); dotted: IQR-V4 measured; lower is better)", fontsize=10)

    # bottom: validation
    for m, s0, s1 in segs:
        rd = [r for r in W["reads"] if r["mode"] == m and v2(r["bpw"]) is not None]
        szs = sorted({r["size"] for r in rd}); last = {}; series = collections.defaultdict(list)
        for r in rd:
            last[r["size"]] = r
            for bd in list(range(4)) + ["all"]:
                kb = [s for s in szs if bd == "all" or band(s) == bd]
                if not kb or not all(s in last for s in kb): continue
                hv = gmean([last[s]["kld"] / v2(last[s]["bpw"]) for s in kb])
                tv = [None if last[s]["cof"] is None or (s, m) not in anchor else
                      anchor[(s, m)]["kld"] * math.exp(last[s]["cof"] - anchor[(s, m)]["cof"]) / v2(last[s]["bpw"]) for s in kb]
                series[bd].append((r["step"], hv, gmean(tv) if all(x is not None for x in tv) else None))
        for bd in range(4):
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
    v2r = sorted(b for b, _ in V2m); iqr = sorted(b for b, _ in IQm)
    note = (f"V2 on {a.monitor}: {len(v2r)} sizes ({v2r[0]:.2f}–{v2r[-1]:.2f} bpw)" if v2r else "no V2 refs") \
        + (f"; IQR-V4: {len(iqr)} ({iqr[0]:.2f}–{iqr[-1]:.2f})" if iqr else "; IQR-V4: none yet") \
        + f"; sizes without a held-out read (not in lines): {', '.join(f'{s:g}' for s in sorted(skipped)) or 'none'}" \
        + f"; bpw = menu bpw (search containers still carry the R16 head in bf16 on disk)"
    note = "\n".join(w for n in [note] + a.note for w in textwrap.wrap(n, 230)); nl = note.count("\n") + 1
    fig.text(0.01, 0.003, note, fontsize=7.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.012 * nl, 1, 1)); fig.savefig(a.out, dpi=100)
    print("\n".join(summary)); print(note); print(f"{W['steps']} results, {len(W['decs'])} decisions, {len(W['reads'])} held-out reads, {nd} slice diamonds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
