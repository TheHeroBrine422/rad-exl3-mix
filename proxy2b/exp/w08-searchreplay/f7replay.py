#!/usr/bin/env python3
"""f7replay.py WORK [--out OUT.json] -- F7 (ranked.md): does the stream's accuracy and win rate fall with the number of
tensors a candidate changes? Replay over V2's stream on the GPU box (gpubox; CPU only, read-only, ~1 min):
  WORK/evq/results.jsonl    every eval (tag, kld)
  WORK/knap/mixes/*.mix     every candidate's mix (tensor -> container; only differences from the base are listed)
  WORK/auto/fit-s<N>.txt    fit N's slice offsets ("slice offsets f2 0.845 f3 0.958")
  WORK/auto/frontier-s<N>.txt  fit N's predictions, made BEFORE its candidates were measured:
                            mean "pred KLD P", Thompson / tilt "pred (mean model) P", local search "pred P0 -> P"
For every measured stream candidate (a|f2|f3-knap-s<N>[-kind<k>]-sc4.00-e4-T):
  anchor  = the best mix measured at target T on any fit slice before it (results order; KLD / that fit's offsets)
  n       = tensors (and the embedding) in which the candidate differs from its anchor
  err     = log(measured / offset / predicted)  (the prediction its fit made; offset = fit N's for that slice)
  win     = measured / offset < anchor's by more than WIN_MARGIN (1%)
Per n-bin (1-2, 3-4, 5-8, 9-16, 17-32, 33+) and kind: count, mean err (bias), RMS err, win rate, mean gain per eval
(E[max(0, 1 - meas/anchor)]). Then the cap rule (README): for c in 2, 4, 8, 16, the slots of candidates with n > c
redistributed to n <= c at the latter's gain per eval; the KLD gained at the band's final best by that redistribution
is estimated as (gain per eval difference) x (evals moved) / (targets in the band): summed gain per eval approximates
the log-improvement of a target's best when wins are small; optimistic, since later wins start from a better best.
Also: error vs moves since the last re-anchor ("recenter every ~16 moves"), i.e. how many ACCEPTED (winning) moves
lie between the candidate's anchor and the fit's last full refit on measured data -- not available per candidate from
logs, so approximated by n."""
import collections, glob, json, math, os, re, sys

WORK = sys.argv[1]
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
WIN = float(os.environ.get("WIN_MARGIN", 0.01))
R = re.compile(r"(a|f2|f3)-knap-s(\d+)(?:-(ts|ls|tl|rd)\d+)?-sc4\.00-e4-([\d.]+)")

off = {}
for f in glob.glob(f"{WORK}/auto/fit-s*.txt"):
    n = int(re.search(r"fit-s(\d+)\.txt", f).group(1)); o = {"a": 1.0}
    m = re.search(r"slice offsets ((?:\w+ [\d.]+ ?)+)", open(f).read())
    if m: v = m.group(1).split(); o.update({v[i]: float(v[i + 1]) for i in range(0, len(v), 2)})
    off[n] = o
pred = {}
for f in glob.glob(f"{WORK}/auto/frontier-s*.txt"):
    for l in open(f):
        m = re.match(r"^(knap-\S+)\s+bpw [\d.]+\s+(?:pred KLD ([\d.]+)|pred \(mean model\) ([\d.]+)|from \S+ \(meas [\d.]+, pred [\d.]+ -> ([\d.]+))", l)
        if m: pred[m.group(1)] = float(next(g for g in m.groups()[1:] if g))
def readmix(name):
    try: return dict(l.split() for l in open(f"{WORK}/knap/mixes/{name}.mix") if l.strip())
    except FileNotFoundError: return None
def ndiff(a, b): return sum(a.get(w) != b.get(w) for w in set(a) | set(b))

rows, best = [], {}                      # best[T] = (normalized kld, mix dict, tag)
for l in open(f"{WORK}/evq/results.jsonl"):
    r = json.loads(l)
    if "kld" not in r: continue
    m = R.fullmatch(r["tag"])
    if not m: continue
    sl, N, kind, T = m.group(1), int(m.group(2)), m.group(3) or "mean", m.group(4)
    name = r["tag"].split("-", 1)[1]; mix = readmix(name)
    if mix is None or N not in off or sl not in off[N]: continue
    k = r["kld"] / off[N][sl]
    if T in best:
        k0, mix0, tag0 = best[T]
        row = {"tag": r["tag"], "T": float(T), "kind": kind, "fit": N, "n": ndiff(mix, mix0), "rel": k / k0,
               "win": k < k0 * (1 - WIN), "err": math.log(k / pred[name]) if name in pred else None}
        rows.append(row)
    if T not in best or k < best[T][0]: best[T] = (k, mix, r["tag"])
print(f"{len(rows)} stream candidates with an anchor; {sum(r['err'] is not None for r in rows)} with a logged prediction")

BINS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 6)]
def stats(rs):
    e = [r["err"] for r in rs if r["err"] is not None]
    return {"n": len(rs), "bias": sum(e) / len(e) if e else None, "rms": math.sqrt(sum(x * x for x in e) / len(e)) if e else None,
            "win": sum(r["win"] for r in rs) / len(rs) if rs else None,
            "gain": sum(max(0.0, 1 - r["rel"]) for r in rs) / len(rs) if rs else None}
def fmt(s): return " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items())
out = {"bins": {}, "kinds": {}, "caps": {}}
print("by n (tensors changed vs anchor):")
for lo, hi in BINS:
    s = stats([r for r in rows if lo <= r["n"] <= hi]); out["bins"][f"{lo}-{hi}"] = s
    if s["n"]: print(f"  n {lo:2d}-{hi if hi < 10 ** 6 else 'inf':>3}: {fmt(s)}")
for kind in ("mean", "ts", "ls", "tl", "rd"):
    rs = [r for r in rows if r["kind"] == kind]
    if not rs: continue
    out["kinds"][kind] = {f"{lo}-{hi if hi < 10 ** 6 else 'inf'}": stats([r for r in rs if lo <= r["n"] <= hi]) for lo, hi in BINS}
    print(f"  {kind:4}: " + " | ".join(f"{b}: n{v['n']} win {v['win']:.2f}" for b, v in out["kinds"][kind].items() if v["n"]))
BANDS = [(0, 2.75), (2.75, 4.5), (4.5, 6.0), (6.0, 99)]
print("cap replay (per band: share of evals above the cap, their gain/eval vs below, estimated final-best gain):")
for c in (2, 4, 8, 16):
    out["caps"][c] = {}
    for lo, hi in BANDS:
        rs = [r for r in rows if lo <= r["T"] < hi]
        if not rs: continue
        a_, b_ = [r for r in rs if r["n"] <= c], [r for r in rs if r["n"] > c]
        if not a_ or not b_: continue
        ga, gb = stats(a_)["gain"], stats(b_)["gain"]
        nT = len({r["T"] for r in rs}); est = (ga - gb) * len(b_) / nT      # extra summed gain a target (optimistic: gains shrink as the best improves)
        out["caps"][c][f"{lo}-{hi}"] = {"above": len(b_), "below": len(a_), "gain_below": ga, "gain_above": gb,
                                        "win_below": stats(a_)["win"], "win_above": stats(b_)["win"],
                                        "rms_below": stats(a_)["rms"], "rms_above": stats(b_)["rms"], "est": est}
        print(f"  cap {c:2d} band {lo}-{hi}: above {len(b_):4d} below {len(a_):4d}  gain/eval {gb:.4f} vs {ga:.4f}  "
              f"win {stats(b_)['win']:.2f} vs {stats(a_)['win']:.2f}  rms {stats(b_)['rms'] or 0:.3f} vs {stats(a_)['rms'] or 0:.3f}  "
              f"est. final-best gain {est:+.2%}")
if OUT: json.dump({"rows": rows, **out}, open(OUT, "w"), indent=0)
print("F7REPLAY DONE")
