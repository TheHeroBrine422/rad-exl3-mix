#!/usr/bin/env python3
"""kindstats.py [SINCE_FIT] -- is each kind of stream candidate pulling its weight? Per kind (mean, ts = Thompson,
ls = local search, tl = structured tilt, rd = random-direction tilt), over stream fits >= SINCE_FIT (default 0):
  evals     measured fit-slice evals
  paired    wins / pairs against the same fit's mean mix at the same target on the same slice
  new-best  evals that set a new best measured mix at their target ON THEIR OWN SLICE, in results order (2026-10-31:
            across slices, via the fitted offsets, it was badly inflated -- offsets are not accurate to 1-2%)
  med ratio median KLD / (the same fit's mean mix) over the pairs
Run on the GPU box (reads $RXM_WORK/evq/results.jsonl and $RXM_WORK/auto/fit-*.txt)."""
import collections, glob, json, os, re, sys
RXM_WORK = os.environ.get("RXM_WORK", "/root")
SINCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
off = {"a": 1.0}
for f in sorted(glob.glob(f"{RXM_WORK}/auto/fit-s*.txt"), key=os.path.getmtime, reverse=True):
    m = re.search(r"slice offsets ((?:\w+ [\d.]+ ?)+)", open(f).read())
    if m:
        v = m.group(1).split(); off.update({v[i]: float(v[i + 1]) for i in range(0, len(v), 2)}); break
R = re.compile(r"(a|f2|f3)-knap-s(\d+)(?:-(ts|ls|tl|rd)\d+)?-sc4\.00-e4-([\d.]+)")
rows, kld = [], {}
for l in open(f"{RXM_WORK}/evq/results.jsonl"):
    r = json.loads(l)
    if "kld" not in r: continue
    kld[r["tag"]] = r["kld"]
    m = R.fullmatch(r["tag"])
    if m: rows.append((m.group(1), int(m.group(2)), m.group(3) or "mean", m.group(4), r["kld"], r["tag"]))
best = {}
st = collections.defaultdict(lambda: [0, 0, 0, 0, []])   # evals, wins, pairs, new-best, ratios
for sl, n, kind, t, k, tag in rows:
    isbest = (sl, t) not in best or k < best[(sl, t)]
    if isbest: best[(sl, t)] = k
    if n < SINCE: continue
    s = st[kind]; s[0] += 1; s[3] += isbest
    mt = f"{sl}-knap-s{n}-sc4.00-e4-{t}"
    if kind != "mean" and mt in kld:
        s[2] += 1; s[1] += k < kld[mt]; s[4].append(k / kld[mt])
print(f"stream fits >= s{SINCE}; slice offsets {off}")
print(f"{'kind':6} {'evals':>6} {'paired':>9} {'win%':>5} {'new-best':>9} {'rate':>6} {'med ratio':>9}")
for kind in ("mean", "ts", "ls", "tl", "rd"):
    if kind not in st: continue
    e, w, p, nb, rr = st[kind]; rr.sort()
    print(f"{kind:6} {e:6d} {f'{w}/{p}' if p else '-':>9} {f'{w / p:.0%}' if p else '-':>5} {nb:9d} {nb / e:6.1%} "
          f"{f'{rr[len(rr) // 2]:.3f}' if rr else '-':>9}")
