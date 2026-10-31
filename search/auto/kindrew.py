#!/usr/bin/env python3
"""kindrew.py [SINCE_FIT] -- per candidate kind: log(measured / the producing fit's predicted KLD of its mean mix at the target), same-fit slice offsets (errors cancel within a fit). streamq.py's auto shares use the same reward."""
import json, re, glob, collections, math, os
W = os.environ.get("RXM_WORK", "/root")
pred, offs = {}, {}
for f in glob.glob(f"{W}/auto/frontier-s*.txt"):
    n = int(re.search(r"frontier-s(\d+)", f).group(1))
    for l in open(f):
        m = re.match(rf"^knap-s{n}-sc4\.00-e4-([\d.]+)\s+bpw [\d.]+\s+pred KLD ([\d.]+)", l)
        if m: pred[(n, m.group(1))] = float(m.group(2))
for f in glob.glob(f"{W}/auto/fit-s*.txt"):
    n = int(re.search(r"fit-s(\d+)", f).group(1)); t = open(f).read(4000)
    m = re.search(r"slice offsets ((?:\w+ [\d.]+ ?)+)", t)
    o = {"a": 1.0}
    if m: v = m.group(1).split(); o.update({v[i]: float(v[i + 1]) for i in range(0, len(v), 2)})
    offs[n] = o
R = re.compile(r"(a|f2|f3)-knap-s(\d+)(?:-(ts|ls|tl|rd)\d+)?-sc4\.00-e4-([\d.]+)")
st = collections.defaultdict(list)
import sys
since = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for l in open(f"{W}/evq/results.jsonl"):
    r = json.loads(l)
    m = R.fullmatch(r["tag"])
    if not m or "kld" not in r: continue
    sl, n, kind, t = m.group(1), int(m.group(2)), m.group(3) or "mean", m.group(4)
    if n < since or (n, t) not in pred or n not in offs: continue
    st[kind].append(math.log(r["kld"] / offs[n].get(sl, 1.0) / pred[(n, t)]))
print(f"fits >= s{since}: log(measured / this fit's predicted KLD of its MEAN mix at the target); <0 = beat the model's best")
for k in ("mean", "ts", "ls", "tl", "rd"):
    v = sorted(st.get(k, []))
    if v: print(f"  {k:5} n={len(v):4}  beat {sum(x < 0 for x in v) / len(v):5.0%}  median {math.exp(v[len(v)//2]):.3f}  mean {math.exp(sum(v)/len(v)):.3f}")
