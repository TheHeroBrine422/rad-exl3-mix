#!/usr/bin/env python3
"""lscheck.py -- local search vs its starting mix (same slice = clean, cross slice = via fitted offsets), and held-out ver1 / fit ratio by candidate kind. Run on the GPU box."""
import json, re, glob
import os
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
r={}
for l in open(f"{RXM_WORK}/evq/results.jsonl"):
    x=json.loads(l)
    if "kld" in x: r[x["tag"]]=x["kld"]
start={}
for f in glob.glob(f"{RXM_WORK}/auto/frontier-s*.txt"):
    for l in open(f):
        m=re.match(r"^(knap-s\d+-ls\d-sc4\.00-e4-[\d.]+)\s.*from (\S+) \(meas ([\d.]+)",l)
        if m: start[m.group(1)]=(m.group(2),float(m.group(3)))
off={"a":1.0,"f2":0.848,"f3":0.958}
same=[];cross=[]
for t,k in r.items():
    m=re.fullmatch(r"(a|f2|f3)-(knap-s\d+-ls\d-sc4\.00-e4-[\d.]+)",t)
    if not m or m.group(2) not in start: continue
    st,k0=start[m.group(2)]; ss=st.split("-")[0]
    rat=(k/off[m.group(1)])/(k0/off[ss])
    (same if ss==m.group(1) else cross).append(rat)
for nm,v in (("same slice",same),("cross slice",cross)):
    v.sort(); print(f"{nm}: n={len(v)}  beat start {sum(x<1 for x in v)}/{len(v)}  median ratio {v[len(v)//2] if v else 0:.3f}")
# ver1/fit ratio by kind
print("held-out ver1 / fit-slice (normalized), by kind:")
for t,k in sorted(r.items()):
    m=re.fullmatch(r"w-knap-(.+)-sc4\.00-e4-([\d.]+)",t)
    if not m: continue
    name=m.group(1); tg=m.group(2)
    fits=[r[f"{s}-knap-{name}-sc4.00-e4-{tg}"]/off[s] for s in off if f"{s}-knap-{name}-sc4.00-e4-{tg}" in r]
    if not fits: continue
    kind="ls" if "-ls" in name else "ts" if "-ts" in name else "mean"
    print(f"  {tg:5} {name:10} {kind:5} ver1/fit {k/(sum(fits)/len(fits)):.2f}")
