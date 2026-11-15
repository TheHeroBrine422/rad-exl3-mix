#!/usr/bin/env python3
# extract_incumbents.py STATE.jsonl RESULTS.jsonl OUT.jsonl -- w177 (#194): our V3 incumbents on t183 at full precision, with
# the same selection as search/v3/run183/sizekl183.py (latest tracked incumbent per size, trk183-<tag> results, size 7.0 left
# out). Cross-checked against sizekl183.py's printed output in make_table.py's header (raw/sizekl183.txt).
import json, sys
st, res, out = sys.argv[1:4]
order = {}; i = 0
for l in open(st):
    e = json.loads(l)
    if e.get("ev") == "incumbent": order[e["tag"]] = i; i += 1
ours = {}
for l in open(res):
    if '"trk183-' not in l: continue
    r = json.loads(l); t = r["tag"][7:]
    if r.get("kld") is None or t not in order: continue
    s = float(t.split("-")[1])
    if s in (7.0,) or (s in ours and order[ours[s][2]] > order[t]): continue
    ours[s] = (r["bpw"], r["kld"], t)
with open(out, "w") as f:
    for s in sorted(ours):
        bpw, kld, t = ours[s]
        f.write(json.dumps({"size": s, "bpw": bpw, "kld_t183": kld, "tag": t}) + "\n")
print(len(ours), "incumbents ->", out)
