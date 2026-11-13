# w132 (#172), on ds8: per-document KL sums (hostd.doc_sums: the golden's gen positions) of the f16 / A8 rows -> JSON
import json, os, sys
sys.path.insert(0, "/data/q27/stage/code/pipeline")
from rxrun.hostd import doc_sums
Q = "/data/q27"; G = {"ver3-8k": f"{Q}/goldens/golden-ver3-8k", "s171b": f"{Q}/w131/slices/golden-s171b"}
INC = {"2.5": ("v3s171-2.5-explore-00419", "v3s171-2.5-mon-00431", "v3s171-2.5-block-00413"),
       "3.5": ("v3s171-3.5-block-00483", "v3s171-3.5-mon-00497", "v3s171-3.5-pair-00396"),
       "5.0": ("v3s171-5-pair-00449", "v3s171-5-mon-00464", "v3s171-5-pair-00409"),
       "6.0": ("v3s171-6-explore-00372", "v3s171-6-mon-00383", "v3s171-6-inc-00358")}
out = {}
def add(key, g, rows, log=None):
    if not os.path.exists(rows): return
    d, k = doc_sums(G[g], rows)
    e = {"docs": d, "kld": float(k.mean()), "positions": int(len(k)), "rows": rows}
    j = rows[:-len(".rows")]
    if os.path.exists(j): e["all_kld"] = json.load(open(j))["all"]["kld"]["mean"]
    if log and os.path.exists(log):
        L = [l for l in open(log) if l.startswith("I kld: ") and " / " in l and "positions," in l]
        if L: e["last_progress"] = L[-1].strip()
        import re
        for l in open(log):
            m = re.match(r"I kld: (\d+) documents in (\d+) s", l)
            if m: e["docs_s"] = int(m.group(2))
    out[key] = e
for s, (t, mon, prev) in INC.items():
    add(f"{s}|inc|f16|ver3-8k", "ver3-8k", f"{Q}/scores/{mon}-ver3-8k-m.json.rows", f"{Q}/scores/{mon}-ver3-8k-m.log")
    add(f"{s}|inc|f16|s171b", "s171b", f"{Q}/scores/{t}-s171b-m.json.rows", f"{Q}/scores/{t}-s171b-m.log")
    add(f"{s}|prev|f16|s171b", "s171b", f"{Q}/scores/{prev}-s171b-m.json.rows")
    for g in ("ver3-8k", "s171b"):
        for m in ("a8", "f16"):
            o = f"{Q}/w132/scores/{t}-{m}-{g}"; add(f"{s}|inc|{m}|{g}|w132", g, o + ".json.rows", o + ".log")
    o = f"{Q}/w132/scores/{prev}-a8-s171b"; add(f"{s}|prev|a8|s171b|w132", "s171b", o + ".json.rows", o + ".log")
json.dump(out, sys.stdout)
