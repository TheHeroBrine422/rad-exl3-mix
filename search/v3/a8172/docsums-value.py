# w132 (#172) RULES-value, on ds8: per-document KL sums (rxrun.hostd.doc_sums) of every value/scores/*.json.rows -> JSON
import glob, json, os, re, sys
sys.path.insert(0, "/data/q27/stage/code/pipeline")
from rxrun.hostd import doc_sums
Q = "/data/q27"; out = {}
for r in sorted(glob.glob(f"{Q}/w132/value/scores/*.json.rows")):
    t, sl = re.match(r"(.+)-(s171[cd])\.json\.rows$", os.path.basename(r)).groups()
    d, k = doc_sums(f"{Q}/w131/slices/golden-{sl}", r)
    lg = r[:-len(".json.rows")] + ".log"; s = None
    for l in open(lg):
        m = re.match(r"I kld: (\d+) documents in (\d+) s", l)
        if m: s = int(m.group(2))
    out[f"{t}|{sl}"] = {"docs": d, "kld": float(k.mean()), "positions": int(len(k)), "docs_s": s}
json.dump(out, sys.stdout)
