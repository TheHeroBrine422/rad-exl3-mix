# w149 (#178-03) RULES-17803, on ds8: per-document KL sums (rxrun.hostd.doc_sums) of every w149/scores/*-s173f.json.rows
# -> JSON on stdout (keys "<size>|<arm>")
import glob, json, os, re, sys
sys.path.insert(0, "/data/q27/stage/code/pipeline")
from rxrun.hostd import doc_sums
Q = "/data/q27"; G = f"{Q}/w131/slices/golden-s173f"; out = {}
for r in sorted(glob.glob(f"{Q}/w149/scores/*-s173f.json.rows")):
    s, a = re.match(r"([\d.]+)-(\w+)-s173f\.json\.rows$", os.path.basename(r)).groups()
    d, kl = doc_sums(G, r)
    out[f"{s}|{a}"] = {"docs": d, "kld": float(kl.mean()), "positions": int(len(kl)), "rows": r}
json.dump(out, sys.stdout)
