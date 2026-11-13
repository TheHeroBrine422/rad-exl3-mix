# w142 (#178) RULES-178, on ds8: per-document KL sums (rxrun.hostd.doc_sums) of every w142/scores/*-s171d.json.rows, and of
# arm B = w132's RULES-value score of the same mix (w132/value/scores/<tag>-s171d) -> JSON on stdout
import glob, json, os, re, sys
sys.path.insert(0, "/data/q27/stage/code/pipeline")
from rxrun.hostd import doc_sums
Q = "/data/q27"; G = f"{Q}/w131/slices/golden-s171d"; out = {}
TAG = {"2.5": "v3s171-2.5-block-00626", "4.0": "v3s171-4-block-00420"}
rows = {f"{s}|B": f"{Q}/w132/value/scores/{t}-s171d.json.rows" for s, t in TAG.items()}
for r in sorted(glob.glob(f"{Q}/w142/scores/*-s171d.json.rows")):
    s, a = re.match(r"([\d.]+)-(\w+)-s171d\.json\.rows$", os.path.basename(r)).groups(); rows[f"{s}|{a}"] = r
for k, r in rows.items():
    if not os.path.exists(r): continue
    d, kl = doc_sums(G, r)
    out[k] = {"docs": d, "kld": float(kl.mean()), "positions": int(len(kl)), "rows": r}
json.dump(out, sys.stdout)
