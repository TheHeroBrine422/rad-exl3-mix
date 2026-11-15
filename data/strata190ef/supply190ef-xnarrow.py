# w179: the x-side N pool under the search-only reading (retired search slices only) vs the wide reading (as built). Read-only.
# Run on ds8 from the code tree:  cd /data/q27/w179/code/search/v3/run190 && python3 -I supply190ef-xnarrow.py
import json, sys
sys.path.insert(0, "/data/q27/w179/code/search/v3/run190")
import draw190ef as D
import numpy as np
REGJ = json.load(open(D.REG))
nr_c, reg_nr, ret_c_map, nr_x, ret_x = D.exclusion(REGJ)
X = [json.loads(l) for l in open(D.XSUP)]
G = {b: np.fromfile(D.XGM[b], np.uint8) for b in D.XGM}
XD = {b: json.load(open(f"{D.XB[b]}/kld.json"))["docs"] for b in D.XB}
def gen(x):
    d = XD[x["batch"]][x["idx"]]; n = d["tokens"] - 1 - d["score_from"]
    cap = D.CUT if (x["category"] in D.HEAVY and n > D.CUT) else 0
    n = min(n, cap) if cap else n
    return int(G[x["batch"]][d["row0"]:d["row0"] + n].sum())
SEARCH = ["s173a", "s173b", "s173c", "s173d", "s173e", "s186a", "s186c", "s190a"]
narrow = set()
for n in SEARCH:
    narrow |= {d["xdoc"] for d in D.docs(D.xdir(n)) if "xdoc" in d}
wide_N = [x for x in X if x["stratum"] == "N" and x["doc"] in ret_x and x["doc"] not in nr_x]
narrow_N = [x for x in X if x["stratum"] == "N" and x["doc"] in narrow and x["doc"] not in nr_x]
for name, pool in [("wide (retired search / confirm / reselect + s190a/b), as built", wide_N), ("narrow (retired search only)", narrow_N)]:
    print(name, "pieces", len(pool), "gen", sum(gen(x) for x in pool), "scripts", len({x["script"] for x in pool}))
