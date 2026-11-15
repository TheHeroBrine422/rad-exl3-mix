# w179 supply for the 4:40 PM amendment (s190e Latin reuse): read-only counts. Run on ds8 with draw190ef's paths.
import collections, json, sys
sys.path.insert(0, "/data/q27/w179/code/search/v3/run190")
import draw190ef as D
import numpy as np
K = json.load(open(D.CAL))["docs"]; P = json.load(open(D.POOLS))["pool"]; GEN = np.fromfile(D.CALGEN, np.uint8)
sc = lambda i: min(K[i]["tokens"], D.C_CAP) - 1 - K[i]["score_from"]
gen_c = {i: int(GEN[d["row0"]:d["row0"] + sc(i)].sum()) for i, d in enumerate(K)}
REGJ = json.load(open(D.REG))
nr_c_set, reg_nr_c, ret_c_map, nr_x_set, ret_x_map = D.exclusion(REGJ)
ex_n, ret_n = D.n_calib_sets(REGJ)
fs = lambda i: P[i] in ("fit", "search")
SEARCH_RET = set()
for name in ["s173a", "s173b", "s173c", "s173d", "s173e", "s186a", "s186c", "s190a"]:
    assert name in D.RETIRED
    SEARCH_RET |= {d["orig"] for d in D.docs(D.slice_dir(name)) if d.get("orig") is not None}
V2 = set()
for l in open(D.LED):
    r = json.loads(l)
    if r["line"] == "v2" and r["doc"] != "all" and fs(int(r["doc"])): V2.add(int(r["doc"]))
L_fresh = [i for i in range(len(P)) if fs(i) and i not in nr_c_set and i not in set(ret_c_map) and K[i]["stratum"] == "L" and sc(i) >= 16]
L_reuse = [i for i in range(len(P)) if fs(i) and i not in ex_n and K[i]["stratum"] == "L" and sc(i) >= 16 and (i in SEARCH_RET or i in V2)]
L_reuse_only_search = [i for i in L_reuse if i in SEARCH_RET]; L_reuse_only_v2 = [i for i in L_reuse if i not in SEARCH_RET]
print(json.dumps({"fresh_L_docs": len(L_fresh), "fresh_L_gen": sum(gen_c[i] for i in L_fresh),
                  "reuse_L_docs": len(L_reuse), "reuse_L_gen": sum(gen_c[i] for i in L_reuse),
                  "reuse_L_search_ret_docs": len(L_reuse_only_search), "reuse_L_v2_only_docs": len(L_reuse_only_v2),
                  "SEARCH_RET_docs": len(SEARCH_RET), "V2_fit_search_docs": len(V2),
                  "V2_excluded_by_ex_n": sum(1 for i in V2 if i in ex_n and K[i]["stratum"] == "L"),
                  "V2_L_docs_total": sum(1 for i in V2 if K[i]["stratum"] == "L" and sc(i) >= 16),
                  "V2_L_docs_in_ret_c_map": sum(1 for i in V2 if K[i]["stratum"] == "L" and sc(i) >= 16 and i in set(ret_c_map)),
                  "V2_L_docs_in_nr_c_set": sum(1 for i in V2 if K[i]["stratum"] == "L" and sc(i) >= 16 and i in nr_c_set)}, indent=1))
