import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_stub import keys, groups, put
T = sys.argv[1]
for d in ("m08", "meta"): os.makedirs(f"{T}/{d}", exist_ok=True)
rng = np.random.default_rng(1)
for g, nd in (("g08-search1", 40), ("g08-fit1", 20)):
    docs, r0 = [], 0
    for i in range(nd): t = int(rng.integers(100, 300)); docs.append({"tokens": t, "score_from": 0, "row0": r0, "orig": i}); r0 += t - 1
    os.makedirs(f"{T}/{g}", exist_ok=True); json.dump({"rows": r0, "docs": docs}, open(f"{T}/{g}/kld.json", "w"))
for k in (3, 2):
    for s in ("", "_s1", "_s2", "_s3"):
        for l in range(24):
            for key in keys(l): put(f"{T}/yb/q-k{k}base{s}/wq/{key}.safetensors", {"weight": torch.full((2, 2), float(hash((k, s, key)) % 997))})
for l in range(24):
    for g in groups(l): put(f"{T}/h08/{g}.safetensors", {"hessian_upper": torch.tensor([1.0, 0.1, 1.0])}, {"k": "2", "rows": "1000"})
# base search scores via the evalq stub's own formula
sys.argv = ["evalq"]; import importlib.util
spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalq.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
for k in (3, 2):
    for s in ("", "_s1", "_s2"): ev.score(f"{T}/g08-search1", f"{T}/yb/q-k{k}base{s}/wq", f"{T}/yb/k{k}base{s}.f32")
