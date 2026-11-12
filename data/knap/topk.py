import json, sys, collections, os
sys.path.insert(0, "pipeline")
from rxq import config as C, knap, ladder
cfg = C.load()
lv = ladder.levels(cfg)
lib = {}
for x in lv:
    K = float(x["K"]) if isinstance(x, dict) else float(x[0]); cb = x["codebook"] if isinstance(x, dict) else x[1]
    lib[knap.level_name(cb, K)] = {"dir": f"/lv/{cb}/K{K:g}", "K": K, "codebook": cb, "pack": None}
print(len(lib), "levels; top:", sorted(v["K"] for v in lib.values())[-6:])
out = os.path.abspath("topk.jsonl")
if os.path.exists(out): os.remove(out)
rc = knap.solve(cfg, lib, out, os.getcwd(), [float(t) for t in sys.argv[1].split(",")], [0.0])
for l in open(out):
    d = json.loads(l); c = collections.Counter()
    for w, (o, m) in d["picks"].items():
        if "-K" in o and ("i8" in o or "i4" in o or "m1" in o): c[o] += 1
    ks = sorted(c, key=lambda o: float(o.split("-K")[1]))
    print(f"target {d['target']}: bpw {d['bpw']:.3f} pred KLD {d['kld']:.6f}  top trunk K picks: " + ", ".join(f"{o}x{c[o]}" for o in ks[-5:]), " head:", d.get("head", "")[:40] if isinstance(d.get("head"),str) else d.get("head"))
