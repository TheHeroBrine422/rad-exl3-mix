import json, re, sys, collections, numpy as np
exec(open("knap.py").read().split("# ------------------------------------------------------------------ calibration")[0].replace("D, targets = sys.argv[1], [float(x) for x in sys.argv[2].split(\",\")]", "D, targets = '.', []"))
res = {json.loads(l)["tag"]: json.loads(l)["kld"] for l in open("evq-results.jsonl") if "fail" not in l}
k0 = res[f"base-{BASE}"]
GROUPS = {"attn_in": (lambda l: l % 4 == 3, ["attn_qg", "attn_k", "attn_v"]), "attn_out": (lambda l: l % 4 == 3, ["attn_output"]),
          "ssm_in": (lambda l: l % 4 != 3, ["ssm_qkv", "ssm_z"]), "ssm_out": (lambda l: l % 4 != 3, ["ssm_out"]),
          "ffn_gu": (lambda l: True, ["ffn_gate_up"]), "ffn_down": (lambda l: True, ["ffn_down"])}
T2G = {t: g for g, (_, ts) in GROUPS.items() for t in ts}
sw = {}
for tag, k in res.items():
    m = re.match(rf"cal-{BASE}-(.+)-(\w+?)-L(\d+)$", tag)
    if not m: continue
    d, g, b0 = m.group(1), m.group(2), int(m.group(3)); sel, ts = GROUPS[g]
    ws = [f"blk.{l}.{t}.weight" for l in range(b0, b0 + 8) if sel(l) for t in ts]
    sw[(g, b0)] = (k - k0) / sum(opt[w][d][1] - opt[w][BASE][1] for w in ws)
med = np.median(list(sw.values()))
def s_of(w):   # band value, no interpolation: the swap's own average
    l, t = re.match(r"blk\.(\d+)\.(\w+)\.weight", w).groups(); l = int(l)
    return sw.get((T2G[t], l // 8 * 8), med)
pub = {"sc1.40-e3": .5749, "sc1.60-e3": .3802, "sc1.80-e3": .2466, "sc2.00-e3": .1480, "plain2.00-e3": .1379, "sc2.20-e3": .0968,
       "plain2.50-e3": .0770, "sc3.00-e4": .0326, "plain3.00-e4": .0323, "plain3.50-e4": .0185, "sc4.00-e4": .0095, "plain4.00-e4": .0086,
       "sc5.00-e5": .00277, "plain5.00-e5": .00305, "sc6.00-e6": .00114, "plain6.00-e6": .00125}
for c, k in sorted(pub.items(), key=lambda x: -x[1]):
    X = sum(s_of(w) * opt[w][c][1] for w in opt if w != "output.weight")
    print(f"{c:13} public KLD {k:.5f}  X {X:.5f}  X/KLD {X/k:5.2f}")
