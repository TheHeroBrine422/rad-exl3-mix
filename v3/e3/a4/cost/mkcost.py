#!/usr/bin/env python3
"""mkcost.py TABLE.jsonl RESULTS.jsonl OUT.jsonl -- knap5a's KN_A4COST file from the #46-03 arms (README amendment 1).
Run from a knap tree's v3/nf (reads ../e1b/data/{aq.jsonl, aqfactors.json, e0-ident.map} and ../e1b/mkmask.py, as
knap5). Per measured cell (family, band, K, group): d = max(dKLD, 0.5 SE) x U, U = the board's V2 3.00 KLD (0.0306) /
V2 3.00's KLD on golden-search27 (RESULTS tag v2-300; knap5 predicts board-scale KLD); d is shared over the cell's
tensors in proportion to their E1c A4 cost (k_family x rel_out at that group, clip 1); a tensor's a4 cost = its E1c
a8g128 cost (what the baseline arm ran) + its share. One line per tensor x group x K: {"w", "group", "K", "cost"}."""
import collections, json, os, sys

sys.path.insert(0, "../e1b")
import mkmask

TAB, RES, OUT = sys.argv[1:4]
V2_BOARD_300 = 0.0306
res = {}
for l in open(RES):
    if l.strip().startswith("{"): r = json.loads(l); res[r["tag"]] = r
U = V2_BOARD_300 / res["v2-300"]["kld_b"] if "v2-300" in res else None
if U is None: print("WARNING: no v2-300 score: unit scale 1.0", file=sys.stderr); U = 1.0
kf = json.load(open("../e1b/data/aqfactors.json"))["factors"]
mp = sorted(tuple(map(int, l.split())) for l in open("../e1b/data/e0-ident.map") if l.strip())
lab = mkmask.label(mp); idx_name = {}
for L in range(64):
    ii = [i for i, (f, ll) in enumerate(lab) if ll == L]
    names = (["attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"] if L % 4 == 3 else
             ["ssm_qkv", "ssm_z", "ssm_out", "ffn_gate_up", "ffn_down"])
    for i, n in zip(ii, names): idx_name[i] = f"blk.{L}.{n}.weight"
AQ = collections.defaultdict(dict); FAMOF = {}
for l in open("../e1b/data/aq.jsonl"):
    r = json.loads(l)
    if abs(r["clip"] - 1.0) > 1e-6 or r["i"] not in idx_name: continue
    AQ[idx_name[r["i"]]][(r["bits"], r["group"])] = kf[r["fam"]] * r["rel_out"]; FAMOF[idx_name[r["i"]]] = (r["fam"], r["layer"])
FAMS = {"gate_up": "gate_up", "down": "down", "attn_in": "attn_in", "attn_out": "attn_out", "gdn_in": "gdn_in", "ssm_out": "ssm_out"}
n = 0
with open(OUT + ".part", "w") as f:
    for l in open(TAB):
        c = json.loads(l); a, b = map(int, c["band"].split("-")); g = c["group"]
        ts = [w for w, (fa, L) in FAMOF.items() if fa == FAMS[c["family"]] and a <= L <= b]
        assert ts, c
        wt = {w: AQ[w].get((4, g), 0.0) for w in ts}; s = sum(wt.values())
        d = max(c["diff"], 0.5 * c["se"]) * U
        for w in ts:
            f.write(json.dumps({"w": w, "group": g, "K": c["K"], "cost": AQ[w][(8, 128)] + d * (wt[w] / s if s > 0 else 1 / len(ts)),
                                "cell": f"{c['family']}/{c['band']}"}) + "\n"); n += 1
os.replace(OUT + ".part", OUT)
print(f"mkcost: {n} tensor lines, unit scale {U:.4f}", file=sys.stderr)
