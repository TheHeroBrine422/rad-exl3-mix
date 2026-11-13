#!/usr/bin/env python3
"""menu17402.py MENU174 PRICES REL.jsonl INSPECT.jsonl OUT.json [--prices-out P.json] -- (w144, #174-02) w135's menu
(menu174.json) + the servable trellis embedding options: one option per quantized level of the token table (rel.jsonl
from quant144.sh / embsrc.py), served by exl3dense's embed_exl3i from the stored record (no NOT-SERVABLE options).
  o      t<K> (i8 grid, mul1i8) or t<K>m (plain mul1)
  bytes  the container entry's bytes (radinspect.py of the donor: record + suh + svh + the 1-byte-a-row codes plane)
  K      bytes * 8 / (248320 * 5120)
  err    w135's floor model (embprices174.json floor_model): F + a * rel, rel = the measured row error of the REAL table
         (sum (w - w_hat)^2 / sum w^2 over all 248320 rows; identity Hessian), the measure Lloyd's options are priced on
  src    the donor /data/q27/emb/trellis/emb-<level>-K<k>.rad (box2 and ds8: same path; radmix takes token_embd.weight)
Options keep menu174's order by bytes; "base" (V2 3.00's Lloyd l3) stays the start (init_emb base, as menu174)."""
import json, sys
menu_p, prices_p, rel_p, insp_p, out_p = sys.argv[1:6]
M = json.load(open(menu_p)); P = json.load(open(prices_p)); F, a = P["floor_model"]["F"], P["floor_model"]["a"]
N = 248320 * 5120
insp = {}
for l in open(insp_p):
    r = json.loads(l); insp[r["src"]] = r
rel = {}
for l in open(rel_p):
    r = json.loads(l); lv = r["level"].rstrip("/").split("/")          # .../trellis/<name>/K<k>
    rel[(lv[-2], lv[-1][1:])] = r["rel"]
emb = next(t for t in M["tensors"] if t["w"] == "token_embd.weight")
emb["opts"] = [o for o in emb["opts"] if not str(o["o"]).startswith("t")]
new = []
for (name, k), r in sorted(rel.items(), key=lambda x: (x[0][0], float(x[0][1]))):
    if name not in ("t-i8", "t-m1"): continue                       # t-i8-r16: the row-order test, not an option
    donor = f"/data/q27/emb/trellis/emb-{name}-K{k}.rad"
    if donor.split("/")[-1] not in insp: print(f"skip {name} K{k}: no donor inspected", file=sys.stderr); continue
    e = insp[donor.split("/")[-1]]
    cb = "mul1i8" if name == "t-i8" else "mul1"
    new.append({"o": f"t{k}" + ("" if cb == "mul1i8" else "m"), "K": round(e["bytes"] * 8 / N, 6), "bytes": e["bytes"],
                "err": F + a * r, "rel": r, "fam": "emb", "enc": f"trellis-exl3i-{cb}-k{k}", "src": {"path": donor}})
emb["opts"] = sorted(emb["opts"] + new, key=lambda o: o["bytes"])
M.setdefault("notes", []).append("w144 #174-02: trellis embedding options t<K> (exl3i, mul1i8) / t<K>m (mul1), served by "
                                 "exl3dense embed_exl3i from the stored record; err = embprices174 floor model on the measured rel")
json.dump(M, open(out_p, "w"))
for o in emb["opts"]: print(f'{o["o"]:>6} K {o["K"]:7.4f} bytes {o["bytes"]:>11,} err {o["err"]:.6f}', o.get("enc", ""))
