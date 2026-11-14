#!/usr/bin/env python3
"""menu17403.py OUT.json --rel REL.jsonl... --donors DONORS.jsonl... --prices PRICES.json --legacy MENU.json [--spot SPOT.json]
-- (w157, #174-03) the embedding block of the V3 search menu (RULES.md §6): one `token_embd.weight` tensor entry, the
form `fin133.py menu --emb-menu` and menu.py read (it replaces the trunk menu's embedding entry):
  o      t<K> (mul1i8 trellis, t-i8 levels: K1.5-5.0 in eighths, K5.5-7 in halves) or t<K>m (plain mul1: K6.5 / 7 / 8)
  bytes  the donor's token_embd.weight entry, every plane (DONORS: embcmp157.py lines, or slim157.jsonl's, whose slim
         entry must equal the full donor's); checked against the layout
         158,924,800 K + 755,200 (record (248320 / 16) (5120 / 16) 8K u32 + 1-bit codes plane + suh + svh)
  K      bytes * 8 / (248320 * 5120)
  err    w135's floor model (PRICES floor_model, embprices174.json): F + a rel, rel = the level's measured row error
         (REL: embsrc.py lines, w144's half steps + w157's eighths; identity Hessian, all 248320 rows)
  src    {"path": the slim or full donor}: radmix takes token_embd.weight from it (menu.py eval_mix: every option names
         its donor, so a mix never falls back to the base container's table)
Legacy options (LEGACY's Lloyd base / e2..e7 and bf16, their err as there) are kept only when on the lower convex hull
of (bytes, err) over all options together. --spot: the spot check's summary (spot157.py), copied into the notes.
Prints the table and the hull."""
import argparse, json, os

N = 248320 * 5120
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--rel", nargs="+", required=True); ap.add_argument("--donors", nargs="+", required=True)
ap.add_argument("--prices", required=True); ap.add_argument("--legacy", required=True); ap.add_argument("--spot")
a = ap.parse_args()

fm = json.load(open(a.prices))["floor_model"]; F, A = fm["F"], fm["a"]
rel = {}
for p in a.rel:
    for l in open(p):
        r = json.loads(l); lv = r["level"].rstrip("/").split("/")          # .../trellis/<name>/K<k>
        if lv[-2] in ("t-i8", "t-m1") and "/rerun/" not in r["level"]: rel[(lv[-2], float(lv[-1][1:]))] = r["rel"]
don = {}
for p in a.donors:
  for l in open(p):
    j = json.loads(l); r = j["cmp"]["a"] if "cmp" in j else j["a"]      # slim157.jsonl lines carry the compare
    if "cmp" in j and not j["cmp"].get("identical"): raise SystemExit(f"{r['file']}: slim token_embd differs from its full donor")
    b = os.path.basename(r["file"]).removesuffix(".part")                # emb-t-<i8|m1>-K<k>.rad
    name, k = b[len("emb-"):].rsplit("-K", 1); don[(name, float(k[:-len(".rad")]))] = dict(r, file=b)

want = [("t-i8", k / 8) for k in range(12, 41)] + [("t-i8", k / 2) for k in (11, 12, 13, 14)] + [("t-m1", k) for k in (6.5, 7, 8)]
opts, missing = [], []
for name, k in want:
    if (name, k) not in rel or (name, k) not in don: missing.append(f"{name} K{k:g}"); continue
    d = don[(name, k)]; by = d["bytes"]
    assert by == round(158_924_800 * k) + 755_200, (name, k, by)
    cb = "mul1i8" if name == "t-i8" else "mul1"
    opts.append({"o": f"t{k:g}" + ("" if cb == "mul1i8" else "m"), "K": round(by * 8 / N, 6), "bytes": by,
                 "err": F + A * rel[(name, k)], "rel": rel[(name, k)], "fam": "emb", "enc": f"trellis-exl3i-{cb}-k{k:g}",
                 "src": {"path": f"/data/q27/emb/trellis/{os.path.basename(d['file'])}"}})
if missing: raise SystemExit(f"missing rel or donor: {', '.join(missing)}")

L = json.load(open(a.legacy)); le = next(t for t in L["tensors"] if t["w"] == "token_embd.weight")
legacy = [dict(o, legacy=True) for o in le["opts"] if not str(o["o"]).startswith("t")]
allp = sorted(opts + legacy, key=lambda o: (o["bytes"], o["err"]))
hull = []                                                                   # lower convex hull, bytes up / err down
for o in allp:
    if hull and o["err"] >= hull[-1]["err"]: continue
    while len(hull) >= 2:
        (x1, y1), (x2, y2) = (hull[-2]["bytes"], hull[-2]["err"]), (hull[-1]["bytes"], hull[-1]["err"])
        if (y2 - y1) * (o["bytes"] - x1) >= (o["err"] - y1) * (x2 - x1): hull.pop()   # hull[-1] on or above the chord
        else: break
    hull.append(o)
on = {o["o"] for o in hull}
kept = [o for o in legacy if o["o"] in on]
for o in kept: o.pop("legacy", None)
emb = {"w": "token_embd.weight", "layer": -1, "group": "emb", "hbar": None,
       "opts": sorted(opts + kept, key=lambda o: o["bytes"])}
notes = [f"w157 #174-03: trellis embedding options (exl3dense embed_exl3i, from the stored record): t<K> mul1i8 "
         f"K1.5-5.0 in eighths + K5.5-7 halves, t<K>m mul1 K6.5 / 7 / 8; bytes = the donor's entry; err = embprices174 "
         f"floor model F {F:.6g} + a {A:.6g} x rel. Legacy kept (on the hull): {[o['o'] for o in kept] or 'none'}; dropped: "
         f"{[o['o'] for o in legacy if o['o'] not in on]}. No 'base' option: every mix names its embedding donor."]
for o in kept:
    if o["o"] == "bf16" and o["err"] < F:
        notes.append(f"bf16 is on the hull only through its menu174 err {o['err']:.3g}, below the floor model's intercept F "
                     f"{F:.4g} (the model's price at rel 0); w144 measured t8m vs bf16 -0.31 % +- 0.32 (A8, s171b): equal. "
                     f"Its slope vs t7m is ~{(next(x for x in opts if x['o'] == 't7m')['err'] - o['err']) / (o['bytes'] - next(x for x in opts if x['o'] == 't7m')['bytes']):.2g} "
                     "err a byte, below any trunk move's; its src is ds8's /data/q27/emb/emb-bf16.rad")
if a.spot: notes.append("spot check (RULES §5): " + json.dumps(json.load(open(a.spot)).get("summary")))
out = {"tensors": [emb], "notes": notes,
       "built": {"by": "search/v3/emb17403/menu17403.py (w157)", "rel": a.rel, "donors": a.donors, "prices": a.prices,
                 "legacy": a.legacy, "hull": [o["o"] for o in hull]}}
json.dump(out, open(a.out, "w"), indent=1)
for o in emb["opts"]:
    print(f'{o["o"]:>7} K {o["K"]:7.4f} bytes {o["bytes"]:>13,} err {o["err"]:.6f} rel {o.get("rel", float("nan")):.6f}'
          f'{"  hull" if o["o"] in on else ""}  {o["enc"]}')
print("hull:", " ".join(o["o"] for o in hull))
print("dropped legacy:", [o["o"] for o in legacy if o["o"] not in on])
