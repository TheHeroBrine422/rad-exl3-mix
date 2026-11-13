#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""embmenu.py IN_MENU.json OUT_MENU.json --inspect DONORS.jsonl [--donors DIR] [--levels e2,e4,...] [--prices OUT.json]
-- a search menu (mkmenu.py) with the embedding ladder the plugin serves from the stored form (w135, #174).

mkmenu.py gives the embedding one option, "base" (the base container's own: V2 3.00 = Lloyd l3). This replaces it with
one option per Lloyd width (embed_tq: a 2^B-level codebook under fwht128, a bf16 scale per 128 values; B 1..8) plus
bf16 (embed_lookup). l3 stays "base" (no donor, so every existing row keeps its option name); the others are named
e<B> / bf16 with src = DIR/emb-<name>.rad (emb174.sh: V2's l4 / l5 / l6 / bf16 taken from the V2 containers; embll.sh:
l2 / l7 converted with V2's rule). Everything else of the menu is copied unchanged.
  bytes = the donor's token_embd.weight entry (radinspect.py output, --inspect)
  err   = the encoding's floor P_0 in V2's kfit2 (calib-A scale): c0 + its emb level for the levels V2 fitted with
          enough data (e1..e6, bf16: data/v2/fit-final.txt); for e7 / e8 (V2's fit put them below bf16: noise) the
          linear floor model P_0 = F + a rel (rel = the level's relative error, v3/e3/data/embmenu2.jsonl; F, a through
          V2's l3 / l4 floors: F 0.000207 ~ V2's bf16 0.00019). fit.py ignores it; fit-embprior.patch uses
          err / err(base) as the prior P_0 ratio. Trellis levels (no gather kernel: not servable from the stored
          form) are priced the same way in --prices only, never put in the menu.
  K     = bits a value incl. the bf16 scale per 128 (l<B>: B + 1/8; bf16: 16)"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import menu as MN            # noqa: E402

V2_C0 = 0.00040                                                        # fit-final.txt: c0 (emb e4 = 0, the base level)
V2_EMB = {"e1": 0.01388, "e2": 0.00279, "e3": 0.00059, "e4": 0.0, "e5": -0.00010, "e6": -0.00015, "bf16": -0.00021}
REL_LLOYD = {1: 0.3606, 2: 0.11426, 3: 0.03158, 4: 0.00779, 5: 0.00207, 6: 0.00059, 7: 0.00016, 8: 0.00004}   # embmenu2.jsonl
A_REL = (V2_C0 + V2_EMB["e3"] - V2_C0) / (REL_LLOYD[3] - REL_LLOYD[4])
F_REL = V2_C0 - A_REL * REL_LLOYD[4]
NV, NE = 248320, 5120


def floor_err(lv):
    if lv in V2_EMB: return V2_C0 + V2_EMB[lv]
    return F_REL + A_REL * REL_LLOYD[int(lv[1:])]


def trellis_prices():
    """trellis embedding levels (embgpu.py: one bf16 scale per row, 20 trellis tiles of 256 a row): bytes at the stored
    rate and the floor-model err from the CPU curve (v3/e3/data/embmenu*.jsonl, group 5120)"""
    rel = {}
    for f in ("embmenu.jsonl", "embmenu2.jsonl"):
        p = os.path.join(HERE, "..", "..", "..", "v3", "e3", "data", f)
        for l in open(p):
            r = json.loads(l)
            if r["method"] == "trellis" and r.get("group") == 5120: rel[r["bits"]] = r["rel"]
    return {f"t{k:g}": {"K": k + 16 / NE, "bytes": int(NV * NE * (k + 16 / NE) / 8), "rel": rel[k], "err": F_REL + A_REL * rel[k]}
            for k in sorted(rel)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="embmenu.py")
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--inspect", required=True)
    ap.add_argument("--donors", default="/data/q27/emb"); ap.add_argument("--levels", default="e2,e4,e5,e6,e7,bf16")
    ap.add_argument("--prices"); ap.add_argument("--whatif-trellis", help="ALSO add these trellis levels (e.g. t2,t2.5): a "
                    "prediction-only menu (src = NOT-SERVABLE: no gather kernel, no donors); never load it into the driver")
    a = ap.parse_args(argv)
    d = json.load(open(a.inp))
    t = next(t for t in d["tensors"] if t["w"] == MN.EMB)
    old = {o["o"]: o for o in t["opts"]}
    if set(old) != {"base"}: raise SystemExit(f"{a.inp}: embedding options {sorted(old)}: expected the base container's only")
    ins = {}
    for l in open(a.inspect):
        r = json.loads(l); ins[r["src"]] = r
    opts = [{"o": "base", "K": 3.125, "bytes": old["base"]["bytes"], "err": floor_err("e3"), "fam": "emb", "enc": "lloyd-l3-g128-fwht128"}]
    for lv in a.levels.split(","):
        r = ins.get(f"emb-{lv}.rad")
        if r is None: raise SystemExit(f"no inspected donor emb-{lv}.rad")
        exp = NV * NE * 2 if lv == "bf16" else NV * NE * int(lv[1:]) // 8 + NV * NE // 128 * 2 + 4 * 2 ** int(lv[1:])
        if not (exp <= r["bytes"] <= exp + 4096): raise SystemExit(f"emb-{lv}.rad: token_embd {r['bytes']} bytes, expected ~{exp}")
        opts.append({"o": lv, "K": 16.0 if lv == "bf16" else int(lv[1:]) + 0.125, "bytes": r["bytes"], "err": floor_err(lv), "fam": "emb",
                     "enc": "bf16" if lv == "bf16" else f"lloyd-l{lv[1:]}-g128-fwht128", "src": {"path": f"{a.donors}/emb-{lv}.rad"}})
    if a.whatif_trellis:
        tp = trellis_prices()
        for n in a.whatif_trellis.split(","):
            q = tp[n]; opts.append({"o": n, "K": q["K"], "bytes": q["bytes"], "err": q["err"], "fam": "emb", "enc": f"trellis-{n}",
                                    "src": {"path": f"NOT-SERVABLE/{n}"}})
    if old["base"]["bytes"] != 496640032: raise SystemExit(f"base embedding {old['base']['bytes']} bytes: not V2 3.00's l3")
    t["opts"] = opts
    d.setdefault("built", {})["emb"] = {"by": "w135 embmenu.py (#174)", "from": a.inp, "donors": a.donors, "levels": a.levels}
    json.dump(d, open(a.out + ".part", "w")); os.replace(a.out + ".part", a.out)
    M = MN.Menu(d)
    eo = M.T[M.WI[MN.EMB]]["opts"]
    print(f"{a.out}: embedding options (name, bytes, bpw, err) " + ", ".join(f"{o['o']} {o['bytes']} {o['bytes'] * 8 / M.P:.4f} {o['err']:.6f}" for o in eo))
    lv = M.levels()
    for n in (lv[0], lv[len(lv) // 2], lv[-1]):
        print(f"  uniform {n}: " + ", ".join(f"{o['o']} {M.bpw(M.uniform(n, None, o['o'])):.3f}" for o in eo))
    if a.prices:
        json.dump({"floor_model": {"F": F_REL, "a": A_REL}, "lloyd": {o["o"]: {k: o[k] for k in ("K", "bytes", "err", "enc")} for o in eo},
                   "lloyd_rel": REL_LLOYD, "trellis_not_servable": trellis_prices()}, open(a.prices, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
