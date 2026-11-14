#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkmenu173.py OUT.json --levels LEVELDIR... --packs PACKDIR --diag LEVELDIR --heads HCURVE151.json --head-dir DIR
       --emb-menu MENU-EMB17403.json --fixed-from MIX.rad [--ab-t4 DONOR_K4.rad --ab-t5 DONOR_K5.rad]
-- (w158, #173-03) the M1 menu over the mad ladder (RUNBOOK-overnight-1113 MENU CHECKLIST). Runs on ds8 (numpy only).

Trunk: one option per mad level dir (rxrun ladder root/<level>: level/proxy.jsonl; final/ may be moved): bytes = the
  weight's entries in the level's pack (PACKDIR/<level>[-s<scales8>].rad: the container bytes a mix clones), err = sum over
  its HF parts of proxy_err x R (R from --diag's out/*/diag files, diagsum.py; mkmenu.py's err), fam "v3mad".
Head: every plan of hcurve151 except its i8 record (`i8_plans` is a separate key; the 45 mad u<K> + the 17 mul1 u<K>m / u7),
  err = the plan's head-only KL, bytes = the served donor's output.weight entry (HEAD_DIR/hd-<plan>.rad) when built, else
  K x vocab x 5120 / 8 + the overhead measured on the built donors; src = HEAD_DIR/hd-<plan>.rad.
Embedding: the options of --emb-menu (w157's block: 36 trellis + bf16), as given.
fixed_bytes: MIX.rad's container bytes minus its trunk / head / embedding entries (vision tower dropped in every mix;
  a/b = the T4 donor's 48 ssm_ab weights); "ab_t5_extra" = T5 - T4 ssm_ab bytes (sizes >= 5.5 add it). Sizes are
  container bytes x 8 / 26,895,998,464 (the board rule)."""
import argparse, collections, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import menu as MN       # noqa: E402
import diagsum as DS    # noqa: E402
import mkmenu as MK     # noqa: E402


def wbytes(p):
    tot, ent = MK.rad_entries(p); byw = collections.Counter()
    for e, b in ent.items():
        w = MK.weight_of(e)
        if w: byw[w] += b
    return tot, ent, byw


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mkmenu173.py")
    ap.add_argument("out"); ap.add_argument("--levels", nargs="+", required=True); ap.add_argument("--packs", required=True)
    ap.add_argument("--diag", required=True); ap.add_argument("--heads", required=True); ap.add_argument("--head-dir", required=True)
    ap.add_argument("--emb-menu", required=True); ap.add_argument("--fixed-from", required=True,
                    help="a census-checked mix .rad, or a previous menu173m JSON whose fixed_bytes / ab_t5_extra are kept")
    ap.add_argument("--ab-t4"); ap.add_argument("--ab-t5"); ap.add_argument("--vocab", type=int, default=248320)
    a = ap.parse_args(argv)
    trunk = MN.trunk_names(64)
    R, hb = {}, {}
    for p in sorted(glob.glob(f"{a.diag}/out/*/diag-*.safetensors")):
        s = DS.summarize(p); R[s["key"]] = s["R"]; w = MK.kname(s["key"])
        if w and s["E1"]: hb.setdefault(w, [0.0, 0.0]); hb[w][0] += s["E1"]; hb[w][1] += s["E2"]
    T = {w: {"w": w, "layer": MN.layer_of(w), "group": MN.group_of(w), "hbar": (hb[w][1] / hb[w][0]) if w in hb else None, "opts": []}
         for w in trunk}
    src = {"trunk_levels": [], "packs": []}
    for d in a.levels:
        name = os.path.basename(d.rstrip("/"))
        dn = json.load(open(f"{d}/DONE")); s8 = (dn.get("scales") or "")[:8]
        pk = f"{a.packs}/{name}" + (f"-s{s8}" if s8 else "") + ".rad"
        if not os.path.exists(pk): raise SystemExit(f"{name}: no pack {pk}")
        _, _, byw = wbytes(pk)
        acc = {}
        for l in open(f"{d}/level/proxy.jsonl"):
            r = json.loads(l); w = MK.kname(r["key"])
            if w is None: continue
            if r.get("codebook") != "mad": raise SystemExit(f"{name}: {r['key']} codebook {r.get('codebook')} (mad-only trunk)")
            if r["key"] not in R: raise SystemExit(f"no diag R for {r['key']}")
            x = acc.setdefault(w, {"o": name, "K": float(r["K"]), "bytes": int(byw[w]), "err": 0.0, "fam": "v3mad"})
            x["err"] += r["proxy_err"] * R[r["key"]]
        miss = [w for w in trunk if w not in acc or not acc[w]["bytes"]]
        if miss: raise SystemExit(f"{name}: {len(miss)} trunk weights missing, e.g. {miss[:3]}")
        for w, x in acc.items(): T[w]["opts"].append(x)
        src["trunk_levels"].append(name); src["packs"].append(pk)
    tens = [T[w] for w in trunk]
    hc = json.load(open(a.heads))
    plans = {k: v for k, v in hc["plans"].items() if v.get("codebook") in ("mad", "mul1")}
    built, ovh = {}, []
    for pn, p in plans.items():
        f = f"{a.head_dir}/hd-{pn}.rad"
        if os.path.exists(f):
            built[pn] = int(wbytes(f)[2][MN.HEAD]); ovh.append(built[pn] - a.vocab * 5120 * p["avg_bits"] / 8)
    oh = int(round(sum(ovh) / len(ovh))) if ovh else 2 ** 20
    hopts = [{"o": pn, "K": p["avg_bits"], "bytes": built.get(pn, int(a.vocab * 5120 * p["avg_bits"] / 8 + oh)), "err": p["kl"],
              "fam": "v3head-" + p["codebook"], "built": pn in built, "src": {"path": f"{a.head_dir}/hd-{pn}.rad"}} for pn, p in plans.items()]
    tens.append({"w": MN.HEAD, "layer": 64, "group": "head", "hbar": None, "opts": hopts})
    eb = [t for t in json.load(open(a.emb_menu))["tensors"] if t["w"] == MN.EMB][0]
    tens.append(eb)
    if a.fixed_from.endswith(".json"):
        pm = json.load(open(a.fixed_from)); fixed, vis, ab = pm["fixed_bytes"], pm["built"]["fixed_vision_bytes"], pm["built"]["ab_bytes"]
    else:
        tot, ent, byw = wbytes(a.fixed_from)
        vis = sum(b for e, b in ent.items() if e.startswith("v."))
        fixed = tot - sum(byw.get(w, 0) for w in trunk) - byw.get(MN.HEAD, 0) - byw.get(MN.EMB, 0)
        ab = {}
        for k, p in (("t4", a.ab_t4), ("t5", a.ab_t5)):
            if p: ab[k] = sum(b for e, b in MK.rad_entries(p)[1].items() if ".ssm_ab." in e)
    out = {"params": MN.PARAMS, "fixed_bytes": fixed, "n_layers": 64, "groups": MN.GROUPS, "tensors": tens,
           "ab_t5_extra": (ab["t5"] - ab["t4"]) if len(ab) == 2 else None,
           "built": {"by": "mkmenu173.py (w158, #173-03)", **src, "diag": a.diag, "heads": a.heads, "head_dir": a.head_dir,
                     "heads_built": sorted(built), "head_overhead": oh, "emb_menu": a.emb_menu, "fixed_from": a.fixed_from,
                     "fixed_vision_bytes": vis, "ab_bytes": ab}}
    json.dump(out, open(a.out + ".part", "w")); os.replace(a.out + ".part", a.out)
    M = MN.Menu(out)
    cnt = collections.Counter(o["fam"] for o in hopts)
    print(f"{a.out}: {len(tens)} tensors; trunk {len(a.levels)} mad levels K{min(x['K'] for x in tens[0]['opts']):g}-"
          f"{max(x['K'] for x in tens[0]['opts']):g}; head {len(hopts)} plans {dict(cnt)} ({len(built)} donors built, i8 plans not offered: "
          f"{len(hc.get('i8_plans', {}))}); embedding {len(eb['opts'])} options; fixed {fixed} B (vision in fixed: {vis} B), "
          f"ab T5 extra {out['ab_t5_extra']}")
    if vis: print("WARNING: the fixed-from mix holds vision-tower weights (they count in its container bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
