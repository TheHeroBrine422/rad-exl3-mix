#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkspecs129.py DIAG_K2.5.jsonl OUT.jsonl -- #169's 84 calibration-swap mixes (92 scorings with the 8 rescores) as rxrun.multi eval lines
(search/refit/w129/RULES-prior.md "Data"). Every mix: V2 3.00's container (alias V2-3.00: head + embedding) with the
trunk given per weight ("levels"); scored on fit129, the 10 H mixes (8 group swaps + the K2.5 / K4 bases) on s129 too
(one mix, two slices). Each line also carries "w129": {kind U|G|P, base, donor, levels used, swapped weights}.
P (the prior test): per family the 8 highest / lowest hbar (E2 / E1 over the HF parts, the K2.5 diag sums) -> K1.5."""
import collections, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3"))
import menu as MN  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kfit3 import kname  # noqa: E402

PFX = "q27v3-144dd6e1"
GOLD = "/data/q27/w129/slices"
FIT = {"name": "fit129", "parts": {"c": f"{GOLD}/golden-fit129"}}
HELD = {"name": "s129", "parts": {"x": f"{GOLD}/golden-s129-x", "c": f"{GOLD}/golden-s129-c"}}
PASS1 = [1.25 + 0.25 * i for i in range(16)] + [5.5, 6.0, 6.5, 7.0]
GROUPS = MN.GROUPS
FAMS = [("attn_qg", 0, 63), ("attn_output", 0, 63), ("ssm_qkv", 0, 31), ("ssm_z", 32, 63), ("ssm_out", 0, 31),
        ("ffn_gate_up", 32, 63), ("ffn_down", 0, 31), ("ffn_down", 32, 63)]
HTAGS = ["g2.5-ffn_down-lo-k1.5", "g2.5-ffn_gu-hi-k1.5", "g2.5-ssm_in-lo-k4", "g2.5-attn_out-hi-k4", "g4-ffn_down-hi-k3",
         "g4-ssm_out-lo-k3", "g4-ffn_gu-lo-k5.5", "g4-attn_in-hi-k5.5"]


def fk(k): return f"{k:g}"
def lv(k): return f"{PFX}-k{fk(k)}-i8"


def hbar(diag):
    e1, e2 = collections.defaultdict(float), collections.defaultdict(float)
    for l in open(diag):
        r = json.loads(l); n = kname(r["key"])
        if n: e1[n] += r["E1"]; e2[n] += r["E2"]
    return {w: e2[w] / e1[w] for w in e1 if e1[w] > 0}


def main(diag, out):
    T = MN.trunk_names(); hb = hbar(diag); lines = []

    def line(tag, base, swap, donor, kind, extra=None, held=False):
        a = {w: lv(base) for w in T}
        for w in swap: a[w] = lv(donor)
        meta = {"kind": kind, "base": base, "donor": donor, "swapped": sorted(swap), **(extra or {})}
        lines.append({"tag": f"w129-{tag}", "mix": {"base": "V2-3.00", "levels": a},
                      "slices": [FIT] + ([HELD] if held else []), "w129": meta})
    for k in PASS1:
        line(f"u-k{fk(k)}", k, [], None, "U", held=k in (2.5, 4.0))
    for base, donors in ((2.5, (1.5, 4.0)), (4.0, (3.0, 5.5))):
        for g in GROUPS:
            for half, (l0, l1) in (("lo", (0, 31)), ("hi", (32, 63))):
                for d in donors:
                    sw = [w for w in T if MN.group_of(w) == g and l0 <= MN.layer_of(w) <= l1]
                    tag = f"g{fk(base)}-{g}-{half}-k{fk(d)}"
                    line(tag, base, sw, d, "G", {"group": g, "layers": [l0, l1]}, held=tag in HTAGS)
    for t, l0, l1 in FAMS:
        ws = [w for w in T if w.split(".")[2] == t and l0 <= MN.layer_of(w) <= l1]
        ws.sort(key=lambda w: hb[w])
        assert len(ws) >= 16, (t, len(ws))
        for arm, sw in (("hi", ws[-8:]), ("lo", ws[:8])):
            line(f"p-{t}-L{l0}-{l1}-{arm}", 2.5, sw, 1.5, "P", {"family": f"{t}-L{l0}-{l1}", "arm": arm,
                 "hbar": [hb[w] for w in sw], "hbar_family_median": sorted(hb[w] for w in ws)[len(ws) // 2]})
    assert len(lines) == 84 and sum(len(l["slices"]) == 2 for l in lines) == 10, (len(lines),)
    with open(out, "w") as f:
        for l in lines: f.write(json.dumps(l) + "\n")
    print(f"{out}: {len(lines)} mixes ({collections.Counter(l['w129']['kind'] for l in lines)}), "
          f"{sum(len(l['slices']) == 2 for l in lines)} also on s129")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
