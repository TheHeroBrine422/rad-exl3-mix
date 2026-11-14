#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkspecs173.py MENU.json OUT.jsonl WAVE -- (w158, #173-03) RULES-173-03's rows as run173.py specs (fit129).
WAVE 1: U 20 + G 48 + H 7 + E 7 + B 2 + F 4 (f16 twins) = 88; WAVE 2: S1 5 + S2 12 = 17 (needs the sub-1 levels in MENU).
Base mixes U-K: uniform level K, head u6.5, embedding t8m. a/b: t4 if the row's container bpw <= 5.25 else t5."""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import menu as MN  # noqa: E402

PFX = "q27v3-635bac02"
HEAD0, EMB0 = "u6.5", "t8m"


def lv(k): return f"{PFX}-k{k:g}-mad"


def main(menu, out, wave):
    M = MN.Menu.load(menu); T = MN.trunk_names(); ext = M.d.get("ab_t5_extra") or 0
    rows = []

    def add(tag, a, block, kind, meta, act=True):
        bpw = M.bpw(a); ab = "t4" if bpw <= 5.25 else "t5"
        cen = (["--expect-codebook", "mad"] if a[MN.HEAD].endswith(tuple("0123456789")) and a[MN.HEAD] != "u7" else
               ["--expect-head", "exl3i mul1 K* r16"])
        rows.append({"tag": f"w158-{tag}", "assign": a, "ab": ab, "act": act, "slices": ["fit129"], "census": cen,
                     "meta": {"block": block, "kind": kind, "bpw_menu": round(bpw + (ext * 8 / M.P if ab == "t5" else 0), 4), **meta}})

    def U(k, head=HEAD0, emb=EMB0): return M.uniform(lv(k), head, emb)

    def grp(base, donor, g, half):
        a = U(base); l0, l1 = (0, 31) if half == "lo" else (32, 63)
        sw = [w for w in T if MN.group_of(w) == g and l0 <= MN.layer_of(w) <= l1]
        for w in sw: a[w] = lv(donor)
        return a, len(sw)
    if wave == 1:
        for k in (1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5):
            add(f"u-k{k:g}", U(k), "U", "U", {"base": k})
        for base, donor in ((2.5, 1.5), (2.5, 3.5), (4, 3), (6, 5)):
            for g in MN.GROUPS:
                for half in ("lo", "hi"):
                    a, n = grp(base, donor, g, half)
                    add(f"g-b{base:g}-{g}-{half}-k{donor:g}", a, "G", "G", {"base": base, "donor": donor, "group": g, "half": half, "swapped": n})
        for base, heads in ((2.5, ("u1.5", "u2.5", "u4")), (4, ("u2.5", "u4")), (6, ("u4", "u5"))):
            for h in heads: add(f"h-b{base:g}-{h}", U(base, head=h), "H", "H", {"base": base, "head": h})
        for base, embs in ((2.5, ("t1.5", "t2.5", "t4")), (4, ("t3", "t4.5")), (6, ("t4.5", "t6"))):
            for e in embs: add(f"e-b{base:g}-{e}", U(base, emb=e), "E", "E", {"base": base, "emb": e})
        for g in ("ffn_gu", "ffn_down"):
            a = U(7)
            for w in T:
                if MN.group_of(w) == g: a[w] = lv(7.5)
            add(f"b-b7-{g}-k7.5", a, "B", "G", {"base": 7, "donor": 7.5, "group": g, "half": "all"})
        for k in (2.5, 4, 5, 6): add(f"f-u-k{k:g}", U(k), "F", "U", {"base": k, "mode": "f16"}, act=False)
        want = {"U": 20, "G": 48, "H": 7, "E": 7, "B": 2, "F": 4}
    else:
        for k in (0.5, 0.625, 0.75, 0.875, 1.125): add(f"s1-u-k{k:g}", U(k), "S1", "U", {"base": k})
        for g in MN.GROUPS:
            for half in ("lo", "hi"):
                a, n = grp(1, 0.625, g, half)
                add(f"s2-b1-{g}-{half}-k0.625", a, "S2", "G", {"base": 1, "donor": 0.625, "group": g, "half": half, "swapped": n})
        want = {"S1": 5, "S2": 12}
    n = collections.Counter(r["meta"]["block"] for r in rows)
    assert n == want, n
    assert len({r["tag"] for r in rows}) == len(rows)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{out}: {len(rows)} specs {dict(n)}; heads {sorted({r['assign'][MN.HEAD] for r in rows})}; "
          f"embs {sorted({r['assign'][MN.EMB] for r in rows})}; a/b {dict(collections.Counter(r['ab'] for r in rows))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
