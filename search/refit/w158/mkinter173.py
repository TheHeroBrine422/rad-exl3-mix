#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkinter173.py MENU OUT.jsonl PICK25.json PICK40.json -- (w158, RULES-173-03b) the interaction test's 77 specs on golden-i173c:
the bases B25 / B40 (the M1m picks at 2.5 / 4.0), each pair's moves alone (a, b) and together (ab). A move sets one trunk weight to
K1 (B25) / K2 (B40); a weight the base already holds at or below that level is replaced by the same type one layer deeper
(recorded). a/b T4 (both sizes <= 5.0)."""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import menu as MN  # noqa: E402

PFX = "q27v3-635bac02"
PAIRS = [("p1", "ffn_pair_early", "blk.4.ffn_gate_up", "blk.4.ffn_down"), ("p2", "ffn_pair_mid", "blk.30.ffn_gate_up", "blk.30.ffn_down"),
         ("p3", "ffn_pair_late", "blk.58.ffn_gate_up", "blk.58.ffn_down"), ("p4", "gdn_input_early", "blk.5.ssm_qkv", "blk.5.ssm_z"),
         ("p5", "gdn_input_late", "blk.53.ssm_qkv", "blk.53.ssm_z"), ("p6", "attn_input_qk", "blk.7.attn_qg", "blk.7.attn_k"),
         ("p7", "attn_input_kv", "blk.39.attn_k", "blk.39.attn_v"), ("p8", "adjacent", "blk.20.ffn_down", "blk.21.ffn_down"),
         ("p9", "adjacent", "blk.44.ssm_qkv", "blk.45.ssm_qkv"), ("p10", "far", "blk.20.ffn_down", "blk.52.ffn_down"),
         ("p11", "far", "blk.8.ssm_qkv", "blk.44.ssm_qkv"), ("p12", "mixer_x_ffn", "blk.12.ssm_out", "blk.12.ffn_down"),
         ("p13", "attnout_x_next_ffn", "blk.27.attn_output", "blk.28.ffn_gate_up")]


def lv(k): return f"{PFX}-k{k:g}-mad"


def main(menu, out, p25, p40):
    M = MN.Menu.load(menu); T = set(MN.trunk_names()); rows = []

    def deeper(w):
        l, typ = re.match(r"blk\.(\d+)\.(\w+)\.weight", w).groups(); l = int(l)
        for l2 in range(l + 1, 64):
            c = f"blk.{l2}.{typ}.weight"
            if c in T: return c
        raise KeyError(w)

    def resolve(base, w, k):
        w0 = w = w + ".weight"
        while M.opt(w, base[w])["K"] <= k: w = deeper(w)
        return w, w0, M.opt(w, base[w])["K"]

    def add(tag, a, meta):
        mad = a[MN.HEAD][-1].isdigit() and a[MN.HEAD] != "u7"
        rows.append({"tag": f"w158i-{tag}", "assign": a, "ab": "t4", "act": True, "slices": ["i173c"],
                     "census": ["--expect-codebook", "mad"] if mad else ["--expect-head", "exl3i mul1 K* r16"], "meta": {"block": "I", **meta}})
    for size, pk, k, pairs in ((2.5, p25, 1, PAIRS), (4.0, p40, 2, PAIRS[:12])):
        base = json.load(open(pk)); bt = f"b{size:g}"
        add(f"{bt}-base", dict(base), {"size": size, "arm": "base", "pair": None})
        for pid, cat, wa, wb in pairs:
            (ra, a0, ka), (rb, b0, kb) = resolve(base, wa, k), resolve(base, wb, k)
            info = {"size": size, "pair": pid, "category": cat, "move_to": k, "a": ra, "b": rb, "a_from": ka, "b_from": kb,
                    "a_replaced": a0 if ra != a0 else None, "b_replaced": b0 if rb != b0 else None}
            for arm, ws in (("a", [ra]), ("b", [rb]), ("ab", [ra, rb])):
                x = dict(base)
                for w in ws: x[w] = lv(k)
                add(f"{bt}-{pid}-{arm}", x, {**info, "arm": arm})
    assert len(rows) == 77, len(rows)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    rep = [r["meta"] for r in rows if r["meta"]["arm"] == "ab"]
    print(f"{out}: {len(rows)} specs; replaced moves: {[(m['size'], m['pair'], m['a_replaced'], m['b_replaced']) for m in rep if m['a_replaced'] or m['b_replaced']]}")
    for m in rep: print(f"  {m['size']:g} {m['pair']:4s} {m['a']} (K{m['a_from']:g}) + {m['b']} (K{m['b_from']:g}) -> K{m['move_to']}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
