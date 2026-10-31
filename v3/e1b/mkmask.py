#!/usr/bin/env python3
"""mkmask.py MAP BITS FAMILIES LAYERS -- an EXL3_AQ_MASK string: BITS for the linears of FAMILIES (comma list, or
'all') in LAYERS ('all' or A-B, inclusive), '0' elsewhere. MAP is the hook's index -> N K mats log (issue order).
Families: attn_in (qg, k, v: they share one input), attn_out, gdn_in (ssm_qkv, ssm_z: one input), ssm_out, gate_up,
down, head. Layer l is full attention when l % 4 == 3 (Qwen3.8's 3 GDN : 1 attention), GDN otherwise."""
import sys

GDN = [("gdn_in", 10240, 5120), ("gdn_in", 6144, 5120), ("ssm_out", 5120, 6144), ("gate_up", 34816, 5120),
       ("down", 5120, 17408)]
ATT = ["attn_in", "attn_in", "attn_in", "attn_out", "gate_up", "down"]


def label(rows):
    """[(family, layer)] for the map's rows; checks the shapes it can."""
    out, i = [], 0
    for l in range(64):
        if l % 4 == 3:
            for f in ATT:
                out.append((f, l)); i += 1
        else:
            for f, n, k in GDN:
                assert (rows[i][1], rows[i][2]) == (n, k), (i, l, f, rows[i])
                out.append((f, l)); i += 1
    assert rows[i][1] == 248320 and i == len(rows) - 1, rows[i]
    out.append(("head", 64))
    for j, (f, l) in enumerate(out):
        if f == "gate_up": assert rows[j][1] == 34816 and rows[j][3] == 2
        if f == "down": assert rows[j][2] == 17408
    return out


def main():
    mp, bits, fams, lay = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    rows = [tuple(map(int, l.split())) for l in open(mp) if l.strip()]
    rows.sort()
    lab = label(rows)
    fs = None if fams == "all" else set(fams.split(","))
    a, b = (0, 64) if lay == "all" else map(int, lay.split("-"))
    print("".join(bits if (fs is None or f in fs) and a <= l <= b else "0" for f, l in lab))


if __name__ == "__main__":
    main()
