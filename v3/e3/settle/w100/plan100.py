#!/usr/bin/env python3
"""plan100.py BOX [COLO_EVERY] > jobs-BOX.txt -- q87.py job lists for w100 (../w100.md): one job = (mode, tensor,
sample), mode sub1 (#128) or hshape (#129); DONE = out/LAYER-TENSOR-sS-MODE.DONE; finished per-K files are skipped on a
rerun. colo (<= 8 GB of ours in total) takes every COLO_EVERY-th non-ffn_down job, Thor the rest (ffn_down first).
"all" = every job, smallest first (if Thor is gone: copy Thor's finished out/ files to colo, then run "all" there with
--maxbig 1 and nothing else running). Every job waits for gates/G1.ok (written by hand after the selftest passed)."""
import sys

TENSORS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
           "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
           "ffn_down": (2, 30, 63)}
EST = {"ffn_down": 14, "ssm_out": 1.5, "attn_output": 1.5}


def jobs():
    return [(m, t, l, s) for m in ("sub1", "hshape", "g125") for t, ls in TENSORS.items() for l in ls for s in (0, 1)]


def main():
    box = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    C = "code/v3/e3/settle/w100/settle100.py"
    big = lambda j: j[1] == "ffn_down"
    small = [j for j in jobs() if not big(j)]
    mine = {"colo": small[::every], "thor": [j for j in jobs() if big(j) or j not in small[::every]], "all": jobs()}[box]
    order = sorted(mine, key=lambda j: (not big(j), j[0] != "sub1")) if box == "thor" else sorted(mine, key=lambda j: (big(j), j[0] != "sub1"))
    out = []
    for m, t, l, s in order:
        done = f"out/{l}-{t}-s{s}-{m}.DONE"
        out.append(f"w100-{m}-{l}-{t}-s{s}|gates/G1.ok|{EST.get(t, 1.2)}|{done}|. ./env.sh; python3 -W ignore {C} {m} {l} {t} {s} out {done}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
