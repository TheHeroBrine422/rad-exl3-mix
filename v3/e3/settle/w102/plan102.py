#!/usr/bin/env python3
"""plan102.py BOX > jobs-BOX.txt -- q87.py job lists for #136-01 / -02 (../w102.md): w87's 27 tensors, sample 0.
b8 jobs first (one per tensor, K0.5 / 2.5 / 4, minutes), then d3 jobs (one per tensor x K: the NTRY-16 arm dominates).
Split: Thor = every ffn_down job + two thirds of the rest; colo = the other third (<= 8 GB of ours); "all" = every job
(colo can run all but ffn_down; finished files are skipped, so after Thor drops out, copy its finished out/ files to
colo and run "all" there without ffn_down: BOX = colo-all)."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w87"))
from plan87 import TENSORS  # noqa: E402

EST = {"ffn_down": 14, "ssm_out": 2.0, "attn_output": 2.0, "ffn_gate_up": 1.8}


def jobs():
    out = []
    for item, ks in (("b8", ["0.5,2.5,4"]), ("d3", ["0.125", "0.5", "1"])):
        for t, ls in TENSORS.items():
            for l in ls:
                for k in ks:
                    tag = f"{item}-{l}-{t}" + ("" if item == "b8" else f"-K{k}")
                    out.append((item, t, l, k, tag))
    return out


def main():
    box = sys.argv[1]   # colo | thor | all | colo-all
    C = "code/v3/e3/settle/w102"
    ENV = ("E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib SETTLE_CACHE=cache "
           "W77_SCALES=scales SETTLE_ORDER=rev W77_HA=hb/hA W77_HB1=hb/hB1 W77_HB2=hb/hB2 W87_HOUT=hout-bn")
    J = jobs(); big = [j for j in J if j[1] == "ffn_down"]; small = [j for j in J if j[1] != "ffn_down"]
    colo = small[::3]
    mine = {"colo": colo, "thor": big + [j for j in small if j not in colo], "all": J, "colo-all": small}[box]
    for item, t, l, k, tag in mine:
        kv = "W102_B8_KS" if item == "b8" else "W102_D3_KS"
        done = f"out/{item}/{tag}.DONE"
        print(f"{tag}|-|{EST.get(t, 1.5)}|{done}|{ENV} {kv}={k} python3 -W ignore {C}/job102.py {item} {l} {t} 0 out/{item} {done}")


if __name__ == "__main__":
    main()
