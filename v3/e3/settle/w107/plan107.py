#!/usr/bin/env python3
"""plan107.py BOX > jobs-BOX.txt -- q87.py job lists for w107 (../w107.md): g1 (gate G1, both boxes), drift-a (#141-01
cur at the new K + tab at the old K), ntry (#141-03, one job per tensor x K), drift-b (tab at the new K; waits for
gates/table.json, written by table107.py from both boxes' cur rows). Thor = every ffn_down job + two thirds of the rest,
colo = the other third (<= 8 GB of ours; drift-b of a tensor runs where its drift-a ran: it reads that cur file).
"all" = every job (if Thor is gone: copy its finished out/ files to colo first; ffn_down needs ~14 GB)."""
import sys

TENSORS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
           "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
           "ffn_down": (2, 30, 63)}
EST = {"ffn_down": 14, "ssm_out": 1.5, "attn_output": 1.5}
C = "code/v3/e3/settle/w107/settle107.py"
E100 = ("E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib SETTLE_CACHE=cache W77_SCALES=scales "
        "SETTLE_ORDER=rev W77_HA=hb/hA W77_HB1=hb/hB1 W77_HB2=hb/hB2 W87_HOUT=hout-bn W100_HF=hout-bn W107_TABLE=gates/table.json W107_S100=gates/s100.json")
E102 = ("E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib SETTLE_CACHE=cache W77_SCALES=scales "
        "SETTLE_ORDER=rev W77_HA=hb/hA W77_HB1=hb/hB1 W77_HB2=hb/hB2 W87_HOUT=hout-bn")


def units():
    u = [(t, l, s) for t, ls in TENSORS.items() for l in ls for s in (0, 1)]
    return u


def main():
    box = sys.argv[1]
    U = units(); small = [x for x in U if x[0] != "ffn_down"]; colo = set(small[::3])
    mine = {"colo": lambda x: x in colo, "thor": lambda x: x not in colo, "all": lambda x: True}[box]
    big = lambda t: t == "ffn_down"
    out = []
    for t, l, s in (("attn_k", 11, 0), ("ssm_out", 10, 1)):
        d = f"out/g1-{l}-{t}-s{s}.jsonl"
        out.append(f"g1-{l}-{t}-s{s}|-|1.5|{d}|{E100} python3 -W ignore {C} g1 {l} {t} {s} out")
    order = sorted([x for x in U if mine(x)], key=lambda x: (not big(x[0])) if box == "thor" else big(x[0]))
    for t, l, s in order:
        d = f"out/{l}-{t}-s{s}-drift-a.DONE"
        out.append(f"da-{l}-{t}-s{s}|-|{EST.get(t, 1.2)}|{d}|{E100} python3 -W ignore {C} drift-a {l} {t} {s} out {d}")
    for t, l, s in order:
        if s != 0: continue
        for k in ("0.125", "0.5", "1"):
            if box == "colo" and big(t): continue
            if box == "thor" and (t, l, 0) in colo: continue
            d = f"out/{l}-{t}-s0-K{k}-ntry.DONE"
            out.append(f"nt-{l}-{t}-K{k}|-|{EST.get(t, 1.5)}|{d}|{E102} W107_NTRY_KS={k} python3 -W ignore {C} ntry {l} {t} 0 out {d}")
    for t, l, s in order:
        d = f"out/{l}-{t}-s{s}-drift-b.DONE"
        out.append(f"db-{l}-{t}-s{s}|gates/table.json|{EST.get(t, 1.2)}|{d}|{E100} python3 -W ignore {C} drift-b {l} {t} {s} out {d}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
