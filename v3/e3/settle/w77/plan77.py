#!/usr/bin/env python3
"""plan77.py BOX > jobs-BOX.txt -- q77.py job lists (../w77.md). BOX colo: H_B half 1 (b1 jobs), the gates, half the
scale jobs, the sweep jobs of ffn_down and of every other (tensor, sample) by a fixed split; thor: half 2 and the rest;
all (11-07 19:40 UTC, Thor down): every job on colo (both halves, both reduces; finished jobs are skipped by DONE).
Order in each file: H_B jobs, reduce, gates, scale jobs, finish, sweep jobs (by tensor size, largest first)."""
import sys

BOX = sys.argv[1]; halves = (1, 2) if BOX == "all" else (1,) if BOX == "colo" else (2,)
C = "code/v3/e3/settle/w77"
PY = "python3 -W ignore"
PRE = "PYTHONPATH=/data/work/w77/pylib " if BOX == "thor" else ""
ENV = ("HB_META=/data/work/w77/meta E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib "
       "SETTLE_CACHE=cache W77_SCALES=scales SETTLE_ORDER=rev W77_HA=hb/hA W77_HB1=hb/hB1 W77_HB2=hb/hB2")
TENSORS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
           "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
           "ffn_down": (2, 30, 63)}
LEVELS = [("i4", "0.125"), ("i4", "0.25"), ("i4", "0.5"), ("i4", "0.75"), ("i4", "1"), ("i4", "2"), ("i8", "1.5"), ("i8", "2"), ("i8", "2.5")]
SIG = ["0.0005", "0.001", "0.0025", "0.005", "0.01", "0.025", "0.05", "0.1", "0.2"]
SIZE = {"ffn_down": 3, "ssm_out": 1, "attn_output": 1}
pairs = [(t, l, s) for t, ls in TENSORS.items() for l in ls for s in (0, 1)]
# split: ffn_down all on colo (memory: ~14 GB a sweep job); the rest alternate by position, colo taking 3 of 5
if BOX == "all": mine = pairs
elif BOX == "thor": mine = [p for p in pairs if p[0] != "ffn_down" and pairs.index(p) % 5 >= 2]
else: mine = [p for p in pairs if p[0] == "ffn_down" or pairs.index(p) % 5 < 2]
out = []
for half in halves:
    for j in range(10):
        jid = f"b{half}j{j:02d}"
        out.append(f"hb-{jid}|-|5|hb/part/{jid}/DONE|{ENV} {PRE}{PY} {C}/hb77.py job hb {jid}")
for half in halves:
    deps = ",".join(f"hb/part/b{half}j{j:02d}/DONE" for j in range(10))
    out.append(f"reduce{half}|{deps}|8|hb/hB{half}/REDUCED|{ENV} {PY} {C}/hb77.py reduce hb {half} && touch hb/hB{half}/REDUCED")
if BOX != "thor":
    out.append(f"gate77|-|2|gate77.json|{ENV} {PY} {C}/gate77.py gate77.json")
    out.append(f"gate77k|-|2|gate77k.json|{ENV} GATE_EXTRA=0.125 {PY} {C}/gate77.py gate77k.json")
    for cb, K in (("i8", "2"), ("i4", "0.5")):
        out.append(f"selftest-{cb}-{K}|-|2|selftest-{cb}-{K}.json|{ENV} {PY} {C}/settle77.py selftest 11 attn_k 0 0.005 {cb} {K} > selftest-{cb}-{K}.json.part && mv selftest-{cb}-{K}.json.part selftest-{cb}-{K}.json")
for t, l, s in sorted(mine, key=lambda p: -SIZE.get(p[0], 0)):
    for cb, K in LEVELS:
        out.append(f"scale-{l}-{t}-s{s}-{cb}-K{K}|-|{6 if t == 'ffn_down' else 2}|scales/{l}-{t}-s{s}-{cb}-K{K}.json|"
                   f"{ENV} {PY} {C}/settle77.py scale {l} {t} {s} {cb} {K}")
out.append(f"finish|hb/hB1/REDUCED,hb/hB2/REDUCED|8|hb/hA/FINISHED|{ENV} {PY} {C}/hb77.py finish hb /data/challenge/hessians/calib > hb/finish.log && touch hb/hA/FINISHED")
for t, l, s in sorted(mine, key=lambda p: -SIZE.get(p[0], 0)):
    dep = ",".join(["hb/hA/FINISHED"] + [f"scales/{l}-{t}-s{s}-{cb}-K{K}.json" for cb, K in LEVELS])
    for sg in SIG:
        o = f"out/s{s}/{l}-{t}-d{sg}.jsonl"
        out.append(f"sweep-{l}-{t}-s{s}-d{sg}|{dep}|{14 if t == 'ffn_down' else 3}|{o}|mkdir -p out/s{s} && {ENV} {PY} {C}/settle77.py {l} {t} {s} {sg} {o}")
print("\n".join(out))
