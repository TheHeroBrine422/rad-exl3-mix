#!/usr/bin/env python3
"""mkratio.py MODEL.json KS [ROWERR2.jsonl] -- KN_HRATIO test files for the 27B knap4 model's trunk tensors at levels KS:
without ROWERR2, r = 1 everywhere (identity: with KN_HF=1 KN_HONLY=1 knap4 must give the same KLD / bytes as without
KN_HRATIO); with it, r = the 0.8B's two-sided ratio bdiag / base by tensor type at K2 / K3 (seed pairs 0-2), applied
to every 27B tensor of that type (illustration only: no 27B H_out exists yet)."""
import collections, json, math, sys
md = json.load(open(sys.argv[1])); KS = [float(k) for k in sys.argv[2].split(",")]
KN2HF = {"attn_qg": ["self_attn.q_proj"], "attn_k": ["self_attn.k_proj"], "attn_v": ["self_attn.v_proj"],
         "attn_output": ["self_attn.o_proj"], "ssm_qkv": ["linear_attn.in_proj_qkv"], "ssm_z": ["linear_attn.in_proj_z"],
         "ssm_out": ["linear_attn.out_proj"], "ffn_gate_up": ["mlp.gate_proj", "mlp.up_proj"], "ffn_down": ["mlp.down_proj"]}
TYPE = {}
if len(sys.argv) > 3:
    R = collections.defaultdict(dict)
    for l in open(sys.argv[3]):
        r = json.loads(l); R[r["arm"][2:]][r["key"]] = r["e2"]
    for K in (2, 3):
        acc = collections.defaultdict(list)
        for s in ("", "_s1", "_s2"):
            for k, v in R[f"k{K}bdiag{s}"].items(): acc[k.split(".", 4)[4]].append(math.log(v / R[f"k{K}base{s}"][k]))
        TYPE[K] = {t: math.exp(sum(v) / len(v)) for t, v in acc.items()}
for t in md["tensors"]:
    p = t["w"].split(".")
    if p[0] != "blk" or p[2] not in KN2HF: continue
    for hf in KN2HF[p[2]]:
        for K in KS:
            r = 1.0 if not TYPE else TYPE[2 if K < 2.5 else 3].get(hf)
            if r: print(json.dumps({"key": f"model.language_model.layers.{p[1]}.{hf}", "K": K, "r": r}))
