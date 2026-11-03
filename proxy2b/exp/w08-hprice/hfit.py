#!/usr/bin/env python3
"""hfit.py MODEL.json RATIO.jsonl K KL_BASE KL_H [--pred-total P] -- KN_HF (the oursH family factor of knap4's
KN_HRATIO) from one measured uniform-trunk swap at level K: KL_BASE = the one-sided ladder level K (every trunk
linear ours-K), KL_H = the same with every trunk linear its H_out quant (oursH-K), same head / embedding, one golden.
knap4's weight cost of a tensor is s eb (phi e / eb)^q, so the oursH option costs the ours one x (r HF)^q. Ignoring
the depth compounding (second order at these sizes), the trunk's predicted KLD moves from T = sum_w cost_w to
sum_w cost_w (r_w HF)^q, and the measured relative change maps onto the model's scale through the base's measured /
predicted total: HF solves  sum_w cost_w (r_w HF)^q - T = (KL_H / KL_BASE - 1) x P, P = the model's predicted total
KLD of the base mix (--pred-total; default T + c0 + the base embedding / head costs, the uniform-mix approximation).
Prints HF and the prediction with HF = 1 (pure two-sided pricing)."""
import json, math, sys
from collections import defaultdict

a = sys.argv[1:]
P_ = float(a[a.index("--pred-total") + 1]) if "--pred-total" in a else None
MJ, RJ, K, KB, KH = a[0], a[1], float(a[2]), float(a[3]), float(a[4])
md = json.load(open(MJ)); q, phi = md["q"], md["phi"]
HF2KN = {"self_attn.q_proj": "attn_qg", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "linear_attn.in_proj_qkv": "ssm_qkv", "linear_attn.in_proj_z": "ssm_z", "linear_attn.out_proj": "ssm_out",
         "mlp.gate_proj": "ffn_gate_up", "mlp.up_proj": "ffn_gate_up", "mlp.down_proj": "ffn_down"}
acc = defaultdict(list)
for l in open(RJ):
    d = json.loads(l); k = d["key"].split(".")
    if float(d["K"]) == K and k[2] == "layers": acc[f"blk.{k[3]}.{HF2KN['.'.join(k[4:6])]}.weight"].append(math.log(d["r"]))
R = {w: math.exp(sum(v) / len(v)) for w, v in acc.items()}
c = f"ours-K{K:g}"; T = 0.0; terms = []
for t in md["tensors"]:
    if t["w"] not in R or c not in t["opts"] or t["eb"] <= 0: continue
    e = t["opts"][c][1]; fam = t["family_of"].get(c); ph = math.exp(phi[fam]) if fam in phi else 1.0
    cw = t["s"] * t["eb"] * (ph * e / t["eb"]) ** q; T += cw; terms.append((cw, R[t["w"]]))
assert terms, f"no tensor has both {c} and a ratio at K{K:g}"
P = P_ if P_ is not None else T + md["c0"]
want = (KH / KB - 1) * P
f = lambda hf: sum(cw * (r * hf) ** q for cw, r in terms) - T - want
lo, hi = 1e-3, 1e3
for _ in range(200):
    mid = math.sqrt(lo * hi); lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
pred1 = (sum(cw * r ** q for cw, r in terms) - T) / P
print(json.dumps({"K": K, "tensors": len(terms), "trunk_cost": T, "pred_total": P, "measured_rel": KH / KB - 1,
                  "pred_rel_hf1": pred1, "KN_HF": math.sqrt(lo * hi)}))
